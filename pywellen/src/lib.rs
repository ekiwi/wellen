use either::Either;
use num_bigint::BigUint;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyNotImplementedError, PyTypeError};
use pyo3::types::{PyInt, PySlice, PySliceIndices};
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use rustc_hash::FxHashMap;
use smallvec::{SmallVec, smallvec};
use std::hash::{Hash, Hasher};
use std::num::NonZeroU32;
use std::ops::DerefMut;
use std::sync::{Arc, Mutex, RwLock};
use wellen::{Hierarchy, VarRef};

pub trait PyErrExt<T> {
    fn toerr(self) -> PyResult<T>;
}

impl<T> PyErrExt<T> for wellen::Result<T> {
    fn toerr(self) -> PyResult<T> {
        self.map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pymodule]
fn pywellen(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Var>()?;
    m.add_class::<Waveform>()?;
    m.add_class::<Signal>()?;
    m.add_class::<Timescale>()?;
    m.add_class::<TimescaleUnit>()?;
    m.add_function(wrap_pyfunction!(waveform_stream, m)?)?;
    Ok(())
}

#[pyclass]
struct Scope {
    waves: Arc<SharedWaves>,
    id: wellen::ScopeRef,
}

#[pymethods]
impl Scope {
    #[getter]
    pub fn name(&self) -> String {
        self.h()[self.id].name(self.h()).to_string()
    }

    #[getter]
    pub fn full_name(&self) -> String {
        self.h()[self.id].full_name(self.h()).to_string()
    }

    #[getter]
    pub fn scope_type(&self) -> String {
        match self.h()[self.id].scope_type() {
            wellen::ScopeType::Module => "module",
            wellen::ScopeType::Task => "task",
            wellen::ScopeType::Function => "function",
            wellen::ScopeType::Begin => "begin",
            wellen::ScopeType::Fork => "fork",
            wellen::ScopeType::Generate => "generate",
            wellen::ScopeType::Struct => "struct",
            wellen::ScopeType::Union => "union",
            wellen::ScopeType::Class => "class",
            wellen::ScopeType::Interface => "interface",
            wellen::ScopeType::Package => "package",
            wellen::ScopeType::Program => "program",
            wellen::ScopeType::VhdlArchitecture => "vhdl_architecture",
            wellen::ScopeType::VhdlProcedure => "vhdl_procedure",
            wellen::ScopeType::VhdlFunction => "vhdl_function",
            wellen::ScopeType::VhdlRecord => "vhdl_record",
            wellen::ScopeType::VhdlProcess => "vhdl_process",
            wellen::ScopeType::VhdlBlock => "vhdl_block",
            wellen::ScopeType::VhdlForGenerate => "vhdl_for_generate",
            wellen::ScopeType::VhdlIfGenerate => "vhdl_if_generate",
            wellen::ScopeType::VhdlGenerate => "vhdl_generate",
            wellen::ScopeType::VhdlPackage => "vhdl_package",
            wellen::ScopeType::GhwGeneric => "ghw_generic",
            wellen::ScopeType::VhdlArray => "vhdl_array",
            wellen::ScopeType::Unknown => "unknown",
            wellen::ScopeType::Clocking => "clocking",
            _ => "unknown", // `ScopeType` is marked as non-exhaustive
        }
        .to_string()
    }

    #[getter]
    #[pyo3(name = "type")]
    pub fn tpe(&self) -> String {
        self.scope_type()
    }

    pub fn vars(&self) -> Vec<Var> {
        self.h()[self.id]
            .vars(self.h())
            .map(|val| Var {
                waves: self.waves.clone(),
                id: val,
            })
            .collect()
    }

    pub fn scopes(&self) -> Vec<Scope> {
        self.h()[self.id]
            .scopes(self.h())
            .map(|val| Scope {
                waves: self.waves.clone(),
                id: val,
            })
            .collect()
    }

    /// Access a scope or variable.
    fn __getitem__(&self, name: &str) -> PyResult<Either<Var, Scope>> {
        let maybe_item = self.h().lookup_item_in_scope_by_name(self.id, name);
        return_item(&self.waves, name, maybe_item)
    }

    pub fn __eq__(&self, other: &Self) -> bool {
        self.id == other.id && std::ptr::eq(self.h(), other.h())
    }

    pub fn __hash__(&self) -> u64 {
        let mut hasher = std::hash::DefaultHasher::new();
        let pointer_num = (self.h() as *const wellen::Hierarchy) as u64;
        pointer_num.hash(&mut hasher);
        self.id.hash(&mut hasher);
        hasher.finish()
    }
}

impl Scope {
    #[inline]
    fn h(&self) -> &wellen::Hierarchy {
        &self.waves.hierarchy
    }
}

fn return_item(
    waves: &Arc<SharedWaves>,
    name: &str,
    maybe_item: Option<wellen::ItemRef>,
) -> PyResult<Either<Var, Scope>> {
    if let Some(item) = maybe_item {
        match item {
            wellen::ItemRef::Scope(id) => Ok(Either::Right(Scope {
                waves: waves.clone(),
                id,
            })),
            wellen::ItemRef::Var(id) => Ok(Either::Left(Var {
                waves: waves.clone(),
                id,
            })),
        }
    } else {
        let item_names: Vec<_> = waves
            .hierarchy
            .items()
            .map(|i| i.name(&waves.hierarchy).to_string())
            .collect();
        let error_msg = format!(
            "Failed to find `{name}`. Did you mean: {}",
            item_names.join(", ")
        );
        Err(PyKeyError::new_err(error_msg))
    }
}

#[derive(Clone)]
#[pyclass(from_py_object)]
struct SignalId(wellen::SignalRef);

#[pymethods]
impl SignalId {
    fn __str__(&self) -> String {
        format!("SignalId({})", self.0.index())
    }
}

#[derive(Clone)]
#[pyclass(from_py_object)]
struct Var {
    waves: Arc<SharedWaves>,
    id: wellen::VarRef,
}

#[pymethods]
impl Var {
    #[getter]
    pub fn name(&self) -> String {
        self.h()[self.id].name(&self.h()).to_string()
    }

    #[getter]
    pub fn full_name(&self) -> String {
        self.h()[self.id].full_name(&self.h()).to_string()
    }

    #[getter]
    pub fn bitwidth(&self) -> Option<u32> {
        self.h()[self.id].length(&self.h())
    }

    #[getter]
    pub fn size(&self) -> Option<u32> {
        self.h()[self.id].length(&self.h())
    }

    #[getter]
    pub fn var_type(&self) -> String {
        format!("{:?}", self.h()[self.id].var_type())
    }

    #[getter]
    #[pyo3(name = "type")]
    pub fn tpe(&self) -> String {
        self.var_type()
    }

    #[getter]
    pub fn enum_type(&self) -> Option<(String, Vec<(String, String)>)> {
        self.h()[self.id]
            .enum_type(&self.h())
            .map(|(name, values)| {
                (
                    name.to_string(),
                    values
                        .into_iter()
                        .map(|(k, v)| (k.to_string(), v.to_string()))
                        .collect(),
                )
            })
    }

    #[getter]
    pub fn vhdl_type_name(&self) -> Option<String> {
        self.h()[self.id]
            .vhdl_type_name(&self.h())
            .map(|s| s.to_string())
    }

    #[getter]
    pub fn direction(&self) -> String {
        format!("{:?}", self.h()[self.id].direction())
    }

    #[getter]
    pub fn length(&self) -> Option<u32> {
        self.h()[self.id].length(&self.h())
    }

    #[getter]
    pub fn is_real(&self) -> bool {
        self.h()[self.id].is_real(&self.h())
    }

    #[getter]
    pub fn is_string(&self) -> bool {
        self.h()[self.id].is_string(&self.h())
    }

    #[getter]
    pub fn is_bit_vector(&self) -> bool {
        self.h()[self.id].is_bit_vector(&self.h())
    }

    #[getter]
    pub fn is_1bit(&self) -> bool {
        self.h()[self.id].is_1bit(&self.h())
    }

    pub fn __eq__(&self, other: &Self) -> bool {
        self.id == other.id && std::ptr::eq(self.h(), other.h())
    }

    pub fn __hash__(&self) -> u64 {
        let mut hasher = std::hash::DefaultHasher::new();
        let pointer_num = (self.h() as *const wellen::Hierarchy) as u64;
        pointer_num.hash(&mut hasher);
        self.id.hash(&mut hasher);
        hasher.finish()
    }

    #[getter]
    pub fn signal_id(&self) -> SignalId {
        SignalId(self.waves.hierarchy[self.id].signal_ref())
    }

    #[getter]
    pub fn signal_ref(&self) -> SignalId {
        self.signal_id()
    }

    #[getter]
    pub fn signal(&self) -> PyResult<Signal> {
        let signal_ref = self.h()[self.id].signal_ref();
        self.waves.get_signal(signal_ref)
    }

    /// for vcdvcd compatibility
    #[getter]
    pub fn tv(&self) -> PyResult<Signal> {
        self.signal()
    }

    /// for vcdvcd compatibility
    pub fn __getitem__(&self, time: isize) -> PyResult<PySignalValue> {
        if let Some(value) = self.tv()?.value_at(time as wellen::Time) {
            Ok(value)
        } else {
            Err(PyIndexError::new_err(format!("No change yet at {time}")))
        }
    }
}

impl Var {
    #[inline]
    fn h(&self) -> &wellen::Hierarchy {
        &self.waves.hierarchy
    }
}

#[pyclass(from_py_object, name = "TimescaleUnit")]
#[derive(Clone)]
struct TimescaleUnit(pub(crate) wellen::TimescaleUnit);

#[pymethods]
impl TimescaleUnit {
    fn __str__(&self) -> String {
        match self.0 {
            wellen::TimescaleUnit::ZeptoSeconds => "zs".to_string(),
            wellen::TimescaleUnit::AttoSeconds => "as".to_string(),
            wellen::TimescaleUnit::FemtoSeconds => "fs".to_string(),
            wellen::TimescaleUnit::PicoSeconds => "ps".to_string(),
            wellen::TimescaleUnit::NanoSeconds => "ns".to_string(),
            wellen::TimescaleUnit::MicroSeconds => "us".to_string(),
            wellen::TimescaleUnit::MilliSeconds => "ms".to_string(),
            wellen::TimescaleUnit::Seconds => "s".to_string(),
            wellen::TimescaleUnit::Unknown => "unknown".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!("TimescaleUnit.{}", self.__str__())
    }

    fn to_exponent(&self) -> Option<i8> {
        self.0.to_exponent()
    }
}

#[pyclass(from_py_object, name = "Timescale")]
#[derive(Clone)]
struct Timescale(pub(crate) wellen::Timescale);

#[pymethods]
impl Timescale {
    #[getter]
    fn factor(&self) -> u32 {
        self.0.factor
    }

    #[getter]
    fn unit(&self) -> TimescaleUnit {
        TimescaleUnit(self.0.unit)
    }

    fn __str__(&self) -> String {
        format!("{}{}", self.0.factor, TimescaleUnit(self.0.unit).__str__())
    }

    fn __repr__(&self) -> String {
        format!(
            "Timescale(factor={}, unit={})",
            self.0.factor,
            TimescaleUnit(self.0.unit).__repr__()
        )
    }
}

#[pyclass(from_py_object)]
#[derive(Clone)]
struct TimeTable(Arc<wellen::TimeTable>);

/// Converts python index to a usize
/// e.g. in python a[-1] is a common way to get an obj from a list
fn convert_py_idx(idx: isize, len: usize) -> usize {
    if idx < 0 {
        (idx + len as isize) as usize
    } else {
        idx as usize
    }
}

#[pymethods]
impl TimeTable {
    fn __getitem__<'a>(&self, idx: isize, py: Python<'a>) -> PyResult<Option<Bound<'a, PyInt>>> {
        let len = self.0.len();
        let idx = convert_py_idx(idx, len);
        Ok(self
            .0
            .get(idx)
            .cloned()
            .map(|val| val.into_pyobject(py).unwrap()))
    }
}

#[pyclass]
pub struct Waveform {
    waves: Arc<SharedWaves>,
}

/// Private struct that contains waveform data structures that we need access to
/// from lots of places.
struct SharedWaves {
    body: Mutex<Option<BodyCont>>,
    stream: Mutex<Option<StreamWave>>,
    bulk: RwLock<Option<BulkData>>,
    hierarchy: wellen::Hierarchy,
    multi_threaded: bool,
    filename: String,
    stream_only: bool,
    opts: wellen::LoadOptions,
    signals_to_var: RwLock<SignalToVarMap>,
}

/// Waveform data that is only relevant if we are bulk parsing.
struct BulkData {
    signals: RwLock<FxHashMap<wellen::SignalRef, Arc<wellen::Signal>>>,
    /// the time table has a separate arc, since it is pointed to by each signal
    time_table: Arc<wellen::TimeTable>,
    wave_source: Mutex<wellen::SignalSource>,
}

type BodyCont = wellen::viewers::ReadBodyContinuation<std::io::BufReader<std::fs::File>>;
type StreamWave = wellen::stream::StreamingWaveform<std::io::BufReader<std::fs::File>>;

#[pyfunction]
#[pyo3(name = "WaveformStream")]
#[pyo3(signature = (path, multi_threaded = true, remove_scopes_with_empty_name = true))]
pub fn waveform_stream(
    path: String,
    multi_threaded: bool,
    remove_scopes_with_empty_name: bool,
) -> PyResult<Waveform> {
    Waveform::new(path, multi_threaded, remove_scopes_with_empty_name, true)
}

#[pymethods]
/// Top level waveform class that end users should use
/// The "egress" point from which all users can read waveforms
impl Waveform {
    #[new]
    #[pyo3(signature = (path, multi_threaded = true, remove_scopes_with_empty_name = true, stream_only = false))]
    fn new(
        path: String,
        multi_threaded: bool,
        remove_scopes_with_empty_name: bool,
        stream_only: bool,
    ) -> PyResult<Self> {
        let opts = wellen::LoadOptions {
            multi_thread: multi_threaded,
            remove_scopes_with_empty_name,
        };
        let header_result = wellen::viewers::read_header_from_file(path.as_str(), &opts).toerr()?;
        let signals_to_var = RwLock::new(SignalToVarMap::new(&header_result.hierarchy));
        let waves = SharedWaves {
            body: Mutex::new(Some(header_result.body)),
            bulk: RwLock::new(None),
            stream: Mutex::new(None),
            hierarchy: header_result.hierarchy,
            multi_threaded,
            filename: path,
            stream_only,
            opts,
            signals_to_var,
        };
        Ok(Self {
            waves: Arc::new(waves),
        })
    }

    /// Access a scope or variable.
    fn __getitem__(&self, key: Bound<'_, PyAny>) -> PyResult<Either<Either<Var, Scope>, Vec<Var>>> {
        if let Ok(name) = key.extract::<&str>() {
            let maybe_item = self.waves.hierarchy.lookup_item_by_name(name);
            return_item(&self.waves, name, maybe_item).map(Either::Left)
        } else if let Ok(signal) = key.extract::<SignalId>() {
            let vars = self
                .waves
                .signals_to_var
                .read()
                .unwrap()
                .get(signal.0)
                .into_iter()
                .map(|id| Var {
                    id,
                    waves: self.waves.clone(),
                })
                .collect();
            Ok(Either::Right(vars))
        } else {
            Err(PyTypeError::new_err(
                "Unsupported type. Expected str or SignalId.",
            ))
        }
    }

    /// All variables in the design.
    fn all_vars(&self) -> Vec<Var> {
        self.waves
            .hierarchy
            .all_vars()
            .map(|id| Var {
                waves: self.waves.clone(),
                id,
            })
            .collect()
    }

    /// Top-level variables.
    fn vars(&self) -> Vec<Var> {
        self.waves
            .hierarchy
            .vars()
            .map(|id| Var {
                waves: self.waves.clone(),
                id,
            })
            .collect()
    }

    /// All scopes in the design.
    fn all_scopes(&self) -> Vec<Scope> {
        self.waves
            .hierarchy
            .all_scopes()
            .map(|id| Scope {
                waves: self.waves.clone(),
                id,
            })
            .collect()
    }

    /// Top-level scopes.
    fn scopes(&self) -> Vec<Scope> {
        self.waves
            .hierarchy
            .scopes()
            .map(|id| Scope {
                waves: self.waves.clone(),
                id,
            })
            .collect()
    }

    /// Get the date metadata from the waveform file
    #[getter]
    fn date(&self) -> String {
        self.waves.hierarchy.date().to_string()
    }

    /// Get the version metadata from the waveform file
    #[getter]
    fn version(&self) -> String {
        self.waves.hierarchy.version().to_string()
    }

    /// Get the timescale metadata from the waveform file
    #[getter]
    fn timescale(&self) -> Option<Timescale> {
        self.waves.hierarchy.timescale().map(Timescale)
    }

    /// Get the file format of the waveform file
    #[getter]
    fn file_format(&self) -> String {
        match self.waves.hierarchy.file_format() {
            wellen::FileFormat::Vcd => "VCD".to_string(),
            wellen::FileFormat::Fst => "FST".to_string(),
            wellen::FileFormat::Ghw => "GHW".to_string(),
            wellen::FileFormat::Unknown => "Unknown".to_string(),
        }
    }

    fn stream_changes(
        &self,
        callback: &Bound<'_, PyAny>,
        include: Option<Vec<Either<Var, String>>>,
    ) -> PyResult<()> {
        self.waves.stream_changes(callback, include)
    }
}

struct SignalToVarMap {
    entries: Vec<Option<(wellen::VarRef, NonZeroU32)>>,
}

impl SignalToVarMap {
    const NO_NEXT: NonZeroU32 = NonZeroU32::MAX;

    fn new(h: &Hierarchy) -> Self {
        let num_signals = h.signals().last().map(|s| s.index()).unwrap_or(0);
        let entries = vec![None; num_signals];
        Self { entries }
    }

    fn contains(&self, signal: wellen::SignalRef) -> bool {
        self.entries[signal.index()].is_some()
    }

    fn insert(&mut self, signal: wellen::SignalRef, var: wellen::VarRef) {
        let mut index = signal.index();
        if self.entries[index].is_none() {
            self.entries[index] = Some((var, Self::NO_NEXT));
        } else {
            let mut entry = self.entries[index].unwrap();
            while entry.1 != Self::NO_NEXT {
                if entry.0 == var {
                    return; // do not insert the same variable twice
                }
                index += entry.1.get() as usize;
                entry = self.entries[index].unwrap();
            }
            let absolute_index = self.entries.len();
            let relative = absolute_index - index;
            self.entries[index] = Some((entry.0, NonZeroU32::new(relative as u32).unwrap()));
            self.entries.push(Some((var, Self::NO_NEXT)));
        }
    }

    fn get(&self, signal: wellen::SignalRef) -> SmallVec<[VarRef; 4]> {
        let mut out = smallvec![];
        let mut index = signal.index();
        while let Some((var, delta)) = self.entries[index] {
            out.push(var);
            if delta == Self::NO_NEXT {
                break;
            }
            index += delta.get() as usize;
        }
        out
    }
}

impl BulkData {
    fn get_signal(
        &self,
        hierarchy: &wellen::Hierarchy,
        multi_threaded: bool,
        signal_ref: wellen::SignalRef,
    ) -> PyResult<Signal> {
        if let Ok(s) = self.signals.read()
            && let Some(signal) = s.get(&signal_ref)
        {
            Ok(Signal {
                signal: signal.clone(),
                time_table: self.time_table.clone(),
            })
        } else if let Ok(mut waves) = self.wave_source.lock() {
            let mut res = waves.load_signals(&[signal_ref], hierarchy, multi_threaded);
            if let Some(wellen_signal) = res.pop() {
                debug_assert!(res.is_empty());
                let signal = Arc::new(wellen_signal);
                if let Ok(mut table) = self.signals.write() {
                    table.insert(signal_ref, signal.clone());
                }
                Ok(Signal {
                    signal: signal.clone(),
                    time_table: self.time_table.clone(),
                })
            } else {
                Err(PyRuntimeError::new_err("failed to load signal"))
            }
        } else {
            Err(PyRuntimeError::new_err("failed to acquire lock"))
        }
    }
}

impl SharedWaves {
    fn get_signal(&self, signal_ref: wellen::SignalRef) -> PyResult<Signal> {
        if self.stream_only {
            return Err(PyRuntimeError::new_err(
                "Cannot access signals directly, since the waveform is in stream only mode.",
            ));
        }
        if let Some(bulk) = self.bulk.read().unwrap().as_ref() {
            bulk.get_signal(&self.hierarchy, self.multi_threaded, signal_ref)
        } else {
            // load bulk data
            let body = self.body()?;
            let body = wellen::viewers::read_body(body, &self.hierarchy, None)
                .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

            let bulk = BulkData {
                signals: Default::default(),
                time_table: Arc::new(body.time_table),
                wave_source: Mutex::new(body.source),
            };
            let signal = bulk.get_signal(&self.hierarchy, self.multi_threaded, signal_ref);

            // store new bulk data
            *self.bulk.write().unwrap() = Some(bulk);

            // now we can return the (maybe) signal
            signal
        }
    }

    /// used internally to get the body continuation
    fn body(&self) -> PyResult<BodyCont> {
        let stolen = std::mem::take(self.body.lock().unwrap().deref_mut());
        if let Some(body) = stolen {
            Ok(body)
        } else {
            // reload the file to get another body continuation
            let header_result =
                wellen::viewers::read_header_from_file(self.filename.as_str(), &self.opts)
                    .toerr()?;
            Ok(header_result.body)
        }
    }

    fn stream_changes(
        &self,
        callback: &Bound<'_, PyAny>,
        include: Option<Vec<Either<Var, String>>>,
    ) -> PyResult<()> {
        // convert includes to a filter and a lookup for variable names
        let mut filter_vec = vec![];
        let filter = if let Some(vars) = include {
            filter_vec = vars_to_signals(
                &self.hierarchy,
                vars.into_iter(),
                &mut self.signals_to_var.write().unwrap(),
            );
            wellen::stream::Filter::include_signals(filter_vec.as_slice())
        } else {
            // TODO: add fast check to see if `signals_to_var` has already been populated
            add_all_vars(&self.hierarchy, &mut self.signals_to_var.write().unwrap());
            wellen::stream::Filter::all()
        };

        self.stream_changes_internal(callback, filter)
    }

    fn stream_changes_internal(
        &self,
        callback: &Bound<'_, PyAny>,
        filter: wellen::stream::Filter,
    ) -> PyResult<()> {
        if let Some(stream) = self.stream.lock().unwrap().as_mut() {
            stream
                .stream_changes(filter, |time, signal, value| {
                    let args = (time, SignalId(signal), to_py_signal_value(value));
                    callback.call1(args).map(|_| ())
                })
                .toerr()
        } else {
            let body = self.body()?;
            let wave: wellen::stream::StreamingWaveform<_> = (self.hierarchy.clone(), body).into();
            *self.stream.lock().unwrap() = Some(wave);
            self.stream_changes_internal(callback, filter)
        }
    }
}

fn add_all_vars(h: &Hierarchy, map: &mut SignalToVarMap) {
    for var in h.all_vars() {
        map.insert(h[var].signal_ref(), var);
    }
}

fn vars_to_signals(
    h: &Hierarchy,
    vars: impl Iterator<Item = Either<Var, String>>,
    map: &mut SignalToVarMap,
) -> Vec<wellen::SignalRef> {
    let mut out = vec![];
    for var in vars.flat_map(|v| resolve_var(h, v)) {
        let signal_ref = h[var].signal_ref();
        if !map.contains(signal_ref) {
            out.push(signal_ref);
        }
        map.insert(signal_ref, var);
    }
    out
}

fn resolve_var(h: &Hierarchy, var: Either<Var, String>) -> Option<wellen::VarRef> {
    match var {
        Either::Left(v) => Some(v.id),
        Either::Right(name) => h.lookup_item_by_name(&name).and_then(|i| {
            if let wellen::ItemRef::Var(v) = i {
                Some(v)
            } else {
                None
            }
        }),
    }
}

#[pyclass(sequence)]
struct Signal {
    signal: Arc<wellen::Signal>,
    time_table: Arc<wellen::TimeTable>,
}

#[pymethods]
impl Signal {
    fn __len__(&self) -> usize {
        self.signal.time_indices().len()
    }

    fn __getitem__(
        &self,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<Either<(wellen::Time, PySignalValue), Vec<(wellen::Time, PySignalValue)>>> {
        if let Ok(offset) = key.extract::<isize>() {
            self.get_offset(offset).map(Either::Left)
        } else if let Ok(slice) = key.cast::<PySlice>() {
            let len = self.signal.time_indices().len();
            let indices = slice.indices(len as isize)?;
            self.get_slice(indices).map(Either::Right)
        } else {
            Err(PyTypeError::new_err(
                "Unsupported type. Expected int or slice.",
            ))
        }
    }

    /// Returns `None` if the value at the given time is not known (because no change has been observed).
    fn value_at(&self, time: wellen::Time) -> Option<PySignalValue> {
        let time_idx = time_to_time_table_idx(&self.time_table, time)?;
        let offset = self.signal.get_offset(time_idx)?;
        let data = self.signal.get_value_at(&offset, offset.elements - 1);
        Some(to_py_signal_value(data))
    }
}

impl Signal {
    fn get_offset(&self, offset: isize) -> PyResult<(wellen::Time, PySignalValue)> {
        if let Some(time_idx) = self.signal.time_indices().get(offset as usize) {
            let data = self.signal.data().get_value_at(offset as usize);
            let time = self.time_table[*time_idx as usize];
            Ok((time, to_py_signal_value(data)))
        } else {
            Err(PyIndexError::new_err(format!("out of bounds {offset}")))
        }
    }

    fn get_slice(&self, slice: PySliceIndices) -> PyResult<Vec<(wellen::Time, PySignalValue)>> {
        if slice.start >= 0 && slice.stop > slice.start && slice.step == 1 {
            let range = (slice.start as usize)..(slice.stop as usize);
            let mut out = Vec::with_capacity(range.len());
            for offset in range {
                if let Some(time_idx) = self.signal.time_indices().get(offset) {
                    let data = self.signal.data().get_value_at(offset);
                    let time = self.time_table[*time_idx as usize];
                    out.push((time, to_py_signal_value(data)));
                } else {
                    return Err(PyIndexError::new_err(format!("out of bounds {offset}")));
                }
            }
            Ok(out)
        } else {
            Err(PyNotImplementedError::new_err(format!(
                "TODO: support slice {slice:?}"
            )))
        }
    }
}

type PySignalValue = Either<String, Either<BigUint, f64>>;
fn to_py_signal_value(value: wellen::SignalValueRef) -> PySignalValue {
    match value {
        wellen::SignalValueRef::Event => Either::Left("".to_string()),
        wellen::SignalValueRef::BitVec(bv) => {
            if let Some(bytes) = bv.be_bytes() {
                Either::Right(Either::Left(BigUint::from_bytes_be(bytes)))
            } else {
                Either::Left(bv.bit_string())
            }
        }
        wellen::SignalValueRef::String(s) => Either::Left(s.to_string()),
        wellen::SignalValueRef::Real(v) => Either::Right(Either::Right(v)),
    }
}

fn time_to_time_table_idx(
    time_table: &wellen::TimeTable,
    time: wellen::Time,
) -> Option<wellen::TimeTableIdx> {
    if time_table.is_empty() || time_table[0] > time {
        None
    } else {
        // binary search to find correct index
        let idx = binary_search(time_table, time);
        assert!(time_table[idx] <= time);
        Some(idx as wellen::TimeTableIdx)
    }
}

#[inline]
fn binary_search(times: &[wellen::Time], needle: wellen::Time) -> usize {
    let mut lower_idx = 0usize;
    let mut upper_idx = times.len() - 1;
    while lower_idx <= upper_idx {
        let mid_idx = lower_idx + ((upper_idx - lower_idx) / 2);

        match times[mid_idx].cmp(&needle) {
            std::cmp::Ordering::Less => {
                lower_idx = mid_idx + 1;
            }
            std::cmp::Ordering::Equal => {
                return mid_idx;
            }
            std::cmp::Ordering::Greater => {
                upper_idx = mid_idx - 1;
            }
        }
    }
    lower_idx - 1
}
