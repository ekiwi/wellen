use either::Either;
use pyo3::exceptions::{PyIndexError, PyKeyError};
use pyo3::types::PyInt;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use rustc_hash::FxHashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, RwLock};
use wellen::{ItemRef, LoadOptions, ScopeType};

pub trait PyErrExt<T> {
    fn toerr(self) -> PyResult<T>;
}

impl<T> PyErrExt<T> for wellen::Result<T> {
    fn toerr(self) -> PyResult<T> {
        self.map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

#[pymodule]
fn pywellen(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Var>()?;
    m.add_class::<Waveform>()?;
    m.add_class::<Signal>()?;
    m.add_class::<Timescale>()?;
    m.add_class::<TimescaleUnit>()?;
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
            ScopeType::Module => "module",
            ScopeType::Task => "task",
            ScopeType::Function => "function",
            ScopeType::Begin => "begin",
            ScopeType::Fork => "fork",
            ScopeType::Generate => "generate",
            ScopeType::Struct => "struct",
            ScopeType::Union => "union",
            ScopeType::Class => "class",
            ScopeType::Interface => "interface",
            ScopeType::Package => "package",
            ScopeType::Program => "program",
            ScopeType::VhdlArchitecture => "vhdl_architecture",
            ScopeType::VhdlProcedure => "vhdl_procedure",
            ScopeType::VhdlFunction => "vhdl_function",
            ScopeType::VhdlRecord => "vhdl_record",
            ScopeType::VhdlProcess => "vhdl_process",
            ScopeType::VhdlBlock => "vhdl_block",
            ScopeType::VhdlForGenerate => "vhdl_for_generate",
            ScopeType::VhdlIfGenerate => "vhdl_if_generate",
            ScopeType::VhdlGenerate => "vhdl_generate",
            ScopeType::VhdlPackage => "vhdl_package",
            ScopeType::GhwGeneric => "ghw_generic",
            ScopeType::VhdlArray => "vhdl_array",
            ScopeType::Unknown => "unknown",
            ScopeType::Clocking => "clocking",
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
            ItemRef::Scope(id) => Ok(Either::Right(Scope {
                waves: waves.clone(),
                id,
            })),
            ItemRef::Var(id) => Ok(Either::Left(Var {
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

#[pyclass]
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

    pub fn signal(&self) -> PyResult<Signal> {
        let signal_ref = self.h()[self.id].signal_ref();
        self.waves.get_signal(signal_ref)
    }

    /// for vcdvcd compatibility
    #[getter]
    pub fn tv(&self) -> PyResult<Signal> {
        self.signal()
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
struct Waveform {
    waves: Arc<SharedWaves>,
}

/// Private struct that contains waveform data structures that we need access to
/// from lots of places.
struct SharedWaves {
    wave_source: Mutex<wellen::SignalSource>,
    signals: RwLock<FxHashMap<wellen::SignalRef, Arc<wellen::Signal>>>,
    hierarchy: wellen::Hierarchy,
    /// the time table has a separate arc, since it is pointed to by each signal
    time_table: Arc<wellen::TimeTable>,
    multi_threaded: bool,
}

#[pymethods]
/// Top level waveform class that end users should use
/// The "egress" point from which all users can read waveforms
impl Waveform {
    #[new]
    #[pyo3(signature = (path, multi_threaded = true, remove_scopes_with_empty_name = true))]
    fn new(
        path: String,
        multi_threaded: bool,
        remove_scopes_with_empty_name: bool,
    ) -> PyResult<Self> {
        let opts = LoadOptions {
            multi_thread: multi_threaded,
            remove_scopes_with_empty_name,
        };
        let header_result = wellen::viewers::read_header_from_file(path.as_str(), &opts).toerr()?;

        let body = wellen::viewers::read_body(header_result.body, &header_result.hierarchy, None)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

        let waves = SharedWaves {
            wave_source: Mutex::new(body.source),
            signals: Default::default(),
            hierarchy: header_result.hierarchy,
            time_table: Arc::new(body.time_table),
            multi_threaded,
        };
        Ok(Self {
            waves: Arc::new(waves),
        })
    }

    /// Access a scope or variable.
    fn __getitem__(&self, name: &str) -> PyResult<Either<Var, Scope>> {
        let maybe_item = self.waves.hierarchy.lookup_item_by_name(name);
        return_item(&self.waves, name, maybe_item)
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
}

impl SharedWaves {
    fn get_signal(&self, signal_ref: wellen::SignalRef) -> PyResult<Signal> {
        if let Ok(s) = self.signals.read()
            && let Some(signal) = s.get(&signal_ref)
        {
            Ok(Signal {
                signal: signal.clone(),
                time_table: self.time_table.clone(),
            })
        } else if let Ok(mut waves) = self.wave_source.lock() {
            let mut res = waves.load_signals(&[signal_ref], &self.hierarchy, self.multi_threaded);
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

    fn __getitem__(&self, offset: isize) -> PyResult<(wellen::Time, String)> {
        if let Some(time_idx) = self.signal.time_indices().get(offset as usize) {
            let data = self.signal.data().get_value_at(offset as usize);
            let time = self.time_table[*time_idx as usize];
            let data = format!("{data}");
            Ok((time, data))
        } else {
            Err(PyIndexError::new_err(format!("out of bounds {offset}")))
        }
    }
}
