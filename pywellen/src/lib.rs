use either::Either;
use pyo3::exceptions::PyKeyError;
use pyo3::types::PyInt;
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use rustc_hash::FxHashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, RwLock};
use wellen::{LoadOptions, ScopeOrVarRef, ScopeType, SignalValueRef, TimeTableIdx};

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
    m.add_class::<VarIter>()?;
    m.add_class::<Waveform>()?;
    m.add_class::<Signal>()?;
    m.add_class::<SignalChangeIter>()?;
    m.add_class::<Timescale>()?;
    m.add_class::<TimescaleUnit>()?;
    Ok(())
}

// #[pyclass(from_py_object)]
// #[derive(Clone)]
// struct Hierarchy(Arc<wellen::Hierarchy>);
//
// #[pymethods]
// impl Hierarchy {
//     fn top_scopes(&self) -> ScopeIter {
//         ScopeIter(Box::new({
//             let hier = self.0.clone();
//             hier.scopes()
//                 .map(|val| Scope {
//                     h: hier.clone(),
//                     id: val,
//                 })
//                 .collect::<Vec<_>>()
//                 .into_iter()
//         }))
//     }
//
//     fn all_vars(&self) -> Vec<Var> {
//         todo!()
//         //self.0.all_vars().map(|v| Var { h: self.0.clone(), id: v }).collect()
//     }
//
//     /// Get the date metadata from the waveform file
//     fn date(&self) -> String {
//         self.0.date().to_string()
//     }
//
//     /// Get the version metadata from the waveform file
//     fn version(&self) -> String {
//         self.0.version().to_string()
//     }
//
//     /// Get the timescale metadata from the waveform file
//     fn timescale(&self) -> Option<Timescale> {
//         self.0.timescale().map(Timescale)
//     }
//
//     /// Get the file format of the waveform file
//     fn file_format(&self) -> String {
//         match self.0.file_format() {
//             wellen::FileFormat::Vcd => "VCD".to_string(),
//             wellen::FileFormat::Fst => "FST".to_string(),
//             wellen::FileFormat::Ghw => "GHW".to_string(),
//             wellen::FileFormat::Unknown => "Unknown".to_string(),
//         }
//     }
// }

#[pyclass]
struct Scope {
    waves: Arc<SharedWaves>,
    id: wellen::ScopeRef,
}

#[pymethods]
impl Scope {
    #[getter]
    pub fn name(&self) -> String {
        self.h()[self.id].name(&self.h()).to_string()
    }

    #[getter]
    pub fn full_name(&self) -> String {
        self.h()[self.id].full_name(&self.h()).to_string()
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

    pub fn vars(&self) -> VarIter {
        //TODO: optimize me! need to rewrite the logic from `HierarchyItemIdIterator` to use
        // Arc<Hierarchy> instead of lifetimes
        //
        // This is because python does not like lifetimes :)
        let hier = self.h();
        let scope_id = self.id;
        VarIter(Box::new({
            hier[scope_id]
                .vars(&hier)
                .map(|val| Var {
                    waves: self.waves.clone(),
                    id: val,
                })
                .collect::<Vec<_>>()
                .into_iter()
        }))
    }

    pub fn scopes(&self) -> ScopeIter {
        //TODO: optimize me! need to rewrite the logic from `HierarchyItemIdIterator` to use
        // Arc<Hierarchy> instead of lifetimes
        //
        // This is because python does not like lifetimes :)
        let hier = self.h();
        let scope_id = self.id;
        ScopeIter(Box::new({
            hier[scope_id]
                .scopes(&hier)
                .map(|val| Scope {
                    waves: self.waves.clone(),
                    id: val,
                })
                .collect::<Vec<_>>()
                .into_iter()
        }))
    }

    /// Access a scope or variable.
    fn __getitem__(&self, name: &str) -> PyResult<Either<Var, Scope>> {
        let maybe_item = self.h().lookup_item_in_scope_by_name(self.id, name);
        return_item(&self.waves, name, maybe_item)
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
    maybe_item: Option<wellen::ScopeOrVarRef>,
) -> PyResult<Either<Var, Scope>> {
    if let Some(item) = maybe_item {
        match item {
            ScopeOrVarRef::Scope(id) => Ok(Either::Right(Scope {
                waves: waves.clone(),
                id,
            })),
            ScopeOrVarRef::Var(id) => Ok(Either::Left(Var {
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
struct ScopeIter(Box<dyn Iterator<Item = Scope> + Send + Sync>);
#[pymethods]
impl ScopeIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<Scope> {
        slf.0.next()
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
        format!("{:?}", self.h()[self.id].var_type()).to_lowercase()
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

#[pyclass]
struct VarIter(Box<dyn Iterator<Item = Var> + Send + Sync>);

#[pymethods]
impl VarIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<Var> {
        slf.0.next()
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

#[pyclass(from_py_object)]
#[derive(Clone)]
struct Signal {
    signal: Arc<wellen::Signal>,
    time_table: Arc<wellen::TimeTable>,
}

#[pymethods]
impl Signal {
    pub fn value_at_time<'a>(
        &self,
        time: wellen::Time,
        py: Python<'a>,
    ) -> Option<Bound<'a, PyAny>> {
        let val = self
            .time_table
            .as_ref()
            .binary_search(&time)
            .unwrap_or_else(|val| val);
        self.value_at_idx(val as TimeTableIdx, py)
    }

    pub fn value_at_idx<'a>(&self, idx: TimeTableIdx, py: Python<'a>) -> Option<Bound<'a, PyAny>> {
        let maybe_signal = self
            .signal
            .get_offset(idx)
            .map(|data_offset| self.signal.get_value_at(&data_offset, 0));
        if let Some(signal) = maybe_signal {
            let output = match signal {
                SignalValueRef::Real(inner) => Some(inner.into_pyobject(py).unwrap().into_any()),
                SignalValueRef::String(str) => Some(str.into_pyobject(py).unwrap().into_any()),
                _ => todo!(),
                // _ => match BigUint::try_from_signal(signal) {
                //     // If this signal is 2bits, this function will return an int
                //     Some(number) => Some(number.into_pyobject(py).unwrap().into_any()),
                //     // if this signal is not 2bits (e.g. it contains z,x, etc) then this function
                //     // will return a string
                //     None => signal
                //         .to_bit_string()
                //         .map(|val| val.into_pyobject(py).unwrap().into_any()),
                // },
            };
            output
        } else {
            None
        }
    }

    pub fn all_changes(&self) -> SignalChangeIter {
        SignalChangeIter {
            signal: self.clone(),
            offset: 0,
        }
    }
}

#[pyclass]
/// Iterates across all changes -- the returned object is a tuple of (Time, Value)
struct SignalChangeIter {
    signal: Signal,
    offset: usize,
}

#[pymethods]
impl SignalChangeIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__<'a>(
        mut slf: PyRefMut<'_, Self>,
        python: Python<'a>,
    ) -> Option<(wellen::Time, Bound<'a, PyAny>)> {
        if let Some(time_idx) = slf.signal.signal.time_indices().get(slf.offset) {
            let data = slf.signal.value_at_idx(*time_idx, python);
            let time = slf.signal.time_table.get(*time_idx as usize).cloned()?;
            slf.offset += 1;
            data.map(|val| (time, val))
        } else {
            None
        }
    }
}
