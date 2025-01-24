mod convert;
use std::sync::Arc;

use convert::Mappable;
use num_bigint::BigUint;
use pyo3::types::PyInt;
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use wellen::{
    viewers::{self},
    LoadOptions, SignalValue, TimeTableIdx,
};

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
    Ok(())
}

#[pyclass]
#[derive(Clone)]
struct Hierarchy(pub(crate) Arc<wellen::Hierarchy>);

#[pymethods]
impl Hierarchy {
    fn all_vars(&self) -> VarIter {
        VarIter(Box::new(
            //TODO: optimize me
            self.0
                .get_unique_signals_vars()
                .into_iter()
                .flatten()
                .map(Var),
        ))
    }

    fn top_scopes(&self) -> ScopeIter {
        ScopeIter(Box::new({
            let hier = self.0.clone();
            hier.scopes()
                .map(|val| Scope(hier[val].clone()))
                .collect::<Vec<_>>()
                .into_iter()
        }))
    }
}

#[pyclass]
struct Scope(pub(crate) wellen::Scope);

#[pymethods]
impl Scope {
    pub fn name(&self, hier: Bound<'_, Hierarchy>) -> String {
        self.0.name(&hier.borrow().0).to_string()
    }
    pub fn full_name(&self, hier: Bound<'_, Hierarchy>) -> String {
        self.0.full_name(&hier.borrow().0).to_string()
    }

    pub fn vars(&self, hier: Bound<'_, Hierarchy>) -> VarIter {
        let locahier = hier.borrow().clone();
        let scope = self.0.clone();

        //TODO: optimize me! need to rewrite the logic from `HierarchyItemIdIterator` to use
        // Arc<Hierarchy> instead of lifetimes
        //
        // This is because python does not like lifetimes :)
        VarIter(Box::new({
            let hier = locahier.clone();
            scope
                .vars(&hier.0)
                .map(|val| Var(hier.0[val].clone()))
                .collect::<Vec<_>>()
                .into_iter()
        }))
    }

    pub fn scopes(&self, hier: Bound<'_, Hierarchy>) -> ScopeIter {
        let locahier = hier.borrow().clone();
        let scope = self.0.clone();

        //TODO: optimize me! need to rewrite the logic from `HierarchyItemIdIterator` to use
        // Arc<Hierarchy> instead of lifetimes
        //
        // This is because python does not like lifetimes :)
        ScopeIter(Box::new({
            let hier = locahier.clone();
            scope
                .scopes(&hier.0)
                .map(|val| Scope(hier.0[val].clone()))
                .collect::<Vec<_>>()
                .into_iter()
        }))
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
struct Var(pub(crate) wellen::Var);

#[pymethods]
impl Var {
    pub fn name(&self, hier: Bound<'_, Hierarchy>) -> String {
        self.0.name(&hier.borrow().0).to_string()
    }
    pub fn full_name(&self, hier: Bound<'_, Hierarchy>) -> String {
        self.0.full_name(&hier.borrow().0).to_string()
    }
    pub fn bitwidth(&self) -> Option<u32> {
        self.0.length()
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

#[pyclass]
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
    #[pyo3(get)]
    hierarchy: Hierarchy,

    wave_source: wellen::SignalSource,
    #[pyo3(get)]
    time_table: TimeTable,
}

#[pymethods]
/// Top level waveform class that end users should use
/// The "egress" point from which all users can read waveforms
impl Waveform {
    #[new]
    #[pyo3(signature = (path, multi_threaded = true, remove_scopes_with_empty_name = false))]
    fn new(
        path: String,
        multi_threaded: bool,
        remove_scopes_with_empty_name: bool,
    ) -> PyResult<Self> {
        let opts = LoadOptions {
            multi_thread: multi_threaded,
            remove_scopes_with_empty_name,
        };
        let header_result = viewers::read_header_from_file(path.as_str(), &opts).toerr()?;
        let hier = Hierarchy(Arc::new(header_result.hierarchy));

        let body = viewers::read_body(header_result.body, &hier.0, None)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;

        Ok(Self {
            hierarchy: hier,
            wave_source: body.source,
            time_table: TimeTable(Arc::new(body.time_table)),
        })
    }
    fn get_signal<'py>(&mut self, var: &Var, py: Python<'py>) -> PyResult<Bound<'py, Signal>> {
        let mut signal =
            self.wave_source
                .load_signals(&[var.0.signal_ref()], &self.hierarchy.0, true);
        let (_sr, sig) = signal.swap_remove(0);
        Bound::new(
            py,
            Signal {
                signal: Arc::new(sig),
                all_times: self.time_table.clone(),
            },
        )
    }

    /// Assumes a dotted signal
    fn get_signal_from_path<'py>(
        &mut self,
        abs_hierarchy_path: String,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, Signal>> {
        let path: Vec<&str> = abs_hierarchy_path.split('.').collect();

        let (path, names) = (
            &path[0..path.len() - 1],
            path.last()
                .ok_or(PyRuntimeError::new_err("Path could not be parsed!")),
        );
        let maybe_var =
            self.hierarchy
                .0
                .lookup_var(path, names?)
                .ok_or(PyRuntimeError::new_err(format!(
                    "No var at path {abs_hierarchy_path}"
                )))?;
        let var = &self.hierarchy.0[maybe_var];
        self.get_signal(&Var(var.clone()), py)
    }
}

#[pyclass]
#[derive(Clone)]
struct Signal {
    signal: Arc<wellen::Signal>,
    all_times: TimeTable,
}

#[pymethods]
impl Signal {
    pub fn value_at_time<'a>(
        &self,
        time: wellen::Time,
        py: Python<'a>,
    ) -> Option<Bound<'a, PyAny>> {
        let val = self
            .all_times
            .0
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
                SignalValue::Real(inner) => Some(inner.into_pyobject(py).unwrap().into_any()),
                SignalValue::String(str) => Some(str.into_pyobject(py).unwrap().into_any()),
                _ => match BigUint::try_from_signal(signal) {
                    // If this signal is 2bits, this function will return an int
                    Some(number) => Some(number.into_pyobject(py).unwrap().into_any()),
                    // if this signal is not 2bits (e.g. it contains z,x, etc) then this function
                    // will return a string
                    None => signal
                        .to_bit_string()
                        .map(|val| val.into_pyobject(py).unwrap().into_any()),
                },
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
            let time = slf.signal.all_times.0.get(*time_idx as usize).cloned()?;
            slf.offset += 1;
            data.map(|val| (time, val))
        } else {
            None
        }
    }
}
