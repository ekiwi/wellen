mod convert;
use std::sync::Arc;

use convert::Mappable;
use pyo3::conversion::ToPyObject;
use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    types::{PyFloat, PyInt, PyString},
};
use wellen::GetItem;
use wellen::{
    viewers::{self, read_body, read_header},
    LoadOptions, SignalSource, SignalValue, TimeTableIdx,
};

pub trait PyErrExt<T> {
    fn toerr(self) -> PyResult<T>;
}

impl<T> PyErrExt<T> for wellen::Result<T> {
    fn toerr(self) -> PyResult<T> {
        self.map_err(|err| PyRuntimeError::new_err(err.to_string()))
    }
}

/// Formats the sum of two numbers as string.

/// A Python module implemented in Rust.
#[pymodule]
fn py_wellen(_py: Python, m: Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}

#[pyclass]
struct Hierarchy(pub(crate) wellen::Hierarchy);

#[pyclass]
struct VarIter(Box<dyn Iterator<Item = Var> + Send>);

#[pymethods]
impl Hierarchy {
    fn all_vars(&self) -> VarIter {
        VarIter(Box::new(
            self.0
                .get_unique_signals_vars()
                .into_iter()
                .flatten()
                .map(|val| Var(val)),
        ))
    }
}

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
struct Var(wellen::Var);

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
struct Body {
    wave_source: wellen::SignalSource,
    time_table: std::sync::Arc<wellen::TimeTable>,
}

#[pyclass]
struct Trace {
    hierarchy: Hierarchy,
    body: Body,
    multi_threaded: bool,
}

#[pymethods]
impl Trace {
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
        let header_result = viewers::read_header(path.as_str(), &opts).toerr()?;
        let hier = Hierarchy(header_result.hierarchy);

        let body = viewers::read_body(header_result.body, &hier.0, None)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        let body = Body {
            wave_source: body.source,
            time_table: Arc::new(body.time_table),
        };
        Ok(Self {
            hierarchy: hier,
            body,
            multi_threaded,
        })
    }

    /// Assumes a dotted signal
    fn get_signal<'py>(
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
        let var = self.hierarchy.0.get(maybe_var);
        let mut signal =
            self.body
                .wave_source
                .load_signals(&[var.signal_ref()], &self.hierarchy.0, true);
        let (sr, sig) = signal.swap_remove(0);
        Bound::new(
            py,
            Signal {
                signal: sig,
                all_times: self.body.time_table.clone(),
            },
        )
    }
}

#[pyclass]
struct Signal {
    signal: wellen::Signal,
    all_times: Arc<wellen::TimeTable>,
}

#[pymethods]
impl Signal {
    pub fn value_at_time(&self, time: wellen::Time, py: Python<'_>) -> Option<Py<PyAny>> {
        let val = self
            .all_times
            .as_ref()
            .binary_search(&time)
            .unwrap_or_else(|val| val);
        self.value_at_idx(val as TimeTableIdx, py)
    }

    pub fn value_at_idx(&self, idx: TimeTableIdx, py: Python<'_>) -> Option<Py<PyAny>> {
        let maybe_signal = self
            .signal
            .get_offset(idx)
            .map(|data_offset| self.signal.get_value_at(&data_offset, 0));
        if let Some(signal) = maybe_signal {
            let output = match signal {
                SignalValue::Real(inner) => Some(inner.to_object(py)),
                SignalValue::String(str) => Some(str.to_object(py)),
                _ => match u64::try_from_signal(signal) {
                    Some(number) => Some(number.to_object(py)),
                    None => signal.to_bit_string().map(|val| val.to_object(py)),
                },
            };
            output
        } else {
            None
        }
    }
}

#[pyclass]
struct SignalCursor {
    signal: Signal,
    idx: TimeTableIdx,
}
// TODO; do pyiter, searchable cursor
