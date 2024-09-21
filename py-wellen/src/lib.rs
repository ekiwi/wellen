use pyo3::{exceptions::PyRuntimeError, prelude::*};
use wellen::{
    viewers::{self, read_body, read_header},
    LoadOptions, Signal, SignalSource,
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
        Ok(Hierarchy(header_result.hierarchy))
    }

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
    time_table: wellen::TimeTable,
}

#[pyclass]
struct Trace {
    hierarchy: Hierarchy,
    body: Body,
}

#[pyclass]
struct WellenSignal(Signal);
