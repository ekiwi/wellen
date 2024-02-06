// Copyright 2023-2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

mod detect;
pub mod fst;
mod hierarchy;
mod signals;
pub mod vcd;
mod wavemem;

#[derive(Debug, Error)]
pub enum WellenError {
    #[error("[vcd] failed to parse length: `{0}` for variable `{1}`")]
    VcdVarLengthParsing(String, String),
    #[error("[vcd] expected command to start with `$`, not `{0}`")]
    VcdStartChar(String),
    #[error("failed to decode string")]
    Utf8(#[from] std::str::Utf8Error),
    #[error("I/O operation failed")]
    Io(#[from] std::io::Error),
}

pub use detect::{detect_file_format, FileFormat};
pub use hierarchy::{
    FileType, GetItem, Hierarchy, HierarchyItem, Scope, ScopeRef, ScopeType, SignalRef, Timescale,
    TimescaleUnit, Var, VarDirection, VarIndex, VarRef, VarType,
};
pub use signals::{Real, SignalValue, Time, TimeTableIdx, Waveform};
use thiserror::Error;

#[cfg(feature = "benchmark")]
pub use wavemem::check_states_pub;
