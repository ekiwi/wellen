// Copyright 2023-2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

mod detect;
pub mod fst;
mod hierarchy;
mod signals;
pub mod vcd;
mod wavemem;

pub use detect::{detect_file_format, FileFormat};
pub use hierarchy::{
    FileType, GetItem, Hierarchy, HierarchyItem, Scope, ScopeRef, ScopeType, SignalRef, Timescale,
    TimescaleUnit, Var, VarDirection, VarIndex, VarRef, VarType,
};
pub use signals::{Real, SignalValue, Time, TimeTableIdx, Waveform};

#[cfg(feature = "benchmark")]
pub use wavemem::check_states_pub;
