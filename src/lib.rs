// Copyright 2023-2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

mod fst;
mod ghw;
mod hierarchy;
mod signals;
mod vcd;
pub mod viewers;
mod wavemem;

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum FileFormat {
    Vcd,
    Fst,
    Ghw,
    Unknown,
}
#[derive(Debug, Copy, Clone)]
pub struct LoadOptions {
    /// Indicates that the loader should use multiple threads if possible.
    pub multi_thread: bool,
    /// Indicates that scopes with empty names should not be part of the hierarchy.
    pub remove_scopes_with_empty_name: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            multi_thread: true,
            remove_scopes_with_empty_name: false,
        }
    }
}

pub type TimeTable = Vec<Time>;

#[derive(Debug, Error)]
pub enum WellenError {
    #[error("failed to load {0:?}:\n{1}")]
    FailedToLoad(FileFormat, String),
    #[error("unknown file format, only GHW, FST and VCD are supported")]
    UnknownFileFormat,
    #[error("io error")]
    Io(#[from] std::io::Error),
}

pub use hierarchy::{
    GetItem, Hierarchy, HierarchyItem, Scope, ScopeRef, ScopeType, SignalRef, Timescale,
    TimescaleUnit, Var, VarDirection, VarIndex, VarRef, VarType,
};
pub use signals::{Real, Signal, SignalValue, Time, TimeTableIdx};
use thiserror::Error;

#[cfg(feature = "benchmark")]
pub use wavemem::check_states_pub;
