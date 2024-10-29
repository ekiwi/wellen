// Copyright 2023-2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

pub mod derived_signals;
mod fst;
mod ghw;
mod hierarchy;
mod signal_utils;
mod signals;
pub mod simple;
mod vcd;
pub mod viewers;
mod wavemem;

/// Cargo.toml version of this library.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, PartialEq, Copy, Clone)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
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

#[derive(Debug, thiserror::Error)]
pub enum WellenError {
    #[error("failed to load {0:?}:\n{1}")]
    FailedToLoad(FileFormat, String),
    #[error("unknown file format, only GHW, FST and VCD are supported")]
    UnknownFileFormat,
    #[error("io error")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, WellenError>;

pub use hierarchy::{
    GetItem, Hierarchy, HierarchyItem, Scope, ScopeRef, ScopeType, SignalEncoding, SignalRef,
    Timescale, TimescaleUnit, Var, VarDirection, VarIndex, VarRef, VarType,
};
pub use signal_utils::all_changes;
pub use signals::{BitVectorBuilder, Real, Signal, SignalSource, SignalValue, Time, TimeTableIdx};
pub use wavemem::States;

#[cfg(feature = "benchmark")]
pub use wavemem::check_states_pub;
