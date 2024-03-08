// Copyright 2023-2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

mod detect;
pub mod fst;
pub mod ghw;
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
    #[error("[vcd] unexpected number of tokens for command {0}: {1}")]
    VcdUnexpectedNumberOfTokens(String, String),
    #[error("[vcd] encountered a attribute with an unsupported type: {0}")]
    VcdUnsupportedAttributeType(String),
    #[error("[vcd] failed to parse VHDL var type from attribute.")]
    VcdFailedToParseVhdlVarType(
        #[from] num_enum::TryFromPrimitiveError<fst_native::FstVhdlVarType>,
    ),
    #[error("[vcd] failed to parse VHDL data type from attribute.")]
    VcdFailedToParseVhdlDataType(
        #[from] num_enum::TryFromPrimitiveError<fst_native::FstVhdlDataType>,
    ),
    #[error("[vcd] unknown var type: {0}")]
    VcdUnknownVarType(String),
    #[error("[vcd] unknown scope type: {0}")]
    VcdUnknownScopeType(String),
    #[error("failed to decode string")]
    Utf8(#[from] std::str::Utf8Error),
    #[error("failed to parse an integer")]
    ParseInt(#[from] std::num::ParseIntError),
    #[error("I/O operation failed")]
    Io(#[from] std::io::Error),
    #[error("failed to load {0:?}:\n{1}")]
    FailedToLoad(FileFormat, String),
}

pub use detect::{detect_file_format, open_and_detect_file_format, FileFormat};
pub use hierarchy::{
    GetItem, Hierarchy, HierarchyItem, Scope, ScopeRef, ScopeType, SignalRef, Timescale,
    TimescaleUnit, Var, VarDirection, VarIndex, VarRef, VarType,
};
pub use signals::{Real, Signal, SignalValue, Time, TimeTableIdx, Waveform};
use thiserror::Error;

#[cfg(feature = "benchmark")]
pub use wavemem::check_states_pub;
