// Copyright 2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

pub enum Command {
    Dumpfile(String),
    DumpfileTime(String),
    DumpfileSize(u64),
    Savefile(String),
    /// Start of the wave view.
    TimeStart(u64),
    /// GTKWave zoom factor.
    ZoomFactor(f64),
    /// Unfold in the hierarchy view.
    TreeOpen(HierarchyPath),
    /// Called trace flags in GTKWave.
    SignalFlags(SignalFlags),
    SignalShift(i64),
    /// Add a signal to the viewport.
    AddSignal(VarName),
}

pub struct HierarchyPath(String);
pub struct VarName(String);
pub struct SignalFlags();

/// List of commands that we ignore because they are too GTKWave specific!
pub const IGNORED: &[&[u8]] = [
    // window size (x,y)
    b"size",
    // window position (x,y)
    b"pos",
    // the following specify the size of GTKWave UI panels
    b"sst_width",
    b"sst_expanded",
    b"sst_vpaned_height",
    b"signals_width",
    // TODO: what does this do?
    b"pattern_trace",
];