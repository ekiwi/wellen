// Copyright 2023-2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>
//
// Small utility to detect a file format.

use std::io::{BufRead, Seek};

pub enum FileFormat {
    Vcd,
    Fst,
    Unknown,
}

/// Tries to guess the file format used by the input.
pub fn detect_file_format(input: &mut (impl BufRead + Seek)) -> FileFormat {
    if crate::vcd::is_vcd(input) {
        FileFormat::Vcd
    } else if fst_native::is_fst_file(input) {
        FileFormat::Fst
    } else {
        FileFormat::Unknown
    }
}
