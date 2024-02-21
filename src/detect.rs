// Copyright 2023-2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>
//
// Small utility to detect a file format.

use std::io::{BufRead, Seek};

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum FileFormat {
    Vcd,
    Fst,
    Ghw,
    Unknown,
}

/// Tries to guess the format of the file.
pub fn open_and_detect_file_format(filename: &str) -> FileFormat {
    let input_file = std::fs::File::open(filename).expect("failed to open input file!");
    let mut reader = std::io::BufReader::new(input_file);
    detect_file_format(&mut reader)
}

/// Tries to guess the file format used by the input.
pub fn detect_file_format(input: &mut (impl BufRead + Seek)) -> FileFormat {
    if crate::vcd::is_vcd(input) {
        FileFormat::Vcd
    } else if fst_native::is_fst_file(input) {
        FileFormat::Fst
    } else if crate::ghw::is_ghw(input) {
        FileFormat::Ghw
    } else {
        FileFormat::Unknown
    }
}
