// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use std::io::BufReader;

fn run_diff_test(vcd_filename: &str, fst_filename: &str) {
    let (vcd_hierarchy, vcd_source) = waveform::vcd::read(vcd_filename);
    let (fst_hierarchy, fst_source) = waveform::fst::read(fst_filename);
    let mut ref_parser =
        vcd::Parser::new(BufReader::new(std::fs::File::open(vcd_filename).unwrap()));
    let ref_header = ref_parser.parse_header().unwrap();
}

#[test]
fn diff_treadle_gcd() {
    run_diff_test("inputs/treadle/GCD.vcd", "inputs/treadle/GCD.vcd.fst");
}
