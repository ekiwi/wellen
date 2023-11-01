// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use waveform::*;

fn run_diff_test(vcd_filename: &str, fst_filename: &str) {
    let (vcd_hierarchy, vcd_source) =

}

#[test]
fn diff_treadle_gcd() {
    run_diff_test("inputs/treadle/GCD.vcd", "inputs/treadle/GCD.vcd.fst");
}