// Copyright 2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>
//
// test fst specific meta data

use wellen::*;

#[test]
fn test_tb_recv_ghw() {
    let filename = "inputs/ghdl/tb_recv.ghw";
    let _waves = ghw::read(filename).expect("failed to parse");
}

#[test]
fn test_oscar_test_ghw() {
    let filename = "inputs/ghdl/oscar/test.ghw";
    let _waves = ghw::read(filename).expect("failed to parse");
}

#[test]
fn test_oscar_test2_ghw() {
    let filename = "inputs/ghdl/oscar/test2.ghw";
    let _waves = ghw::read(filename).expect("failed to parse");
}

#[test]
#[ignore] // neither ghwdump nor gtkwave seem to be able to open this file
fn test_ghdl_issue_538_ghw() {
    let filename = "inputs/ghdl/ghdl_issue_538.ghw";
    let _waves = ghw::read(filename).expect("failed to parse");
}
