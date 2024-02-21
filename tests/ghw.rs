// Copyright 2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>
//
// test fst specific meta data

use wellen::*;

#[test]
fn test_tb_recv_ghw() {
    let filename = "inputs/ghdl/tb_recv.ghw";
    let waves = ghw::read(filename).expect("failed to parse");
}
