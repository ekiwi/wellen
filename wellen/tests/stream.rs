// Copyright 2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use wellen::stream::*;
use wellen::{GetItem, LoadOptions};

#[test]
fn test_stream_vcd() {
    let opts = LoadOptions::default();
    let mut waves = read_from_file("inputs/xilinx_isim/test1.vcd", &opts).unwrap();
    let nounce_1 = waves
        .hierarchy()
        .lookup_var(&["tbtop", "hash_cond", "comparador"], &"nonce_1")
        .unwrap();
    let signal = waves.hierarchy().get(nounce_1).signal_ref();
    let filter = Filter::new(0, u64::MAX, &[signal]);
    for (time, sig, value) in waves.stream(&filter) {
        assert_eq!(signal, sig);
        println!("nounce_1@{time}: {}", value.to_string());
    }
}
