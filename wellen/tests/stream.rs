// Copyright 2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use wellen::stream::*;
use wellen::LoadOptions;

#[test]
fn test_stream_vcd() {
    let opts = LoadOptions::default();
    let mut waves = read_from_file("inputs/xilinx_isim/test1.vcd", &opts).unwrap();
    let nounce_1 = waves
        .hierarchy()
        .lookup_var(&["tbtop", "hash_cond", "comparador"], &"nonce_1")
        .unwrap();
    let signals = [waves.hierarchy()[nounce_1].signal_ref()];
    let filter = Filter::new(0, u64::MAX, &signals);
    waves
        .stream(&filter, |time, sig, value| {
            assert_eq!(signals[0], sig);
            println!("nounce_1@{time}: {}", value.to_string());
        })
        .unwrap();
}
