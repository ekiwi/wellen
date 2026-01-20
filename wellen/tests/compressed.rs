// Copyright 2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use wellen::simple::Waveform;
use wellen::*;

fn test_compression(wave: &mut Waveform) {
    let all_signals: Vec<(SignalRef, String)> = wave
        .hierarchy()
        .get_unique_signals_vars()
        .iter()
        .flatten()
        .map(|v| (v.signal_ref(), v.full_name(wave.hierarchy())))
        .collect();
    for (idx, signal_name) in all_signals {
        wave.load_signals(&[idx]);
        let signal = wave.get_signal(idx).expect("signal should be loaded!");
        let compressed = CompressedSignal::compress(signal);
        let uncompressed: Signal = compressed.uncompress();
        assert_eq!(signal, &uncompressed, "{signal_name}");
        wave.unload_signals(&[idx]);
    }
    // test time table compression
    let compressed_tt = CompressedTimeTable::compress(wave.time_table());
    let uncompressed_tt = compressed_tt.uncompress();
    assert_eq!(uncompressed_tt.as_slice(), wave.time_table());
}

fn do_test_from_file(filename: &str) {
    let mut wave = simple::read(filename).expect("failed to load input file");
    test_compression(&mut wave);
}

#[test]
fn test_compressed_jtag_atxmega256a3u_bmda() {
    do_test_from_file("inputs/jtag/atxmega256a3u-bmda-jtag.vcd");
}

#[test]
fn test_compressed_gameroy_trace() {
    do_test_from_file("inputs/gameroy/trace_prefix.vcd");
}

#[test]
fn test_compressed_surfer_counter() {
    do_test_from_file("inputs/surfer/counter.vcd");
}

#[test]
fn test_compressed_tb_recv_ghw() {
    do_test_from_file("inputs/ghdl/tb_recv.ghw");
}

#[test]
fn test_compressed_oscar_test_ghw() {
    do_test_from_file("inputs/ghdl/oscar/test.ghw");
}

#[test]
fn test_compressed_oscar_test2_ghw() {
    do_test_from_file("inputs/ghdl/oscar/test2.ghw");
}

#[test]
#[ignore] // TODO: this test fails because a signal fails to load, not because of compression!
fn test_compressed_issue_12_regression() {
    do_test_from_file("inputs/ghdl/wellen_issue_12.ghw");
}

#[test]
fn test_compressed_issue_6_generate_for_aliasing() {
    do_test_from_file("inputs/ghdl/wellen_issue_6.ghw");
}

#[test]
fn test_compressed_verilator_surfer_issue_201() {
    do_test_from_file("inputs/verilator/surfer_issue_201.fst");
}

#[test]
fn test_compressed_nvc_xwb_fofb_shaper_filt_tb() {
    do_test_from_file("inputs/nvc/xwb_fofb_shaper_filt_tb.fst");
}

#[test]
fn test_compressed_nvc_vhdl_test_bool_issue_16() {
    do_test_from_file("inputs/nvc/vhdl_test_bool_issue_16.fst");
}
