// Copyright 2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

#[cfg(feature = "serde1")]
use bincode::Options;
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
        assert_eq!(signal, &uncompressed, "{}", signal_name);
        compare_size(&uncompressed, &compressed);
        wave.unload_signals(&[idx]);
    }
    // test time table compression
    let compressed_tt = CompressedTimeTable::compress(wave.time_table());
    let uncompressed_tt = compressed_tt.uncompress();
    assert_eq!(uncompressed_tt.as_slice(), wave.time_table());
    compare_time_table_size(wave.time_table(), &compressed_tt);
}

#[cfg(not(feature = "serde1"))]
fn compare_size(_a: &Signal, _b: &CompressedSignal) {
    // nothing to do without serdes
}

#[cfg(feature = "serde1")]
fn compare_size(a: &Signal, b: &CompressedSignal) {
    let opts = bincode::DefaultOptions::new();
    let uncompressed = opts.serialize(a).unwrap();
    let lz4_only = lz4_flex::compress_prepend_size(&uncompressed);
    let compressed = opts.serialize(b).unwrap();
    let delta = uncompressed.len() - compressed.len();
    let relative_delta = 10000 * delta / uncompressed.len();
    let relative_delta_lz4 = (lz4_only.len() - compressed.len()) * 10000 / lz4_only.len();
    println!(
        "Saved {}%     {} vs. {}  ... {}% vs using only lz4",
        relative_delta as f64 / 100.0,
        uncompressed.len(),
        compressed.len(),
        relative_delta_lz4 as f64 / 100.0,
    );
}

#[cfg(not(feature = "serde1"))]
fn compare_time_table_size(_a: &[Time], _b: &CompressedTimeTable) {
    // nothing to do without serdes
}

#[cfg(feature = "serde1")]
fn compare_time_table_size(a: &[Time], b: &CompressedTimeTable) {
    let opts = bincode::DefaultOptions::new();
    let uncompressed = opts.serialize(a).unwrap();
    let lz4_only = lz4_flex::compress_prepend_size(&uncompressed);
    let compressed = opts.serialize(b).unwrap();
    let delta = uncompressed.len() - compressed.len();
    let relative_delta = 10000 * delta / uncompressed.len();
    let relative_delta_lz4 = (lz4_only.len() - compressed.len()) * 10000 / lz4_only.len();
    println!(
        "Timetable: saved {}%     {} vs. {}  ... {}% vs using only lz4",
        relative_delta as f64 / 100.0,
        uncompressed.len(),
        compressed.len(),
        relative_delta_lz4 as f64 / 100.0,
    );
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
