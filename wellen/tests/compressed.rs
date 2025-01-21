// Copyright 2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

#[cfg(feature = "serde1")]
use bincode::Options;
use wellen::simple::Waveform;
use wellen::{CompressedSignal, Signal, SignalRef};

fn test_compression(wave: &mut Waveform) {
    let all_signals: Vec<SignalRef> = wave
        .hierarchy()
        .get_unique_signals_vars()
        .iter()
        .flatten()
        .map(|v| v.signal_ref())
        .collect();
    for idx in all_signals {
        wave.load_signals(&[idx]);
        let signal = wave.get_signal(idx).expect("signal should be loaded!");
        let compressed: CompressedSignal = signal.into();
        let uncompressed: Signal = (&compressed).into();
        assert_eq!(signal, &uncompressed);
        compare_size(&uncompressed, &compressed);
        wave.unload_signals(&[idx]);
    }
}

#[cfg(not(feature = "serde1"))]
fn compare_size(_a: &Signal, _b: &CompressedSignal) {
    // nothing to do without serdes
}

#[cfg(feature = "serde1")]
fn compare_size(a: &Signal, b: &CompressedSignal) {
    let opts = bincode::DefaultOptions::new();
    let uncompressed = opts.serialize(a).unwrap();
    let compressed = opts.serialize(b).unwrap();
    let delta = uncompressed.len() - compressed.len();
    let relative_delta = 10000 * delta / uncompressed.len();
    println!(
        "Saved {}%     {} vs. {}",
        relative_delta as f64 / 100.0,
        uncompressed.len(),
        compressed.len()
    );
}

fn do_test_from_file(filename: &str) {
    let mut wave = wellen::simple::read(filename).expect("failed to load input file");
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
