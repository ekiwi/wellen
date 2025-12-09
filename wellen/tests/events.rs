// Copyright 2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use wellen::simple::{Waveform, read};
use wellen::{Signal, SignalRef, SignalValue, Time};

/// The test files were provided by Roman Popov here: https://github.com/ekiwi/wellen/pull/67
#[test]
fn test_vcd_fst_events() {
    test_pull_67_event_example("inputs/icarus/pull_67_event_example.fst");
    test_pull_67_event_example("inputs/icarus/pull_67_event_example.vcd");
}

fn test_pull_67_event_example(filename: &str) {
    let mut waves = read(filename).unwrap();
    let event_1 = get_signal_ref(&waves, &["event_example"], "event1");
    let event_2 = get_signal_ref(&waves, &["event_example"], "event2");
    waves.load_signals([event_1, event_2].as_slice());
    let e1 = waves.get_signal(event_1).unwrap();
    let e2 = waves.get_signal(event_2).unwrap();

    let e1_changes = collect_events(&waves, e1);
    let e2_changes = collect_events(&waves, e2);
    assert_eq!(e1_changes, [0, 10, 45, 70]);
    assert_eq!(e2_changes, [0, 30, 70]);
}

fn collect_events(w: &Waveform, s: &Signal) -> Vec<Time> {
    s.iter_changes()
        .map(|(time_idx, value)| {
            assert!(
                matches!(value, SignalValue::Event),
                "{value:?} is not an event"
            );
            assert_eq!(value.bits().unwrap(), 0);
            w.time_table()[time_idx as usize]
        })
        .collect()
}

fn get_signal_ref<N: AsRef<str>>(w: &Waveform, path: &[N], name: N) -> SignalRef {
    let var_ref = w.hierarchy().lookup_var(path, name).unwrap();
    w.hierarchy()[var_ref].signal_ref()
}
