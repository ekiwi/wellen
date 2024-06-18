// Copyright 2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use wellen::simple::*;
use wellen::*;

#[test]
fn test_vcd_not_starting_at_zero() {
    let filename = "inputs/gameroy/trace_prefix.vcd";
    let mut waves = read(filename).expect("failed to parse");

    let (pc, sp) = {
        let h = waves.hierarchy();

        // the first signal change only happens at 4
        assert_eq!(waves.time_table()[0], 4);

        let top = h.first_scope().unwrap();
        assert_eq!("gameroy", top.name(h));
        let cpu = h.get(top.scopes(h).next().unwrap());
        assert_eq!("cpu", cpu.name(h));

        let pc = h.get(cpu.vars(h).find(|r| h.get(*r).name(h) == "pc").unwrap());
        assert_eq!("gameroy.cpu.pc", pc.full_name(h));
        let sp = h.get(cpu.vars(h).find(|r| h.get(*r).name(h) == "sp").unwrap());
        assert_eq!("gameroy.cpu.sp", sp.full_name(h));
        (pc.clone(), sp.clone())
    };

    // querying a signal before it has a value should return none
    waves.load_signals(&[pc.signal_ref(), sp.signal_ref()]);

    // pc is fine since it changes at 4 which is time_table idx 0
    let pc_signal = waves.get_signal(pc.signal_ref()).unwrap();
    assert!(pc_signal.get_offset(0).is_some());

    // sp only changes at 16 which is time table idx 1
    let sp_signal = waves.get_signal(sp.signal_ref()).unwrap();
    assert!(sp_signal.get_offset(1).is_some());
    assert!(sp_signal.get_offset(0).is_none());
}

/// If a VCD records a change to an input, but the value is the same, our signal loader ensures to remove that change.
#[test]
fn test_vcd_with_fake_changes() {
    let filename = "inputs/surfer/issue_145.vcd";
    let waves = read(filename).expect("failed to parse");
    check_no_fake_changes(waves);
}

/// Make sure that we also do not get duplicate changes for the fst
#[test]
fn test_fst_created_from_vcd_with_fake_changes() {
    let filename = "inputs/surfer/issue_145.vcd.fst";
    let waves = read(filename).expect("failed to parse");
    check_no_fake_changes(waves);
}

fn check_no_fake_changes(mut waves: Waveform) {
    let data = {
        let h = waves.hierarchy();
        let logic = h.first_scope().unwrap();
        h.get(logic.vars(h).next().unwrap()).clone()
    };
    assert_eq!(data.full_name(waves.hierarchy()), "logic.data");

    waves.load_signals(&[data.signal_ref()]);
    let signal = waves.get_signal(data.signal_ref()).unwrap();

    assert_eq!(
        signal.time_indices(),
        [0],
        "The signal never actually changes, thus only a change at zero should be recorded."
    );

    // make sure there are no delta cycles
    let offset = signal.get_offset(0).unwrap();
    assert_eq!(offset.elements, 1);
    assert_eq!(offset.next_index, None);
}

/// If a VCD contains a time change that goes back in time, wellen will skip it.
#[test]
fn test_vcd_with_decreasing_time() {
    let filename = "inputs/wellen/issue_5.vcd";
    let waves = read(filename).expect("failed to parse");
    assert_eq!(waves.time_table(), [4, 5]);
}

/// This VCD has multiple overlapping scopes that need to be merged.
/// src: https://gitlab.com/surfer-project/surfer/-/issues/256
#[test]
fn test_vcd_scope_merging() {
    let filename = "inputs/icarus/surfer_issue_256.vcd";
    let waves = read(filename).expect("failed to parse");
    let h = waves.hierarchy();
    let top_scopes = h
        .scopes()
        .map(|s| h.get(s).full_name(h))
        .collect::<Vec<_>>();
    assert_eq!(top_scopes, ["tb"]);
}
