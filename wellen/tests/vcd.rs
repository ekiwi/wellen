// Copyright 2024 The Regents of the University of California
// Copyright 2024-2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

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
        let cpu = &h[top.scopes(h).next().unwrap()];
        assert_eq!("cpu", cpu.name(h));

        let pc = &h[cpu.vars(h).find(|r| h[*r].name(h) == "pc").unwrap()];
        assert_eq!("gameroy.cpu.pc", pc.full_name(h));
        let sp = &h[cpu.vars(h).find(|r| h[*r].name(h) == "sp").unwrap()];
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
        h[logic.vars(h).next().unwrap()].clone()
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
    let top_scopes = h.scopes().map(|s| h[s].full_name(h)).collect::<Vec<_>>();
    assert_eq!(top_scopes, ["tb"]);
}

/// This test file was provided by Gianluca Bellocchi in the following issue:
/// https://github.com/ekiwi/wellen/issues/27
/// Because of our automatic scope creation from variable names, we aren't able to run a diff test
/// on this file which is why we opt to just check that we can load it without crashing.
#[test]
fn load_vivado_surfer_test() {
    let filename = "inputs/vivado/vivado_surfer_test.vcd";
    let _waves = read(filename).expect("failed to parse");
}

/// A user reported problems with parsing a VCD that contains change value entries for a 0-bit signal
/// https://github.com/ekiwi/wellen/issues/28
/// The VCD was created by a VHDL simulator, and thus it isn't possible to compare the hierarchy
/// we create directly.
#[test]
fn load_github_issue_28() {
    let filename = "inputs/github_issues/issue28.vcd";
    let _waves = read(filename).expect("failed to parse");
}

/// This invalid VCD file used to crash wellen instead of returning an error.
/// https://github.com/ekiwi/wellen/issues/18
#[test]
fn load_github_issue_18() {
    let filename = "inputs/github_issues/issue18.vcd";
    let opts = LoadOptions {
        multi_thread: false,
        remove_scopes_with_empty_name: false,
    };
    let r = read_with_options(filename, &opts);
    assert!(r.is_err());
    assert!(r.err().unwrap().to_string().contains("expected an id"));
}

/// https://github.com/ekiwi/wellen/issues/36
/// The problem was that wellen was interpreting an array index as a bit index.
/// We are working around this now by checking bit index against the width.
#[test]
fn amaranth_array_support_issue_36() {
    let filename = "inputs/amaranth/array-names_wellen_issue_36.vcd";
    let waves = read(filename).expect("failed to parse");
    let h = waves.hierarchy();
    let o_md_0_0 = &h[h
        .lookup_var(&["bench", "top", "\\o_md", "[0]"], &"[0]")
        .expect("failed to find bench.top.o_md.[0].[0]")];
    assert!(o_md_0_0.index().is_none());
    assert_eq!(o_md_0_0.length(), Some(32));
}

#[test]
fn load_github_issue_42() {
    let filename = "inputs/github_issues/issue42.vcd";
    let _waves = read(filename).expect("failed to parse");
}

/// https://github.com/ekiwi/wellen/issues/40
/// Invalid commands in the VCD file should lead to an error being returned, not a panic.
#[test]
fn load_github_issue_40() {
    let filename = "inputs/github_issues/issue40.vcd";
    let r = read(filename);
    assert!(r.is_err());
    assert!(r
        .err()
        .unwrap()
        .to_string()
        .contains("unknown or invalid command"));
}

/// Time stamps ending with .0 can show up from Migen. Make sure they can be parsed.
#[test]
fn load_github_issue_55_float_time_stamp() {
    let filename = "inputs/migen/migen.vcd";
    let _waves = read(filename).expect("failed to parse");
}

/// Error for float time stamps that does not have a zero fractional part.
/// Migen will not produce these, but someone may...
#[test]
fn load_github_issue_55_fractional_time_stamp() {
    let filename = "inputs/migen/fractional_time_stamp.vcd";
    let r = read(filename);
    assert!(r.is_err());
    assert!(r.err().unwrap().to_string().contains("parse an integer"));
}
