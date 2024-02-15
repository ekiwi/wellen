// Copyright 2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use wellen::*;

#[test]
fn test_vcd_not_starting_at_zero() {
    let filename = "inputs/gameroy/trace_prefix.vcd";
    let mut waves = vcd::read(filename).expect("failed to parse");

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
