// Copyright 2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use wellen::*;

#[test]
fn test_vcd_not_starting_at_zero() {
    let filename = "inputs/gameroy/trace_prefix.vcd";
    let waves = vcd::read(filename).expect("failed to parse");
    let h = waves.hierarchy();

    // the first signal change only happens at 4
    assert_eq!(waves.time_table()[0], 4);

    let top = h.first_scope().unwrap();
    assert_eq!("gameroy", top.name(h));
    let cpu = h.get(top.scopes(h).next().unwrap());
    assert_eq!("cpu", cpu.name(h));

    let pc = h.get(cpu.vars(h).find(|r| h.get(*r).name(h) == "pc").unwrap());
    assert_eq!("gameroy.cpu.pc", pc.full_name(h));

    // querying a signal before it has a value should return none
}
