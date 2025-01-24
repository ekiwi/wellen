// Copyright 2024 The Regents of the University of California
// Copyright 2024-2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>
//
// test VHDL specific features

use wellen::simple::*;
use wellen::*;

#[test]
fn test_vhdl_ghdl_fst() {
    let filename = "inputs/ghdl/oscar/ghdl.fst";
    let waves = read(filename).expect("failed to parse");
    let h = waves.hierarchy();
    let ee = &h[h.vars().next().unwrap()];
    assert_eq!("ee", ee.name(h));
    assert_eq!(Some("e"), ee.vhdl_type_name(h)); // lol: VHDL is case insensitive!
}

fn test_vhdl_3(waves: Waveform) {
    let h = waves.hierarchy();

    let top = h.first_scope().unwrap();
    assert_eq!("test", top.name(h));
    assert_eq!(top.scope_type(), ScopeType::VhdlArchitecture);

    let ee = &h[top.vars(h).next().unwrap()];
    assert_eq!("ee", ee.name(h));
    assert_eq!(Some("E"), ee.vhdl_type_name(h));

    let rr = &h[top.scopes(h).next().unwrap()];
    assert_eq!("rr", rr.name(h));
    assert_eq!(rr.source_loc(h), None);
    assert_eq!(rr.instantiation_source_loc(h), None);
    assert_eq!(rr.scope_type(), ScopeType::VhdlRecord);

    for var in rr.vars(h).map(|v| &h[v]) {
        match var.name(h) {
            "a" => {
                assert_eq!(var.vhdl_type_name(h), Some("STD_LOGIC"));
                // TODO: this should be "Logic" and not "ULogic". Not sure why we end up with this though...
                assert_eq!(var.var_type(), VarType::StdULogic);
            }
            "b" | "c" => {
                assert_eq!(var.vhdl_type_name(h), Some("STD_LOGIC_VECTOR"));
                assert_eq!(var.var_type(), VarType::StdLogicVector);
            }
            "d" => {
                assert_eq!(var.vhdl_type_name(h), Some("E"));
                assert_eq!(var.var_type(), VarType::String);
            }
            other => unreachable!("Unexpected var: {other}"),
        }
    }
}

#[test]
fn test_vhdl_3_fst() {
    let filename = "inputs/ghdl/oscar/vhdl3.fst";
    let waves = read(filename).expect("failed to parse");
    test_vhdl_3(waves);
}

#[test]
fn test_vhdl_3_vcd() {
    let filename = "inputs/ghdl/oscar/vhdl3.vcd";
    let waves = read(filename).expect("failed to parse");
    test_vhdl_3(waves);
}
