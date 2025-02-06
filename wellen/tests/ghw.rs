// Copyright 2024 The Regents of the University of California
// Copyright 2024-2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>
//
// test fst specific meta data

use wellen::simple::*;
use wellen::*;

#[test]
fn test_tb_recv_ghw() {
    let filename = "inputs/ghdl/tb_recv.ghw";
    let _waves = read(filename).expect("failed to parse");
}

#[test]
fn test_oscar_test_ghw() {
    let filename = "inputs/ghdl/oscar/test.ghw";
    let _waves = read(filename).expect("failed to parse");
}

#[test]
fn test_oscar_test2_ghw() {
    let filename = "inputs/ghdl/oscar/test2.ghw";
    let _waves = read(filename).expect("failed to parse");
}

/// we are trying to match the encoding that FST uses
#[test]
fn test_ghw_enum_encoding() {
    let filename = "inputs/ghdl/oscar/test2.ghw";
    let mut waves = read(filename).expect("failed to parse");
    let h = waves.hierarchy();
    let top = h.scopes().find(|s| h[*s].name(h) == "test2").unwrap();
    let bbb = h[top].vars(h).find(|v| h[*v].name(h) == "bbb").unwrap();
    let ee = h[top].vars(h).find(|v| h[*v].name(h) == "ee").unwrap();

    // signal bbb: boolean
    {
        let (enum_name, enum_lits) = h[bbb].enum_type(h).unwrap();
        assert_eq!(enum_name, "boolean");
        assert_eq!(enum_lits, [("0", "false"), ("1", "true")]);
    }

    // type e is(foo, bar, tada);
    // signal ee: e
    {
        let (enum_name, enum_lits) = h[ee].enum_type(h).unwrap();
        assert_eq!(enum_name, "e");
        assert_eq!(h[ee].length().unwrap(), 2);
        // 2 bits are used for the encoding and thus everything is padded to a width of 2
        assert_eq!(enum_lits, [("00", "foo"), ("01", "bar"), ("10", "tada")]);
    }

    // make sure that all enum values are binary
    let ee_signal_ref = h[ee].signal_ref();
    waves.load_signals(&[ee_signal_ref]);
    let ee_signal = waves.get_signal(ee_signal_ref).unwrap();
    for id in 0..waves.time_table().len() {
        let off = ee_signal.get_offset(id as TimeTableIdx).unwrap();
        let value = ee_signal.get_value_at(&off, 0);
        assert_eq!(value.to_bit_string().unwrap().len(), 2);
        assert!(matches!(value, SignalValue::Binary(_, 2)));
    }
}

#[test]
#[ignore] // neither ghwdump nor gtkwave seem to be able to open this file
fn test_ghdl_issue_538_ghw() {
    let filename = "inputs/ghdl/ghdl_issue_538.ghw";
    let _waves = read(filename).expect("failed to parse");
}

/// See: https://github.com/ekiwi/wellen/issues/12
/// Data of a 3-bit signal was corrupted in the `compress` function.
#[test]
fn test_issue_12_regression() {
    let mut wave = read("inputs/ghdl/wellen_issue_12.ghw").unwrap();

    let var_id = wave
        .hierarchy()
        .lookup_var(&["test_rom_tb", "soc_inst", "core_inst"], &"state")
        .unwrap();
    let signal_ref = wave.hierarchy()[var_id].signal_ref();
    wave.load_signals(&[signal_ref]);

    let signal = wave.get_signal(signal_ref).unwrap();
    let mut values = vec![];
    for (_, value) in signal.iter_changes() {
        values.push(value.to_bit_string().unwrap());
    }

    assert_eq!(
        values,
        ["uuu", "111", "000", "001", "010", "111", "000", "001"]
    );
}

/// See: https://github.com/ekiwi/wellen/issues/32
/// SubtypeRecord was not implemented
#[test]
fn test_issue_32_ghw_subtype_record() {
    let filename = "inputs/ghdl/wellen_issue_32.ghw";
    let wave = read(filename).expect("failed to parse");
    let h = wave.hierarchy();

    // check that the record has been turned into a scope
    let scope = ["wellen_32", "spisub_s"];
    let s = &h[h.lookup_scope(&scope).unwrap()];
    assert_eq!(s.scope_type(), ScopeType::VhdlRecord);

    // check that all fields have been converted into signals
    let spi_signals = ["sclk", "mosi", "miso", "cs_n"];
    for signal in spi_signals {
        let var_id = h.lookup_var(&scope, &signal).unwrap();
        assert_eq!(
            h[var_id].full_name(h),
            format!("wellen_32.spisub_s.{signal}")
        );
    }
}

/// See: https://github.com/ekiwi/wellen/issues/6
/// signals resulting from a `generate-for` construct used to get aliasing scope names
#[test]
fn test_issue_6_generate_for_aliasing() {
    let filename = "inputs/ghdl/wellen_issue_6.ghw";
    let wave = read(filename).expect("failed to parse");
    let h = wave.hierarchy();
    let root_scope = &h[h.lookup_scope(&["wellen_6"]).unwrap()];
    let scopes = root_scope.scopes(h).collect::<Vec<_>>();
    let scope_names = scopes.iter().map(|s| h[*s].name(h)).collect::<Vec<_>>();
    assert_eq!(scope_names, ["gen(0)", "gen(1)", "gen(2)", "gen(3)"]);
}

/// See: https://github.com/ekiwi/wellen/issues/34
/// Previously we only handled subtype of constrained records
#[test]
fn test_issue_34_ghw_unconstrained_subtype_record() {
    let filename = "inputs/ghdl/wellen_issue_34.ghw";
    let wave = read(filename).expect("failed to parse");
    let h = wave.hierarchy();

    // find record scope
    let root_scope = &h[h.lookup_scope(&["wellen_34"]).unwrap()];
    let constrained_s = &h[root_scope.scopes(h).next().unwrap()];
    assert_eq!(constrained_s.full_name(h), "wellen_34.constrained_s");
    assert_eq!(constrained_s.scope_type(), ScopeType::VhdlRecord);

    // check record fields
    let p = ["wellen_34", "constrained_s"];

    let datavalid = &h[h.lookup_var(&p, &"datavalid").unwrap()];
    assert!(datavalid.is_1bit());
    assert!(datavalid.is_bit_vector());
    assert_eq!(datavalid.var_type(), VarType::StdLogic);
    assert_eq!(datavalid.vhdl_type_name(h), Some("std_logic"));

    let data = &h[h.lookup_var(&p, &"data").unwrap()];
    assert_eq!(data.length().unwrap(), 33);
    assert_eq!(data.index().unwrap().lsb(), 0);
    assert_eq!(data.index().unwrap().msb(), 32);
    assert_eq!(data.var_type(), VarType::StdLogicVector);
    assert_eq!(data.vhdl_type_name(h), Some("std_logic_vector"));

    let address = &h[h.lookup_var(&p, &"address").unwrap()];
    assert_eq!(address.length().unwrap(), 8);
    assert_eq!(address.index().unwrap().lsb(), 0);
    assert_eq!(address.index().unwrap().msb(), 7);
    assert_eq!(address.var_type(), VarType::StdLogicVector);
    assert_eq!(address.vhdl_type_name(h), Some("std_logic_vector"));
}

/// Check that we encode physical types properly as integers
#[test]
fn test_physical_type_parsing() {
    let filename = "inputs/ghdl/time_test.ghw";
    let mut wave = read(filename).expect("failed to parse");

    let var_names = wave
        .hierarchy()
        .iter_vars()
        .map(|v| v.name(wave.hierarchy()).to_string())
        .collect::<Vec<_>>();
    let signal_refs = wave
        .hierarchy()
        .iter_vars()
        .map(|v| v.signal_ref())
        .collect::<Vec<_>>();
    wave.load_signals(&signal_refs);
    let var_values: Vec<String> = signal_refs
        .iter()
        .filter_map(|s| wave.get_signal(*s))
        .flat_map(|s| s.iter_changes())
        .flat_map(|(_, sv)| sv.to_bit_string())
        .collect();

    assert_eq!(var_names, ["t1", "t2", "t3"]);
    assert_eq!(
        var_values,
        [
            "0000000000000000000000000000000000011101110011010110010100000000",
            "0000000000000111000110101111110101001001100011010000000000000000",
            "1000000000000000000000000000000000000000000000000000000000000000",
            "0000000000000111000110101111110101100111010110100110010100000000"
        ]
    )
}

/// See: https://github.com/ekiwi/wellen/issues/35
/// The underlying problem was that we needed to parse type bound if the element type
/// of the array that is pointed to by a subtype array is infinite.
#[test]
fn test_issue_35_ghw_failed_to_parse_rtik() {
    let filename = "inputs/ghdl/wellen_issue_35.ghw";
    let wave = read(filename).expect("failed to parse");
    let _h = wave.hierarchy();
}
