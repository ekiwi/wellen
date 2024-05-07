// Copyright 2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>
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
    let top = h.scopes().find(|s| h.get(*s).name(h) == "test2").unwrap();
    let bbb = h
        .get(top)
        .vars(h)
        .find(|v| h.get(*v).name(h) == "bbb")
        .unwrap();
    let ee = h
        .get(top)
        .vars(h)
        .find(|v| h.get(*v).name(h) == "ee")
        .unwrap();

    // signal bbb: boolean
    {
        let (enum_name, enum_lits) = h.get(bbb).enum_type(h).unwrap();
        assert_eq!(enum_name, "boolean");
        assert_eq!(enum_lits, [("0", "false"), ("1", "true")]);
    }

    // type e is(foo, bar, tada);
    // signal ee: e
    {
        let (enum_name, enum_lits) = h.get(ee).enum_type(h).unwrap();
        assert_eq!(enum_name, "e");
        assert_eq!(h.get(ee).length().unwrap(), 2);
        // 2 bits are used for the encoding and thus everything is padded to a width of 2
        assert_eq!(enum_lits, [("00", "foo"), ("01", "bar"), ("10", "tada")]);
    }

    // make sure that all enum values are binary
    let ee_signal_ref = h.get(ee).signal_ref();
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
    let signal_ref = wave.hierarchy().get(var_id).signal_ref();
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
