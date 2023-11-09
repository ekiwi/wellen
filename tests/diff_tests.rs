// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use std::io::{BufRead, BufReader};
use waveform::{
    Hierarchy, HierarchyItem, ScopeType, SignalLength, TimescaleUnit, VarType, Waveform,
};

fn run_diff_test(vcd_filename: &str, fst_filename: &str) {
    {
        let wave = waveform::vcd::read(vcd_filename).expect("Failed to load VCD");
        diff_test_one(vcd_filename, wave);
    }
    {
        let wave = waveform::fst::read(fst_filename);
        diff_test_one(vcd_filename, wave);
    }
}

fn diff_test_one(vcd_filename: &str, mut our: Waveform) {
    let mut ref_parser =
        vcd::Parser::new(BufReader::new(std::fs::File::open(vcd_filename).unwrap()));
    let ref_header = ref_parser.parse_header().unwrap();
    diff_hierarchy(our.hierarchy(), &ref_header);
    diff_signals(&mut ref_parser, &mut our);
}

fn diff_hierarchy(ours: &Hierarchy, ref_header: &vcd::Header) {
    diff_meta(ours, ref_header);

    for (ref_child, our_child) in itertools::zip_eq(
        ref_header
            .items
            .iter()
            .filter(|i| !matches!(i, vcd::ScopeItem::Comment(_))),
        ours.items(),
    ) {
        diff_hierarchy_item(ref_child, our_child, ours)
    }
}

fn diff_meta(ours: &Hierarchy, ref_header: &vcd::Header) {
    match &ref_header.version {
        None => assert!(ours.version().is_empty()),
        Some(version) => assert_eq!(version, ours.version()),
    }

    match &ref_header.date {
        None => assert!(ours.date().is_empty()),
        Some(date) => assert_eq!(date, ours.date()),
    }

    match ref_header.timescale {
        None => assert!(ours.timescale().is_none()),
        Some((factor, unit)) => {
            let our_time = ours.timescale().unwrap();
            assert_eq!(factor, our_time.factor);
            match unit {
                vcd::TimescaleUnit::S => assert_eq!(our_time.unit, TimescaleUnit::Seconds),
                vcd::TimescaleUnit::MS => assert_eq!(our_time.unit, TimescaleUnit::MilliSeconds),
                vcd::TimescaleUnit::US => assert_eq!(our_time.unit, TimescaleUnit::MicroSeconds),
                vcd::TimescaleUnit::NS => assert_eq!(our_time.unit, TimescaleUnit::NanoSeconds),
                vcd::TimescaleUnit::PS => assert_eq!(our_time.unit, TimescaleUnit::PicoSeconds),
                vcd::TimescaleUnit::FS => assert_eq!(our_time.unit, TimescaleUnit::FemtoSeconds),
            }
        }
    }
}

fn waveform_scope_type_to_string(tpe: ScopeType) -> &'static str {
    match tpe {
        ScopeType::Module => "module",
        ScopeType::Todo => "todo",
    }
}

fn waveform_var_type_to_string(tpe: VarType) -> &'static str {
    match tpe {
        VarType::Wire => "wire",
        VarType::Todo => "todo",
    }
}

fn diff_hierarchy_item(ref_item: &vcd::ScopeItem, our_item: HierarchyItem, our_hier: &Hierarchy) {
    match (ref_item, our_item) {
        (vcd::ScopeItem::Scope(ref_scope), HierarchyItem::Scope(our_scope)) => {
            assert_eq!(ref_scope.identifier, our_scope.name(our_hier));
            assert_eq!(
                ref_scope.scope_type.to_string(),
                waveform_scope_type_to_string(our_scope.scope_type())
            );
            println!("Scope");
            for (ref_child, our_child) in itertools::zip_eq(
                ref_scope
                    .items
                    .iter()
                    .filter(|i| !matches!(i, vcd::ScopeItem::Comment(_))),
                our_scope.items(our_hier),
            ) {
                diff_hierarchy_item(ref_child, our_child, our_hier)
            }
        }
        (vcd::ScopeItem::Var(ref_var), HierarchyItem::Var(our_var)) => {
            assert_eq!(ref_var.reference, our_var.name(our_hier));
            assert_eq!(
                ref_var.var_type.to_string(),
                waveform_var_type_to_string(our_var.var_type())
            );
            match our_var.length() {
                SignalLength::Variable => panic!("TODO: check for varlen!"),
                SignalLength::Fixed(size) => assert_eq!(ref_var.size, size.get()),
            }
            assert!(ref_var.index.is_none(), "TODO: expose index");
        }
        (vcd::ScopeItem::Comment(_), _) => {} // we do not care about comments
        (other_ref, our) => panic!(
            "Unexpected combination of scope items: {:?} (expected) vs. {:?}",
            other_ref, our
        ),
    }
}

fn diff_signals<R: BufRead>(ref_reader: &mut vcd::Parser<R>, our: &mut Waveform) {
    // load all signals
    let all_signals: Vec<_> = our
        .hierarchy()
        .get_unique_signals_vars()
        .iter()
        .flatten()
        .map(|v| v.signal_idx())
        .collect();
    our.load_signals(&all_signals);

    println!("TODO")
}

#[test]
fn diff_treadle_gcd() {
    run_diff_test("inputs/treadle/GCD.vcd", "inputs/treadle/GCD.vcd.fst");
}
