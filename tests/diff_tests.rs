// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use std::io::BufReader;
use vcd::ScopeItem;
use waveform::hierarchy::{Hierarchy, HierarchyItem, ScopeType, SignalLength, VarType};

fn run_diff_test(vcd_filename: &str, fst_filename: &str) {
    let (vcd_hierarchy, vcd_source) = waveform::vcd::read(vcd_filename);
    let (fst_hierarchy, fst_source) = waveform::fst::read(fst_filename);
    let mut ref_parser =
        vcd::Parser::new(BufReader::new(std::fs::File::open(vcd_filename).unwrap()));
    let ref_header = ref_parser.parse_header().unwrap();
    diff_hierarchy(&vcd_hierarchy, &ref_header);
    diff_hierarchy(&fst_hierarchy, &ref_header);
}

fn diff_hierarchy(ours: &Hierarchy, ref_header: &vcd::Header) {
    println!("{:?}", ref_header.version);
    println!("{:?}", ref_header.date);
    println!("{:?}", ref_header.timescale);

    for (ref_child, our_child) in itertools::zip_eq(
        ref_header
            .items
            .iter()
            .filter(|i| !matches!(i, ScopeItem::Comment(_))),
        ours.items(),
    ) {
        diff_hierarchy_item(ref_child, our_child, ours)
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

fn diff_hierarchy_item(ref_item: &ScopeItem, our_item: HierarchyItem, our_hier: &Hierarchy) {
    match (ref_item, our_item) {
        (ScopeItem::Scope(ref_scope), HierarchyItem::Scope(our_scope)) => {
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
                    .filter(|i| !matches!(i, ScopeItem::Comment(_))),
                our_scope.items(our_hier),
            ) {
                diff_hierarchy_item(ref_child, our_child, our_hier)
            }
        }
        (ScopeItem::Var(ref_var), HierarchyItem::Var(our_var)) => {
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
        (ScopeItem::Comment(_), _) => {} // we do not care about comments
        (other_ref, our) => panic!(
            "Unexpected combination of scope items: {:?} (expected) vs. {:?}",
            other_ref, our
        ),
    }
}

#[test]
fn diff_treadle_gcd() {
    run_diff_test("inputs/treadle/GCD.vcd", "inputs/treadle/GCD.vcd.fst");
}
