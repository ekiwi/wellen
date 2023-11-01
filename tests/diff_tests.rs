// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use std::io::BufReader;
use vcd::ScopeItem;
use waveform::hierarchy::Hierarchy;

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

    for item in ref_header.items.iter() {
        match item {
            ScopeItem::Scope(scope) => {
                println!("TODO: {} {}", scope.identifier, scope.scope_type);
            }
            ScopeItem::Var(var) => {
                println!(
                    "TODO: {} {} {:?} {}",
                    var.reference, var.size, var.index, var.var_type
                );
            }
            ScopeItem::Comment(_) => {} // we do not care about comments
            other => panic!("Unexpected scope item: {:?}", other),
        }
    }
}

#[test]
fn diff_treadle_gcd() {
    run_diff_test("inputs/treadle/GCD.vcd", "inputs/treadle/GCD.vcd.fst");
}
