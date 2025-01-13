// Copyright 2023-2024 The Regents of the University of California
// Copyright 2024-2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>
//
// here we are comparing fst and ghw files, both loaded with wellen

mod utils;
use utils::{diff_signals, load_all_signals};
use wellen::simple::*;
use wellen::*;

fn run_diff_test(ghw_filename: &str, fst_filename: &str) {
    let mut ghw_wave = read(ghw_filename).expect("failed to load GHW file!");
    assert_eq!(ghw_wave.hierarchy().file_format(), FileFormat::Ghw);
    let mut fst_wave = read(fst_filename).expect("failed to load FST file!");
    assert_eq!(fst_wave.hierarchy().file_format(), FileFormat::Fst);

    let time_factor = diff_hierarchy(ghw_wave.hierarchy(), fst_wave.hierarchy());

    // diff signals
    load_all_signals(&mut ghw_wave);
    load_all_signals(&mut fst_wave);
    diff_signals(&mut ghw_wave, &mut fst_wave, time_factor);
}

fn diff_hierarchy(ghw: &Hierarchy, fst: &Hierarchy) -> u64 {
    // note: date and version may differ because different simulators are used
    // note: the timescale is also not as well defined as in Verilog

    // time scaling, since the ghw will be in femto seconds
    let time_factor = as_fs(fst.timescale().expect("no timescale"));

    // compare actual hierarchy entries
    // note: we only focus on the fst top scope, since ghw tends to include empty scopes like "standard"
    let fst_top = fst.first_scope().unwrap();
    let ghw_top_ref = ghw
        .scopes()
        .find(|s| ghw[*s].name(ghw) == fst_top.name(fst))
        .unwrap();
    let ghw_top = &ghw[ghw_top_ref];
    diff_hierarchy_item(
        ScopeOrVar::Scope(ghw_top),
        ghw,
        ScopeOrVar::Scope(fst_top),
        fst,
    );

    time_factor
}

fn diff_hierarchy_item(
    ghw_item: ScopeOrVar,
    ghw: &Hierarchy,
    fst_item: ScopeOrVar,
    fst: &Hierarchy,
) {
    match (ghw_item, fst_item) {
        (ScopeOrVar::Scope(g), ScopeOrVar::Scope(f)) => {
            assert_eq!(g.name(ghw), f.name(fst));
            assert_eq!(g.component(ghw), f.component(fst));
            assert_eq!(g.scope_type(), f.scope_type());
            // ghw has no way to provide source locs, so we aren't comparing here
            for (ghw_item, fst_item) in g.items(ghw).zip(f.items(fst)) {
                diff_hierarchy_item(ghw_item.deref(ghw), ghw, fst_item.deref(fst), fst);
            }
        }
        (ScopeOrVar::Var(g), ScopeOrVar::Var(f)) => {
            assert_eq!(g.name(ghw), f.name(fst));
            // in the fst all enums are encoded as strings
            if g.enum_type(ghw).is_some() {
                assert_eq!(f.var_type(), VarType::String);
            } else {
                // the fst sometime confuses ulogic and logic
                if f.var_type() == VarType::StdULogic {
                    assert!(
                        g.var_type() == VarType::StdULogic || g.var_type() == VarType::StdLogic
                    );
                } else {
                    assert_eq!(g.var_type(), f.var_type(), "{}", g.full_name(ghw));
                }
                assert_eq!(g.length(), f.length());
            }
            // check to see if signal refs are the same, this is not relly guaranteed, but it makes comparing signals nicer!
            assert_eq!(g.signal_ref(), f.signal_ref());
            assert_eq!(g.index(), f.index());
            assert_eq!(g.direction(), f.direction());

            // VHDL is case insensitive
            assert_eq!(
                g.vhdl_type_name(ghw).map(|n| n.to_ascii_lowercase()),
                f.vhdl_type_name(fst).map(|n| n.to_ascii_lowercase()),
                "{} {:?} {:?}",
                g.full_name(ghw),
                g.var_type(),
                f.var_type()
            );
        }
        (ghw, fst) => {
            panic!("Unexpected combination of scope items: {ghw:?} (ghw) vs. {fst:?} (fst)",)
        }
    }
}

fn as_fs(timescale: Timescale) -> u64 {
    let factor = match timescale.unit {
        TimescaleUnit::ZeptoSeconds => unreachable!("should not get here!"),
        TimescaleUnit::AttoSeconds => unreachable!("should not get here!"),
        TimescaleUnit::FemtoSeconds => 1,
        TimescaleUnit::PicoSeconds => 1000,
        TimescaleUnit::NanoSeconds => 1000 * 1000,
        TimescaleUnit::MicroSeconds => 1000 * 1000 * 1000,
        TimescaleUnit::MilliSeconds => 1000 * 1000 * 1000 * 1000,
        TimescaleUnit::Seconds => 1000 * 1000 * 1000 * 1000 * 1000,
        TimescaleUnit::Unknown => unreachable!("should not get here!"),
    };
    timescale.factor as u64 * factor
}

#[test]
fn diff_ghdl_oscar_test() {
    run_diff_test("inputs/ghdl/oscar/test.ghw", "inputs/ghdl/oscar/vhdl3.fst");
}
