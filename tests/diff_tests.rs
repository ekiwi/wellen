// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use std::io::{BufRead, BufReader};
use waveform::{Hierarchy, HierarchyItem, ScopeType, SignalRef, TimescaleUnit, VarType, Waveform};

fn run_diff_test(vcd_filename: &str, fst_filename: &str) {
    {
        let wave = waveform::vcd::read_single_thread(vcd_filename).expect("Failed to load VCD");
        diff_test_one(vcd_filename, wave);
    }
    {
        println!("TODO: test FST")
        // let wave = waveform::fst::read(fst_filename);
        // diff_test_one(vcd_filename, wave);
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
        diff_hierarchy_item(ref_child, our_child, ours);
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
        ScopeType::Task => "task",
        ScopeType::Function => "function",
        ScopeType::Begin => "begin",
        ScopeType::Fork => "fork",
    }
}

fn waveform_var_type_to_string(tpe: VarType) -> &'static str {
    match tpe {
        VarType::Wire => "wire",
        VarType::Reg => "reg",
        VarType::Parameter => "parameter",
        VarType::Integer => "integer",
        VarType::String => "string",
        VarType::Event => "event",
        VarType::Real => "real",
        VarType::Supply0 => "supply0",
        VarType::Supply1 => "supply1",
        VarType::Time => "time",
        VarType::Tri => "tri",
        VarType::TriAnd => "triand",
        VarType::TriOr => "trior",
        VarType::TriReg => "trireg",
        VarType::Tri0 => "tri0",
        VarType::Tri1 => "tri1",
        VarType::WAnd => "wand",
        VarType::WOr => "wor",
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
                None => {} // nothing to check
                Some(size) => assert_eq!(ref_var.size, size),
            }
            match ref_var.index {
                None => assert!(our_var.index().is_none()),
                Some(vcd::ReferenceIndex::BitSelect(bit)) => {
                    assert_eq!(our_var.index().unwrap().msb(), bit);
                    assert_eq!(our_var.index().unwrap().lsb(), bit);
                }
                Some(vcd::ReferenceIndex::Range(msb, lsb)) => {
                    assert_eq!(our_var.index().unwrap().msb(), msb);
                    assert_eq!(our_var.index().unwrap().lsb(), lsb);
                }
            }
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
    let time_table = our.time_table();

    // iterate over reference VCD and compare with signals in our waveform
    let mut current_time = 0;
    let mut time_table_idx = 0;
    for cmd_res in ref_reader {
        match cmd_res.unwrap() {
            vcd::Command::Timestamp(new_time) => {
                if new_time > current_time {
                    time_table_idx += 1;
                    current_time = new_time;
                }
                assert_eq!(current_time, time_table[time_table_idx]);
            }
            vcd::Command::ChangeScalar(id, value) => {
                let signal_ref = vcd_lib_id_to_signal_ref(id);
                let our_value = our.get_signal_value_at(signal_ref, time_table_idx as u32);
                let our_value_str = our_value.to_bit_string().unwrap();
                assert_eq!(
                    our_value_str,
                    value.to_string(),
                    "{} ({:?}) = {} @ {} ({})",
                    id,
                    signal_ref,
                    value,
                    current_time,
                    our_value_str
                );
            }
            vcd::Command::ChangeVector(id, value) => {
                let signal_ref = vcd_lib_id_to_signal_ref(id);
                let our_value = our.get_signal_value_at(signal_ref, time_table_idx as u32);
                let our_value_str = our_value.to_bit_string().unwrap();
                if value.len() < our_value_str.len() {
                    let prefix_len = our_value_str.len() - value.len();
                    // we are zero / x extending, so our string might be longer
                    let suffix: String = our_value_str.chars().skip(prefix_len).collect();
                    assert_eq!(
                        suffix,
                        value.to_string(),
                        "{} ({:?}) = {} @ {} ({})",
                        id,
                        signal_ref,
                        value,
                        current_time,
                        our_value_str
                    );
                    let is_x_extended = suffix.chars().next().unwrap() == 'x';
                    let is_z_extended = suffix.chars().next().unwrap() == 'z';
                    for c in our_value_str.chars().take(prefix_len) {
                        if is_x_extended {
                            assert_eq!(c, 'x');
                        } else if is_z_extended {
                            assert_eq!(c, 'z');
                        } else {
                            assert_eq!(c, '0');
                        }
                    }
                } else {
                    assert_eq!(
                        our_value_str,
                        value.to_string(),
                        "{} ({:?}) = {} @ {} ({})",
                        id,
                        signal_ref,
                        value,
                        current_time,
                        our_value_str
                    );
                }
            }
            vcd::Command::ChangeReal(_, _) => {
                todo!("compare real")
            }
            vcd::Command::ChangeString(id, value) => {
                let signal_ref = vcd_lib_id_to_signal_ref(id);
                let our_value = our.get_signal_value_at(signal_ref, time_table_idx as u32);
                let our_value_str = our_value.to_string();
                assert_eq!(our_value_str, value);
            }
            vcd::Command::Begin(_) => {} // ignore
            vcd::Command::End(_) => {}   // ignore
            other => panic!("Unhandled command: {:?}", other),
        }
    }
}

fn vcd_lib_id_to_signal_ref(id: vcd::IdCode) -> SignalRef {
    let num = id_to_int(id.to_string().as_bytes()).unwrap();
    SignalRef::from_index(num as usize).unwrap()
}

const ID_CHAR_MIN: u8 = b'!';
const ID_CHAR_MAX: u8 = b'~';
const NUM_ID_CHARS: u64 = (ID_CHAR_MAX - ID_CHAR_MIN + 1) as u64;

/// Copied from https://github.com/kevinmehall/rust-vcd, licensed under MIT
fn id_to_int(id: &[u8]) -> Option<u64> {
    if id.is_empty() {
        return None;
    }
    let mut result = 0u64;
    for &i in id.iter().rev() {
        if !(ID_CHAR_MIN..=ID_CHAR_MAX).contains(&i) {
            return None;
        }
        let c = ((i - ID_CHAR_MIN) as u64) + 1;
        result = match result
            .checked_mul(NUM_ID_CHARS)
            .and_then(|x| x.checked_add(c))
        {
            None => return None,
            Some(value) => value,
        };
    }
    Some(result - 1)
}

#[test]
#[ignore] // TODO: this file has a delta cycle, i.e. the same signal (`/`) changes twice in the same cycle (35185000)
fn diff_aldec_spi_write() {
    run_diff_test(
        "inputs/aldec/SPI_Write.vcd",
        "inputs/aldec/SPI_Write.vcd.fst",
    );
}

#[test]
fn diff_amaranth_up_counter() {
    run_diff_test(
        "inputs/amaranth/up_counter.vcd",
        "inputs/amaranth/up_counter.vcd.fst",
    );
}

#[test]
fn diff_ghdl_alu() {
    run_diff_test("inputs/ghdl/alu.vcd", "inputs/ghdl/alu.vcd.fst");
}

#[test]
#[ignore] // TODO: this test requires VHDL 9-state support
fn diff_ghdl_idea() {
    run_diff_test("inputs/ghdl/idea.vcd", "inputs/ghdl/idea.vcd.fst");
}

#[test]
#[ignore] // TODO: this test requires VHDL 9-state support
fn diff_ghdl_pcpu() {
    run_diff_test("inputs/ghdl/pcpu.vcd", "inputs/ghdl/pcpu.vcd.fst");
}

#[test]
fn diff_gtkwave_perm_current() {
    run_diff_test(
        "inputs/gtkwave-analyzer/perm_current.vcd",
        "inputs/gtkwave-analyzer/perm_current.vcd.fst",
    );
}

#[test]
fn diff_icarus_cpu() {
    run_diff_test("inputs/icarus/CPU.vcd", "inputs/icarus/CPU.vcd.fst");
}

#[test]
fn diff_icarus_rv32_soc_tb() {
    run_diff_test(
        "inputs/icarus/rv32_soc_TB.vcd",
        "inputs/icarus/rv32_soc_TB.vcd.fst",
    );
}

#[test]
fn diff_icarus_test1() {
    run_diff_test("inputs/icarus/test1.vcd", "inputs/icarus/test1.vcd.fst");
}

#[test]
fn diff_model_sim_clkdiv2n_tb() {
    run_diff_test(
        "inputs/model-sim/clkdiv2n_tb.vcd",
        "inputs/model-sim/clkdiv2n_tb.vcd.fst",
    );
}

#[test]
#[ignore] // TODO: this file has a delta cycle, i.e. the same signal (`p`) changes twice in the same cycle (20000)
fn diff_model_sim_cpu_design() {
    run_diff_test(
        "inputs/model-sim/CPU_Design.msim.vcd",
        "inputs/model-sim/CPU_Design.msim.vcd.fst",
    );
}

#[test]
#[ignore] // TODO: this file declares a `real` signal and then emits strings ... :(
fn diff_my_hdl_sigmoid_tb() {
    run_diff_test(
        "inputs/my-hdl/sigmoid_tb.vcd",
        "inputs/my-hdl/sigmoid_tb.vcd.fst",
    );
}

#[test]
fn diff_my_hdl_simple_memory() {
    run_diff_test(
        "inputs/my-hdl/Simple_Memory.vcd",
        "inputs/my-hdl/Simple_Memory.vcd.fst",
    );
}

#[test]
#[ignore] // TODO: this file has a delta cycle, i.e. the same signal (`@`) changes several times in the same cycle (20)
fn diff_my_hdl_top() {
    run_diff_test("inputs/my-hdl/top.vcd", "inputs/my-hdl/top.vcd.fst");
}

#[test]
#[ignore] // TODO: add full real support
fn diff_ncsim_ffdiv_32bit_tb() {
    run_diff_test(
        "inputs/ncsim/ffdiv_32bit_tb.vcd",
        "inputs/ncsim/ffdiv_32bit_tb.vcd.fst",
    );
}

#[test]
fn diff_treadle_gcd() {
    run_diff_test("inputs/treadle/GCD.vcd", "inputs/treadle/GCD.vcd.fst");
}
