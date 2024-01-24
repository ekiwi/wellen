// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use waveform::{
    Hierarchy, HierarchyItem, ScopeType, SignalRef, SignalValue, TimescaleUnit, VarType, Waveform,
};

fn run_diff_test(vcd_filename: &str, fst_filename: &str) {
    run_diff_test_internal(vcd_filename, fst_filename, false);
}

/// Skips trying to load the content with the `vcd` library. This is important for files
/// with 9-state values since these cannot be read by the `vcd` library.
fn run_load_test(vcd_filename: &str, fst_filename: &str) {
    run_diff_test_internal(vcd_filename, fst_filename, true);
}

fn run_diff_test_internal(vcd_filename: &str, fst_filename: &str, skip_content_comparison: bool) {
    {
        let wave = waveform::vcd::read_single_thread(vcd_filename).expect("Failed to load VCD");
        diff_test_one(vcd_filename, wave, skip_content_comparison);
    }
    {
        println!("TODO: test FST")
        // let wave = waveform::fst::read(fst_filename);
        // diff_test_one(vcd_filename, wave);
    }
}

fn diff_test_one(vcd_filename: &str, mut our: Waveform, skip_content_comparison: bool) {
    let mut ref_parser =
        vcd::Parser::new(BufReader::new(std::fs::File::open(vcd_filename).unwrap()));
    let ref_header = ref_parser.parse_header().unwrap();
    diff_hierarchy(our.hierarchy(), &ref_header);
    load_all_signals(&mut our);
    if !skip_content_comparison {
        diff_signals(&mut ref_parser, &mut our);
    }
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

fn load_all_signals(our: &mut Waveform) {
    let all_signals: Vec<_> = our
        .hierarchy()
        .get_unique_signals_vars()
        .iter()
        .flatten()
        .map(|v| v.signal_idx())
        .collect();
    our.load_signals(&all_signals);
}

fn diff_signals<R: BufRead>(ref_reader: &mut vcd::Parser<R>, our: &mut Waveform) {
    let time_table = our.time_table();

    // iterate over reference VCD and compare with signals in our waveform
    let mut current_time = 0;
    let mut time_table_idx = 0;
    let mut delta_counter = HashMap::new();
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
                let our_value = get_value(our, id, time_table_idx, &mut delta_counter);
                let our_value_str = our_value.to_bit_string().unwrap();
                let signal_ref = vcd_lib_id_to_signal_ref(id);
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
                let our_value = get_value(our, id, time_table_idx, &mut delta_counter);
                let our_value_str = our_value.to_bit_string().unwrap();
                let signal_ref = vcd_lib_id_to_signal_ref(id);
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
            vcd::Command::ChangeReal(id, value) => {
                let our_value = get_value(our, id, time_table_idx, &mut delta_counter);
                if let SignalValue::Real(our_real) = our_value {
                    assert_eq!(our_real, value);
                } else {
                    panic!("Expected real value, got: {our_value:?}");
                }
            }
            vcd::Command::ChangeString(id, value) => {
                let our_value = get_value(our, id, time_table_idx, &mut delta_counter);
                let our_value_str = our_value.to_string();
                assert_eq!(our_value_str, value);
            }
            vcd::Command::Begin(_) => {}   // ignore
            vcd::Command::End(_) => {}     // ignore
            vcd::Command::Comment(_) => {} // ignore
            other => panic!("Unhandled command: {:?}", other),
        }
    }
}

fn get_value<'a>(
    our: &'a Waveform,
    id: vcd::IdCode,
    time_table_idx: usize,
    delta_counter: &mut HashMap<SignalRef, u16>,
) -> SignalValue<'a> {
    let signal_ref = vcd_lib_id_to_signal_ref(id);
    let our_signal = our.get_signal(signal_ref).unwrap();
    let our_offset = our_signal.get_offset(time_table_idx as u32);
    assert!(
        our_offset.time_match,
        "Was not able to find an entry for {time_table_idx}"
    );
    // deal with delta cycles
    if our_offset.elements > 1 {
        let element = delta_counter.get(&signal_ref).map(|v| *v + 1).unwrap_or(0);
        if element == our_offset.elements - 1 {
            // last element
            delta_counter.remove(&signal_ref);
        } else {
            delta_counter.insert(signal_ref, element);
        }
        our_signal.get_value_at(&our_offset, element)
    } else {
        // no delta cycle -> just get the element and be happy!
        our_signal.get_value_at(&our_offset, 0)
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
fn diff_ghdl_idea() {
    // note that we cannot compare actual signal values since the vcd library is unable to
    // read the VCD files generated by ghdl (it encounters a parser error on changes to, e.g., `U`.
    run_load_test("inputs/ghdl/idea.vcd", "inputs/ghdl/idea.vcd.fst");
}

#[test]
fn diff_ghdl_pcpu() {
    // note that we cannot compare actual signal values since the vcd library is unable to
    // read the VCD files generated by ghdl (it encounters a parser error on changes to, e.g., `U`.
    run_load_test("inputs/ghdl/pcpu.vcd", "inputs/ghdl/pcpu.vcd.fst");
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
fn diff_my_hdl_top() {
    run_diff_test("inputs/my-hdl/top.vcd", "inputs/my-hdl/top.vcd.fst");
}

#[test]
fn diff_ncsim_ffdiv_32bit_tb() {
    run_diff_test(
        "inputs/ncsim/ffdiv_32bit_tb.vcd",
        "inputs/ncsim/ffdiv_32bit_tb.vcd.fst",
    );
}

#[test]
fn diff_quartus_mips_hardware() {
    run_diff_test(
        "inputs/quartus/mipsHardware.vcd",
        "inputs/quartus/mipsHardware.vcd.fst",
    );
}

#[test]
#[ignore] // TODO: gets stuck!
fn diff_quartus_wave_registradores() {
    run_diff_test(
        "inputs/quartus/wave_registradores.vcd",
        "inputs/quartus/wave_registradores.vcd.fst",
    );
}

#[test]
fn diff_questa_sim_dump() {
    run_diff_test(
        "inputs/questa-sim/dump.vcd",
        "inputs/questa-sim/dump.vcd.fst",
    );
}

#[test]
fn diff_questa_sim_test() {
    run_diff_test(
        "inputs/questa-sim/test.vcd",
        "inputs/questa-sim/test.vcd.fst",
    );
}

#[test]
fn diff_riviera_pro_dump() {
    run_diff_test(
        "inputs/riviera-pro/dump.vcd",
        "inputs/riviera-pro/dump.vcd.fst",
    );
}

#[test]
fn diff_systemc_waveform() {
    run_diff_test(
        "inputs/systemc/waveform.vcd",
        "inputs/systemc/waveform.vcd.fst",
    );
}

#[test]
fn diff_treadle_gcd() {
    run_diff_test("inputs/treadle/GCD.vcd", "inputs/treadle/GCD.vcd.fst");
}

#[test]
fn diff_vcs_apb_uvm_new() {
    run_diff_test(
        "inputs/vcs/Apb_slave_uvm_new.vcd",
        "inputs/vcs/Apb_slave_uvm_new.vcd.fst",
    );
}

#[test]
fn diff_vcs_datapath_log() {
    run_diff_test(
        "inputs/vcs/datapath_log.vcd",
        "inputs/vcs/datapath_log.vcd.fst",
    );
}

#[test]
fn diff_vcs_processor() {
    run_diff_test("inputs/vcs/processor.vcd", "inputs/vcs/processor.vcd.fst");
}

#[test] // TODO: takes longer than expected! (only a 14M VCD)
fn diff_verilator_swerv1() {
    run_diff_test(
        "inputs/verilator/swerv1.vcd",
        "inputs/verilator/swerv1.vcd.fst",
    );
}

#[test]
fn diff_verilator_vlt_dump() {
    run_diff_test(
        "inputs/verilator/vlt_dump.vcd",
        "inputs/verilator/vlt_dump.vcd.fst",
    );
}

#[test]
fn diff_vivado_iladata() {
    run_diff_test("inputs/vivado/iladata.vcd", "inputs/vivado/iladata.vcd.fst");
}

#[test]
#[ignore] // TODO: triggers an assertion
fn diff_xilinx_isim_test() {
    run_diff_test(
        "inputs/xilinx_isim/test.vcd",
        "inputs/xilinx_isim/test.vcd.fst",
    );
}

#[test] // TODO: takes longer than expected! (only a 8.7M VCD)
fn diff_xilinx_isim_test1() {
    run_diff_test(
        "inputs/xilinx_isim/test1.vcd",
        "inputs/xilinx_isim/test1.vcd.fst",
    );
}

#[test]
#[ignore] // TODO: triggers an assertion!
fn diff_xilinx_isim_test2x2_regex22_string1() {
    run_diff_test(
        "inputs/xilinx_isim/test2x2_regex22_string1.vcd",
        "inputs/xilinx_isim/test2x2_regex22_string1.vcd.fst",
    );
}

#[test]
#[ignore] // TODO: triggers an assertion!
fn diff_scope_with_comment() {
    run_diff_test(
        "inputs/scope_with_comment.vcd",
        "inputs/scope_with_comment.vcd.fst",
    );
}
