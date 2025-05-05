// Copyright 2023-2024 The Regents of the University of California
// Copyright 2024-2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use rustc_hash::FxHashMap;
use std::io::{BufRead, BufReader};
use wellen::simple::*;
use wellen::*;

fn run_diff_test(vcd_filename: &str, fst_filename: &str) {
    run_diff_test_internal(vcd_filename, Some(fst_filename), false, LoadType::FromFile);
}

fn run_diff_test_from_bytes(vcd_filename: &str, fst_filename: &str) {
    run_diff_test_internal(vcd_filename, Some(fst_filename), false, LoadType::FromBytes);
}

fn run_diff_test_from_generic_file(vcd_filename: &str, fst_filename: &str) {
    run_diff_test_internal(
        vcd_filename,
        Some(fst_filename),
        false,
        LoadType::FromFileGeneric,
    );
}

fn run_diff_test_vcd_only(vcd_filename: &str) {
    run_diff_test_internal(vcd_filename, None, false, LoadType::FromFile);
}

/// Skips trying to load the content with the `vcd` library. This is important for files
/// with 9-state values since these cannot be read by the `vcd` library.
fn run_load_test(vcd_filename: &str, fst_filename: &str) {
    run_diff_test_internal(vcd_filename, Some(fst_filename), true, LoadType::FromFile);
}

fn run_load_test_vcd(vcd_filename: &str) {
    run_diff_test_internal(vcd_filename, None, true, LoadType::FromFile);
}

#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
enum LoadType {
    FromFile,
    FromBytes,
    /// Uses the read that is generic over any ReadBuf to read from file instead of the
    /// specialized implementation which will mmap for VCDs
    FromFileGeneric,
}

fn run_diff_test_internal(
    vcd_filename: &str,
    fst_filename: Option<&str>,
    skip_content_comparison: bool,
    load_tpe: LoadType,
) {
    {
        let single_thread = LoadOptions {
            multi_thread: false,
            ..Default::default()
        };
        let wave = read_with_options(vcd_filename, &single_thread)
            .expect("Failed to load VCD with a single thread");
        diff_test_one(vcd_filename, wave, skip_content_comparison);
    }

    match load_tpe {
        LoadType::FromFile => {
            let wave = read(vcd_filename).expect("Failed to load VCD with multiple threads");
            diff_test_one(vcd_filename, wave, skip_content_comparison);
        }
        LoadType::FromBytes => {
            let bytes = std::io::Cursor::new(std::fs::read(vcd_filename).expect("failed"));
            let wave = read_from_reader(bytes)
                .expect("Failed to load VCD with multiple threads from bytes");
            diff_test_one(vcd_filename, wave, skip_content_comparison);
        }
        LoadType::FromFileGeneric => {
            let file = std::io::BufReader::new(std::fs::File::open(vcd_filename).expect("failed"));
            let wave = read_from_reader(file)
                .expect("Failed to load VCD with multiple threads from generic file");
            diff_test_one(vcd_filename, wave, skip_content_comparison);
        }
    }
    if let Some(fst_filename) = fst_filename {
        match load_tpe {
            LoadType::FromFile => {
                let wave = read(fst_filename).expect("Failed to load FST");
                diff_test_one(vcd_filename, wave, skip_content_comparison);
            }
            LoadType::FromBytes => {
                let bytes = std::io::Cursor::new(std::fs::read(fst_filename).expect("failed"));
                let wave = read_from_reader(bytes)
                    .expect("Failed to load FST with multiple threads from bytes");
                diff_test_one(fst_filename, wave, skip_content_comparison);
            }
            LoadType::FromFileGeneric => {
                let file =
                    std::io::BufReader::new(std::fs::File::open(fst_filename).expect("failed"));
                let wave = read_from_reader(file)
                    .expect("Failed to load FST with multiple threads from generic file");
                diff_test_one(fst_filename, wave, skip_content_comparison);
            }
        }
    }
}

fn diff_test_one(vcd_filename: &str, mut our: Waveform, skip_content_comparison: bool) {
    let mut ref_parser =
        ::vcd::Parser::new(BufReader::new(std::fs::File::open(vcd_filename).unwrap()));
    let ref_header = match ref_parser.parse_header() {
        Ok(parsed) => parsed,
        Err(e) => {
            println!("WARN: skipping difftest because file cannot be parsed by the (3rd party!) rust vcd library");
            println!("{e:?}");
            return;
        }
    };
    let mut id_map = FxHashMap::default();
    diff_hierarchy(our.hierarchy(), &ref_header, &mut id_map);
    load_all_signals(&mut our);
    if !skip_content_comparison {
        diff_signals(&mut ref_parser, &mut our, &id_map);
    }
}

fn diff_hierarchy(
    ours: &Hierarchy,
    ref_header: &::vcd::Header,
    id_map: &mut FxHashMap<::vcd::IdCode, SignalRef>,
) {
    diff_meta(ours, ref_header);

    for (ref_child, our_child) in itertools::zip_eq(
        ref_header
            .items
            .iter()
            .filter(|i| !matches!(i, ::vcd::ScopeItem::Comment(_))),
        ours.items(),
    ) {
        diff_hierarchy_item(ref_child, our_child.deref(ours), ours, id_map);
    }
}

fn diff_meta(ours: &Hierarchy, ref_header: &::vcd::Header) {
    match &ref_header.version {
        None => {
            if ours.file_format() == FileFormat::Vcd {
                assert!(ours.version().is_empty(), "{}", ours.version());
            }
        }
        Some(version) => assert_eq!(version, ours.version()),
    }

    match &ref_header.date {
        None => {
            if ours.file_format() == FileFormat::Vcd {
                assert!(ours.date().is_empty(), "{}", ours.date());
            }
        }
        Some(date) => assert_eq!(date, ours.date()),
    }

    match ref_header.timescale {
        None => assert!(ours.timescale().is_none()),
        Some((factor, unit)) => {
            let our_time = ours.timescale().unwrap();
            assert_eq!(factor, our_time.factor);
            match unit {
                ::vcd::TimescaleUnit::S => assert_eq!(our_time.unit, TimescaleUnit::Seconds),
                ::vcd::TimescaleUnit::MS => assert_eq!(our_time.unit, TimescaleUnit::MilliSeconds),
                ::vcd::TimescaleUnit::US => assert_eq!(our_time.unit, TimescaleUnit::MicroSeconds),
                ::vcd::TimescaleUnit::NS => assert_eq!(our_time.unit, TimescaleUnit::NanoSeconds),
                ::vcd::TimescaleUnit::PS => assert_eq!(our_time.unit, TimescaleUnit::PicoSeconds),
                ::vcd::TimescaleUnit::FS => assert_eq!(our_time.unit, TimescaleUnit::FemtoSeconds),
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
        other => todo!("Convert {other:?} to string."),
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
        VarType::Port => "port",
        VarType::Bit => "bit",
        VarType::Logic => "logic",
        VarType::Int => "int",
        VarType::Enum => "enum",
        other => todo!("Convert {other:?} to string."),
    }
}

fn diff_hierarchy_item(
    ref_item: &::vcd::ScopeItem,
    our_item: ScopeOrVar,
    our_hier: &Hierarchy,
    id_map: &mut FxHashMap<::vcd::IdCode, SignalRef>,
) {
    match (ref_item, our_item) {
        (::vcd::ScopeItem::Scope(ref_scope), ScopeOrVar::Scope(our_scope)) => {
            assert_eq!(ref_scope.identifier, our_scope.name(our_hier));
            assert_eq!(
                ref_scope.scope_type.to_string(),
                waveform_scope_type_to_string(our_scope.scope_type())
            );
            for (ref_child, our_child) in itertools::zip_eq(
                ref_scope
                    .items
                    .iter()
                    .filter(|i| !matches!(i, ::vcd::ScopeItem::Comment(_))),
                our_scope.items(our_hier),
            ) {
                diff_hierarchy_item(ref_child, our_child.deref(our_hier), our_hier, id_map)
            }
        }
        (::vcd::ScopeItem::Var(ref_var), ScopeOrVar::Var(our_var)) => {
            id_map.insert(ref_var.code, our_var.signal_ref());
            // this happens because simulators like GHDL forget the space before the index
            let ref_name_contains_index =
                ref_var.reference.contains('[') && ref_var.reference.contains(']');
            if !ref_name_contains_index {
                assert_eq!(ref_var.reference, our_var.name(our_hier));
            } else {
                assert!(
                    ref_var.reference.starts_with(our_var.name(our_hier)),
                    "{} vs. {} (OUR)",
                    ref_var.reference,
                    our_var.name(our_hier)
                );
            }
            assert_eq!(
                ref_var.var_type.to_string(),
                waveform_var_type_to_string(our_var.var_type())
            );
            match our_var.length() {
                None => {} // nothing to check
                Some(size) => {
                    if ref_var.size == 0 {
                        assert_eq!(1, size)
                    } else {
                        assert_eq!(ref_var.size, size)
                    }
                }
            }
            match ref_var.index {
                None => assert!(our_var.index().is_none() || ref_name_contains_index),
                Some(::vcd::ReferenceIndex::BitSelect(bit)) => {
                    assert_eq!(our_var.index().unwrap().msb(), bit as i64);
                    assert_eq!(our_var.index().unwrap().lsb(), bit as i64);
                }
                Some(::vcd::ReferenceIndex::Range(msb, lsb)) => {
                    let (our_msb, our_lsb) = (
                        our_var.index().unwrap().msb(),
                        our_var.index().unwrap().lsb(),
                    );
                    assert_eq!(our_msb, msb as i64);
                    assert_eq!(our_lsb, lsb as i64);
                }
            }
        }
        (::vcd::ScopeItem::Comment(_), _) => {} // we do not care about comments
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
        .map(|v| v.signal_ref())
        .collect();
    our.load_signals(&all_signals);
}

fn diff_signals<R: BufRead>(
    ref_reader: &mut ::vcd::Parser<R>,
    our: &mut Waveform,
    id_map: &FxHashMap<::vcd::IdCode, SignalRef>,
) {
    let time_table = our.time_table();

    // iterate over reference VCD and compare with signals in our waveform
    let mut time_table_idx = 0;
    let mut current_time: Option<u64> = None;
    let mut delta_counter = FxHashMap::default();
    for cmd_res in ref_reader {
        match cmd_res.unwrap() {
            ::vcd::Command::Timestamp(new_time) => {
                match current_time {
                    None => {
                        current_time = Some(new_time);
                        if time_table[time_table_idx] < new_time {
                            time_table_idx += 1;
                        }
                    }
                    Some(time) => {
                        if new_time > time {
                            time_table_idx += 1;
                            current_time = Some(new_time);
                        }
                    }
                }
                assert_eq!(current_time.unwrap(), time_table[time_table_idx]);
            }
            ::vcd::Command::ChangeScalar(id, value) => {
                let signal_ref = id_map[&id];
                let our_value = get_value(our, signal_ref, time_table_idx, &mut delta_counter);
                let our_value_str = our_value.to_bit_string().unwrap();
                assert_eq!(
                    our_value_str,
                    value.to_string(),
                    "{} ({:?}) = {} @ {} (idx: {}) ({})",
                    id,
                    signal_ref,
                    value,
                    current_time.unwrap_or(0),
                    time_table_idx,
                    our_value_str
                );
            }
            ::vcd::Command::ChangeVector(id, value) => {
                let signal_ref = id_map[&id];
                assert_eq!(current_time.unwrap_or(0), time_table[time_table_idx]);
                let our_value = get_value(our, signal_ref, time_table_idx, &mut delta_counter);
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
                        current_time.unwrap_or(0),
                        our_value_str
                    );
                    let is_x_extended = suffix.starts_with('x');
                    let is_z_extended = suffix.starts_with('z');
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
                        current_time.unwrap_or(0),
                        our_value_str
                    );
                }
            }
            ::vcd::Command::ChangeReal(id, value) => {
                let signal_ref = id_map[&id];
                assert_eq!(current_time.unwrap_or(0), time_table[time_table_idx]);
                let our_value = get_value(our, signal_ref, time_table_idx, &mut delta_counter);
                if let SignalValue::Real(our_real) = our_value {
                    assert_eq!(our_real, value);
                } else {
                    panic!("Expected real value, got: {our_value:?}");
                }
            }
            ::vcd::Command::ChangeString(id, value) => {
                let signal_ref = id_map[&id];
                assert_eq!(current_time.unwrap_or(0), time_table[time_table_idx]);
                let our_value = get_value(our, signal_ref, time_table_idx, &mut delta_counter);
                let our_value_str = our_value.to_string();
                assert_eq!(our_value_str, value);
            }
            ::vcd::Command::Begin(_) => {}   // ignore
            ::vcd::Command::End(_) => {}     // ignore
            ::vcd::Command::Comment(_) => {} // ignore
            other => panic!("Unhandled command: {:?}", other),
        }
    }
}

fn get_value<'a>(
    our: &'a Waveform,
    signal_ref: SignalRef,
    time_table_idx: usize,
    delta_counter: &mut FxHashMap<SignalRef, u16>,
) -> SignalValue<'a> {
    let our_signal = our.get_signal(signal_ref).unwrap();
    let our_offset = our_signal.get_offset(time_table_idx as u32).unwrap();
    // deal with delta cycles
    if our_offset.elements > 1 {
        if our_offset.time_match {
            let element = delta_counter.get(&signal_ref).map(|v| *v + 1).unwrap_or(0);
            if element == our_offset.elements - 1 {
                // last element
                delta_counter.remove(&signal_ref);
            } else {
                delta_counter.insert(signal_ref, element);
            }
            our_signal.get_value_at(&our_offset, element)
        } else {
            // if we are looking at a past offset, we always want to get the last element
            our_signal.get_value_at(&our_offset, our_offset.elements - 1)
        }
    } else {
        // no delta cycle -> just get the element and be happy!
        our_signal.get_value_at(&our_offset, 0)
    }
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

/// from https://github.com/ekiwi/wellen/issues/3
#[test]
fn diff_gameroy_trace() {
    run_diff_test_vcd_only("inputs/gameroy/trace_prefix.vcd");
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
#[ignore] // currently fails since our diff test does not take de-duplicated scopes into account
fn diff_gtkwave_perm_current() {
    run_diff_test(
        "inputs/gtkwave-analyzer/perm_current.vcd",
        "inputs/gtkwave-analyzer/perm_current.vcd.fst",
    );
}

#[test]
fn diff_jtag_atxmega256a3u_bmda() {
    run_diff_test_vcd_only("inputs/jtag/atxmega256a3u-bmda-jtag.vcd");
}

#[test]
fn diff_icarus_cpu() {
    run_diff_test("inputs/icarus/CPU.vcd", "inputs/icarus/CPU.vcd.fst");
}

#[test]
fn diff_icarus_dc_crossbar() {
    run_diff_test_vcd_only("inputs/icarus/DCCrossbar.vcd");
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

/// tests loading a VCD and FST from a Vec<u8>
#[test]
fn diff_icarus_test1_from_bytes() {
    run_diff_test_from_bytes("inputs/icarus/test1.vcd", "inputs/icarus/test1.vcd.fst");
}

/// tests loading a VCD and FST from a BufReader<File> without using mmap
#[test]
fn diff_icarus_test1_from_generic_file() {
    run_diff_test_from_generic_file("inputs/icarus/test1.vcd", "inputs/icarus/test1.vcd.fst");
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

/// This file was provided by Augusto Fraga Giachero in the following issue:
/// https://github.com/ekiwi/wellen/issues/10
#[test]
fn diff_nvc_xwb_fofb_shaper_filt_tb_arrays() {
    run_diff_test(
        // generated with fst2vcd
        "inputs/nvc/xwb_fofb_shaper_filt_tb_arrays.fst.vcd",
        "inputs/nvc/xwb_fofb_shaper_filt_tb_arrays.fst",
    );
}

/// Apparently pymtl3 emits bit-vectors differently than every other simulator we support/
/// This was pointed out in: https://gitlab.com/surfer-project/surfer/-/issues/202
/// We can only run a load test since the VCD library we use for diff testing also rejects this encoding.
#[test]
fn diff_pymtl3_cgra() {
    run_load_test_vcd("inputs/pymtl3/CGRA.vcd")
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
#[ignore] // currently fails since our diff test does not take de-duplicated scopes into account
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
fn diff_questa_sim_issue_57() {
    run_diff_test_vcd_only("inputs/questa-sim/wellen-issue-57-uart.vcd");
}

#[test]
fn diff_riviera_pro_dump() {
    run_diff_test(
        "inputs/riviera-pro/dump.vcd",
        "inputs/riviera-pro/dump.vcd.fst",
    );
}

#[test]
fn diff_sigrok_libsigrok() {
    run_diff_test_vcd_only("inputs/sigrok/libsigrok.vcd");
}

/// This VCD file contains very large signal references which cannot directly be converted to a SignalRef
#[test]
fn diff_specs_tracefile() {
    run_diff_test_vcd_only("inputs/specs/tracefile.vcd");
}

#[test]
fn diff_surfer_counter() {
    run_diff_test("inputs/surfer/counter.vcd", "inputs/surfer/counter.vcd.fst");
}

/// source: https://gitlab.com/surfer-project/surfer/-/issues/145
#[test]
fn diff_surfer_issue_145() {
    run_diff_test(
        "inputs/surfer/issue_145.vcd",
        "inputs/surfer/issue_145.vcd.fst",
    );
}

#[test]
#[ignore] // VCD parser stumbles over unexpected $dumpall
fn diff_surfer_picorv32() {
    run_diff_test(
        "inputs/surfer/picorv32.vcd",
        "inputs/surfer/picorv32.vcd.fst",
    );
}

#[test]
fn diff_surfer_spade() {
    run_diff_test("inputs/surfer/spade.vcd", "inputs/surfer/spade.vcd.fst");
}

#[test]
fn diff_surfer_verilator_empty_scope() {
    run_diff_test(
        "inputs/surfer/verilator_empty_scope.vcd",
        "inputs/surfer/verilator_empty_scope.vcd.fst",
    );
}

#[test]
fn diff_surfer_xx_1() {
    run_diff_test("inputs/surfer/xx_1.vcd", "inputs/surfer/xx_1.vcd.fst");
}

#[test]
fn diff_surfer_xx_2() {
    run_diff_test("inputs/surfer/xx_2.vcd", "inputs/surfer/xx_2.vcd.fst");
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

// See: https://gitlab.com/surfer-project/surfer/-/issues/201
// This file contains two separate declarations of the `top` scope which need to be merged
// automatically by `wellen`.
#[test]
fn diff_verilator_surfer_issue_201() {
    run_diff_test(
        "inputs/verilator/surfer_issue_201.vcd",
        "inputs/verilator/surfer_issue_201.fst",
    );
}

#[test]
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
fn diff_wikipedia_example() {
    run_diff_test_vcd_only("inputs/wikipedia/example.vcd");
}

#[test]
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
fn diff_xilinx_isim_test2x2_regex22_string1() {
    run_diff_test(
        "inputs/xilinx_isim/test2x2_regex22_string1.vcd",
        "inputs/xilinx_isim/test2x2_regex22_string1.vcd.fst",
    );
}

#[test]
fn diff_scope_with_comment() {
    run_diff_test(
        "inputs/scope_with_comment.vcd",
        "inputs/scope_with_comment.vcd.fst",
    );
}

/// The test file was provided by Robin Heinemann here:
/// https://gitlab.com/surfer-project/surfer/-/issues/315
/// The problem that this reveals is that Yosys SMTBMC assigns
/// VCD identifiers by incrementing a counter and turning the number
/// directly into a string. Like n1, n2, n3 ....
/// This means that we really need to create a hash table to get sensible internal IDs.
#[test]
fn diff_yosys_smtbmc_surfer_issue_315() {
    run_diff_test_vcd_only("inputs/yosys_smtbmc/surfer_issue_315.vcd");
}
