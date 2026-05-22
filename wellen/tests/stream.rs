// Copyright 2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use crate::utils::*;
use rand::rngs::Xoshiro256PlusPlus;
use rand::{Rng, RngExt, SeedableRng};
use rustc_hash::{FxHashMap, FxHashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Seek};
use wellen::simple::Waveform;
use wellen::stream::*;
use wellen::{Hierarchy, LoadOptions, SignalRef, SignalValue, SignalValueRef, Time, TimeTableIdx};

mod utils;

/// diff tests the streaming vs. the viewer centric interface
fn diff_stream(filename: &str) {
    let mut streamed = load_streaming(filename);
    let mut batch = wellen::simple::read(filename).expect("failed to open file in batch mode");
    load_all_signals(&mut batch);
    let time_to_idx = FxHashMap::from_iter(
        batch
            .time_table()
            .iter()
            .enumerate()
            .map(|(ii, &t)| (t, ii as TimeTableIdx)),
    );

    // simply stream each individual signal change
    diff_stream_changes(&batch, &mut streamed, &time_to_idx, Filter::all());
    // make sure we can stream twice
    diff_stream_changes(&batch, &mut streamed, &time_to_idx, Filter::all());
    // compare for three random signal subselections
    let mut rnd = Xoshiro256PlusPlus::from_seed([0; 32]);
    for _ in 0..3 {
        let signals = random_signals(&mut rnd, batch.hierarchy());
        let filter = Filter::include_signals(&signals);
        diff_stream_changes(&batch, &mut streamed, &time_to_idx, filter);
    }
    // batch changes in a single time step
    diff_stream_time_change(&batch, &mut streamed, &time_to_idx, Filter::all());
    // batch changes with three random signal subselections
    for _ in 0..3 {
        let signals = random_signals(&mut rnd, batch.hierarchy());
        let filter = Filter::include_signals(&signals);
        diff_stream_time_change(&batch, &mut streamed, &time_to_idx, filter);
    }
}

fn diff_stream_for_vars(filename: &str, vars: &[&str]) {
    let mut streamed = load_streaming(filename);
    let mut batch = wellen::simple::read(filename).expect("failed to open file in batch mode");
    let time_to_idx = FxHashMap::from_iter(
        batch
            .time_table()
            .iter()
            .enumerate()
            .map(|(ii, &t)| (t, ii as TimeTableIdx)),
    );

    let signals: Vec<_> = vars
        .iter()
        .map(|name| {
            let mut parts: Vec<_> = name.split('.').collect();
            let basename = parts.pop().unwrap();
            let var_ref = batch
                .hierarchy()
                .lookup_var(&parts, basename)
                .expect("unable to find var!");
            batch.hierarchy()[var_ref].signal_ref()
        })
        .collect();
    batch.load_signals(&signals);

    diff_stream_changes(
        &batch,
        &mut streamed,
        &time_to_idx,
        Filter::include_signals(&signals),
    );
}

fn random_signals(rnd: &mut impl Rng, h: &Hierarchy) -> Vec<SignalRef> {
    let all_signals: Vec<_> = h.signals().collect();
    let mut signals = FxHashSet::default();
    let include_num = rnd.random_range(0..all_signals.len());
    while signals.len() < include_num {
        let index = rnd.random_range(0..all_signals.len());
        signals.insert(all_signals[index]);
    }
    signals.into_iter().collect()
}

fn diff_stream_changes<R: BufRead + Seek>(
    batch: &Waveform,
    streamed: &mut StreamingWaveform<R>,
    time_to_idx: &FxHashMap<Time, TimeTableIdx>,
    filter: Filter,
) {
    let mut delta_counter = FxHashMap::default();
    // keeps track of the times when we see a signal changing in the stream
    let mut observed_changes = FxHashMap::default();
    // remember previous value in order to filter out no-op changes
    let mut prev_value: FxHashMap<SignalRef, SignalValue> = FxHashMap::default();

    streamed
        .stream_changes(filter, |time, sig, a_value| {
            // find corresponding signal value in memory
            let idx = *time_to_idx
                .get(&time)
                .expect("failed to find time in time table") as usize;
            if batch.get_signal(sig).is_none() {
                panic!("Received a change for a signal that we did not request: {time} {sig:?} {a_value:?}");
            }

            let b_value = get_value(batch, sig, idx, &mut delta_counter);
            // println!("{time}, {a_value} vs {b_value}");
            diff_signal_value(time, sig, a_value, b_value, None, batch.hierarchy());
            // record observed change if the value actually changed (or if we are dealing with an event)
            if !prev_value.contains_key(&sig)
                || a_value.is_event()
                || SignalValueRef::from(&prev_value[&sig]) != a_value
            {
                observed_changes
                    .entry(sig)
                    .or_insert_with(Vec::new)
                    .push(time);
            }
            let value: SignalValue = a_value.into();
            prev_value.insert(sig, value);
        })
        .unwrap();

    // make sure we actually saw all the changes!
    for signal in batch.hierarchy().signals() {
        if filter.includes_signal(signal) {
            let time_indices = batch.get_signal(signal).unwrap().time_indices();
            let expected: Vec<_> = time_indices
                .iter()
                .map(|ii| batch.time_table()[*ii as usize])
                .collect();
            let empty = vec![];
            let observed = observed_changes.get(&signal).unwrap_or(&empty);
            assert_eq!(
                &expected,
                observed,
                "{} sees different changes (batch vs. stream)",
                find_signal_name(batch.hierarchy(), signal)
            );
        }
    }
}

/// Checks that the filter has no duplicates.
fn check_filter(filter: Filter) {
    if let Some(signals) = filter.signals {
        let mut seen = FxHashSet::default();
        for signal in signals {
            assert!(
                !seen.contains(signal),
                "Duplicate signal in filter: {signal:?}"
            );
            seen.insert(*signal);
        }
    }
}

fn diff_stream_time_change<R: BufRead + Seek>(
    batch: &Waveform,
    streamed: &mut StreamingWaveform<R>,
    time_to_idx: &FxHashMap<Time, TimeTableIdx>,
    filter: Filter,
) {
    check_filter(filter);
    let mut prev_time = None;
    let mut observed_times = FxHashSet::default();
    let signals = filter
        .signals
        .map(|s| s.to_vec())
        .unwrap_or_else(|| batch.hierarchy().signals().collect());

    streamed
        .stream_time_steps(filter, |time, values| {
            if let Some(prev_time) = prev_time {
                assert!(time > prev_time, "time must be incrementing!");
            }
            println!("OBSERVED: {time}  ({prev_time:?})");
            prev_time = Some(time);

            let idx = *time_to_idx
                .get(&time)
                .expect("failed to find time in time table") as usize;
            observed_times.insert(time);

            // compare all signals at this time step
            for sig in &signals {
                // only check if there is a value at this time in the reference
                if let Some(b_value) = get_maybe_final_value(batch, *sig, idx) {
                    let maybe_a_value = values.get(sig);
                    assert!(maybe_a_value.is_some(), "Failed to get value of signal {sig:?} at time {time} from the dispatcher map.,The expected value is: {b_value:?}");
                    let a_value: SignalValueRef = maybe_a_value.unwrap().into();
                    diff_signal_value(time, *sig, a_value, b_value, None, batch.hierarchy());
                }
            }
        })
        .expect("failed to stream!");

    // make sure we did not skip a time step
    let mut prev_time = None;
    for (time_idx, time) in batch.time_table().iter().enumerate() {
        let time_idx = time_idx as TimeTableIdx;
        if let Some(prev) = prev_time {
            assert!(time > prev);
        }
        prev_time = Some(time);
        if observed_times.contains(time) {
            let mut change_exists = false;
            // ensure that a change occurred for at least one signal
            for &sig_ref in &signals {
                let sig = batch.get_signal(sig_ref).unwrap();
                let off = sig.get_offset(time_idx);
                change_exists |= off.map(|o| o.time_match).unwrap_or(false);
                if change_exists {
                    // a change occurred -> we are good!
                    break;
                }
            }
            assert!(
                change_exists,
                "The callback was called at time {}, but there was no change in the observed signals!",
                time
            );
        } else {
            // ensure that no change occurred!
            for &sig in &signals {
                let sig = batch.get_signal(sig).unwrap();
                let changed = sig
                    .get_offset(time_idx)
                    .map(|o| o.time_match)
                    .unwrap_or(false);
                assert!(
                    !changed,
                    "Signal {sig:?} changed at {time}, but the callback was not called!"
                );
            }
        }
    }
}

fn find_signal_name(h: &Hierarchy, s: SignalRef) -> String {
    for var in h.all_vars() {
        if var.signal_ref() == s {
            return var.full_name(h);
        }
    }
    "unknown".into()
}

fn load_streaming(filename: &str) -> StreamingWaveform<BufReader<File>> {
    let opts = LoadOptions::default();
    read_from_file(filename, &opts).expect("failed to open file for streaming")
}

#[test]
fn diff_stream_aldec_spi_write() {
    diff_stream("inputs/aldec/SPI_Write.vcd");
}

#[test]
fn diff_stream_amaranth_up_counter() {
    diff_stream("inputs/amaranth/up_counter.vcd");
}

#[test]
fn diff_stream_gameroy_trace() {
    diff_stream("inputs/gameroy/trace_prefix.vcd");
}

#[test]
fn diff_stream_ghdl_alu() {
    diff_stream("inputs/ghdl/alu.vcd");
}

#[test]
fn diff_stream_ghdl_idea() {
    diff_stream("inputs/ghdl/idea.vcd");
}

#[test]
fn diff_stream_ghdl_pcpu() {
    diff_stream("inputs/ghdl/pcpu.vcd");
}

#[test]
fn diff_stream_ghdl_oscar_ghdl() {
    diff_stream("inputs/ghdl/oscar/ghdl.fst");
}

#[test]
fn diff_stream_ghdl_oscar_vhdl3() {
    diff_stream("inputs/ghdl/oscar/vhdl3.fst");
}

#[test]
fn diff_stream_gtkwave_perm_current() {
    diff_stream("inputs/gtkwave-analyzer/perm_current.vcd");
}

#[test]
fn diff_stream_jtag_atxmega256a3u_bmda() {
    diff_stream("inputs/jtag/atxmega256a3u-bmda-jtag.vcd");
}

#[test]
fn diff_stream_icarus_cpu() {
    diff_stream("inputs/icarus/CPU.vcd");
}

#[test]
fn diff_stream_icarus_dc_crossbar() {
    diff_stream("inputs/icarus/DCCrossbar.vcd");
}

#[test]
fn diff_stream_icarus_rv32_soc_tb() {
    diff_stream("inputs/icarus/rv32_soc_TB.vcd");
}

#[test]
fn diff_stream_icarus_test1() {
    diff_stream("inputs/icarus/test1.vcd");
}

#[test]
fn diff_stream_model_sim_clkdiv2n_tb() {
    diff_stream("inputs/model-sim/clkdiv2n_tb.vcd");
}

#[test]
fn diff_stream_model_sim_cpu_design() {
    diff_stream("inputs/model-sim/CPU_Design.msim.vcd");
}

#[test]
fn diff_stream_my_hdl_simple_memory() {
    diff_stream("inputs/my-hdl/Simple_Memory.vcd");
}

#[test]
fn diff_stream_my_hdl_top() {
    diff_stream("inputs/my-hdl/top.vcd");
}

#[test]
fn diff_stream_ncsim_ffdiv_32bit_tb() {
    diff_stream("inputs/ncsim/ffdiv_32bit_tb.vcd");
}

#[test]
fn diff_stream_nvc_xwb_fofb_shaper_filt_tb_arrays() {
    diff_stream("inputs/nvc/xwb_fofb_shaper_filt_tb_arrays.fst");
}

#[test]
fn diff_stream_nvc_xwb_fofb_shaper_filt_tb() {
    diff_stream("inputs/nvc/xwb_fofb_shaper_filt_tb.fst");
}

#[test]
fn diff_stream_nvc_overlay_tb_issue_21() {
    diff_stream("inputs/nvc/overlay_tb_issue_21.fst");
}

#[test]
fn diff_stream_nvc_vhdl_test_bool_issue_16() {
    diff_stream("inputs/nvc/vhdl_test_bool_issue_16.fst");
}

#[test]
fn diff_stream_pymtl3_cgra() {
    diff_stream("inputs/pymtl3/CGRA.vcd");
}

#[test]
fn diff_stream_quartus_mips_hardware() {
    diff_stream("inputs/quartus/mipsHardware.vcd");
}

#[test]
fn diff_stream_questa_sim_dump() {
    diff_stream("inputs/questa-sim/dump.vcd");
}

#[test]
fn diff_stream_questa_sim_test() {
    diff_stream("inputs/questa-sim/test.vcd");
}

#[test]
fn diff_stream_questa_wellen_issue_57() {
    diff_stream("inputs/questa-sim/wellen-issue-57-uart.vcd");
}

#[test]
fn diff_stream_riviera_pro_dump() {
    diff_stream("inputs/riviera-pro/dump.vcd");
}

#[test]
fn diff_stream_sigrok_libsigrok() {
    diff_stream("inputs/sigrok/libsigrok.vcd");
}

#[test]
fn diff_stream_specs_tracefile() {
    diff_stream("inputs/specs/tracefile.vcd");
}

#[test]
fn diff_stream_surfer_counter() {
    diff_stream("inputs/surfer/counter.vcd");
}

#[test]
fn diff_stream_surfer_issue_145() {
    diff_stream("inputs/surfer/issue_145.vcd");
}

#[test]
fn diff_stream_surfer_picorv32() {
    diff_stream("inputs/surfer/picorv32.vcd");
}

#[test]
fn diff_stream_surfer_spade() {
    diff_stream("inputs/surfer/spade.vcd");
}

#[test]
fn diff_stream_surfer_verilator_empty_scope() {
    diff_stream("inputs/surfer/verilator_empty_scope.vcd");
}

#[test]
fn diff_stream_surfer_xx_1() {
    diff_stream("inputs/surfer/xx_1.vcd");
}

#[test]
fn diff_stream_surfer_xx_2() {
    diff_stream("inputs/surfer/xx_2.vcd");
}

#[test]
fn diff_stream_systemc_waveform() {
    diff_stream("inputs/systemc/waveform.vcd");
}

#[test]
fn diff_stream_treadle_gcd() {
    diff_stream("inputs/treadle/GCD.vcd");
}

#[test]
fn diff_stream_vcs_apb_uvm_new() {
    diff_stream("inputs/vcs/Apb_slave_uvm_new.vcd");
}

#[test]
fn diff_stream_vcs_datapath_log() {
    diff_stream("inputs/vcs/datapath_log.vcd");
}

#[test]
fn diff_stream_vcs_processor() {
    diff_stream("inputs/vcs/processor.vcd");
}

#[test]
fn diff_stream_verilator_surfer_issue_201() {
    diff_stream("inputs/verilator/surfer_issue_201.vcd");
}

#[test]
fn diff_stream_verilator_swerv1() {
    diff_stream("inputs/verilator/swerv1.vcd");
}

#[test]
fn diff_stream_verilator_vlt_dump() {
    diff_stream("inputs/verilator/vlt_dump.vcd");
}

#[test]
fn diff_stream_verilator_basic_test() {
    diff_stream("inputs/verilator/basic_test.fst");
}

#[test]
fn diff_stream_verilator_many_sv_datatypes() {
    diff_stream("inputs/verilator/many_sv_datatypes.fst");
}

#[test]
fn diff_stream_verilator_surfer_issue_201_fst() {
    diff_stream("inputs/verilator/surfer_issue_201.fst");
}

#[test]
fn diff_stream_verilator_verilator_incomplete() {
    diff_stream("inputs/verilator/verilator-incomplete.fst");
}

#[test]
fn diff_stream_verilator_complex_structs() {
    diff_stream("inputs/verilator/verilator-pull-7255-t_trace_complex_structs_cc_fst.fst");
}

#[test]
fn diff_stream_vivado_iladata() {
    diff_stream("inputs/vivado/iladata.vcd");
}

#[test]
fn diff_stream_wikipedia_example() {
    diff_stream("inputs/wikipedia/example.vcd");
}

#[test]
fn diff_stream_xilinx_isim_test() {
    diff_stream("inputs/xilinx_isim/test.vcd");
}

#[test]
fn diff_stream_xilinx_isim_test1() {
    diff_stream("inputs/xilinx_isim/test1.vcd");
}

#[test]
fn diff_stream_xilinx_isim_test2x2_regex22_string1() {
    diff_stream("inputs/xilinx_isim/test2x2_regex22_string1.vcd");
}

#[test]
fn diff_stream_scope_with_comment() {
    diff_stream("inputs/scope_with_comment.vcd");
}

#[test]
fn diff_stream_yosys_smtbmc_surfer_issue_315() {
    diff_stream("inputs/yosys_smtbmc/surfer_issue_315.vcd");
}

#[test]
fn diff_stream_questa_sim_derived_signal() {
    let filename = "inputs/questa-sim/wellen-issue-57-uart.vcd";
    let vars = ["tb_uart.dut.prescale"];
    diff_stream_for_vars(filename, vars.as_slice());
}
