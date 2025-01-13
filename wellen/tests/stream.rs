// Copyright 2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use crate::utils::*;
use rustc_hash::FxHashMap;
use std::fs::File;
use std::io::BufReader;
use wellen::stream::*;
use wellen::{LoadOptions, TimeTableIdx};

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

    let mut delta_counter = FxHashMap::default();

    let filter = Filter::all();
    streamed
        .stream(&filter, |time, sig, a_value| {
            // find corresponding signal value in memory
            let idx = *time_to_idx
                .get(&time)
                .expect("failed to find time in time table") as usize;
            let b_value = get_value(&batch, sig, idx, &mut delta_counter);
            diff_signal_value(time, sig, a_value, b_value, None, batch.hierarchy());
        })
        .unwrap();
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
#[ignore] // TODO: add streaming support for FST
fn diff_stream_nvc_xwb_fofb_shaper_filt_tb_arrays() {
    diff_stream("inputs/nvc/xwb_fofb_shaper_filt_tb_arrays.fst");
}

#[test]
#[ignore] // TODO: why does this one fail?
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
