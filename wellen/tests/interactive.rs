// Copyright 2026 Cornell University
// released under BSD 3-Clause License

use std::collections::HashMap;
use wellen::{
    Encoder, Hierarchy, HierarchyBuilder, ScopeType, Signal, SignalEncoding, SignalRef,
    SignalSource, Timescale, TimescaleUnit, VarDirection, VarIndex, VarType,
};

#[test]
fn test_snapshot_allows_continued_encoding() {
    let (hierarchy, clk, data) = build_hierarchy();
    assert!(hierarchy.version().contains("wellen"));
    assert_eq!(hierarchy.timescale().unwrap().unit, TimescaleUnit::Seconds);
    assert_eq!(hierarchy.timescale().unwrap().factor, 100);
    let mut encoder = Encoder::new(&hierarchy);

    // Initial data, used for the first snapshot.
    encoder.time_change(0);
    encoder.vcd_value_change(clk, b"0");
    encoder.vcd_value_change(data, b"b0000");
    encoder.time_change(5);
    encoder.vcd_value_change(clk, b"1");
    encoder.vcd_value_change(data, b"b0011");

    let (mut source1, time_table1) = encoder.snapshot();
    assert_eq!(time_table1, vec![0, 5]);
    assert_snapshot(
        &mut source1,
        &hierarchy,
        &time_table1,
        clk,
        data,
        vec![(0, "0".to_string()), (5, "1".to_string())],
        vec![(0, "0000".to_string()), (5, "0011".to_string())],
    );

    // More data after the first snapshot, then snapshot again.
    encoder.time_change(9);
    encoder.vcd_value_change(data, b"b1010");
    encoder.time_change(12);
    encoder.vcd_value_change(clk, b"0");
    encoder.vcd_value_change(data, b"b1111");

    let (mut source2, time_table2) = encoder.snapshot();
    assert_eq!(time_table2, vec![0, 5, 9, 12]);
    assert_snapshot(
        &mut source2,
        &hierarchy,
        &time_table2,
        clk,
        data,
        vec![
            (0, "0".to_string()),
            (5, "1".to_string()),
            (12, "0".to_string()),
        ],
        vec![
            (0, "0000".to_string()),
            (5, "0011".to_string()),
            (9, "1010".to_string()),
            (12, "1111".to_string()),
        ],
    );
}

fn build_hierarchy() -> (Hierarchy, SignalRef, SignalRef) {
    let timescale = Timescale::new(100, TimescaleUnit::Seconds);
    let mut builder = HierarchyBuilder::new(timescale, None, None);

    builder.push_scope("top", ScopeType::Module, Some("Top"));
    builder.push_scope("dut", ScopeType::Module, Some("MyCoolDesign"));

    let clk = builder.new_signal(SignalEncoding::BitVector(1));
    builder.add_var(
        "clk",
        VarType::Wire,
        VarDirection::Unknown,
        Some("clock_t"),
        None,
        None,
        clk,
    );

    let data = builder.new_signal(SignalEncoding::BitVector(4));
    let data_index = VarIndex::new(-1, -4);
    builder.add_var(
        "data",
        VarType::Reg,
        VarDirection::Unknown,
        Some("data_t"),
        Some(data_index),
        None,
        data,
    );

    let enum_data = builder.new_signal(SignalEncoding::BitVector(2));
    let four_state_t = builder.declare_enum(
        "four_state_t",
        [("00", "0"), ("01", "1"), ("10", "x"), ("11", "z")].into_iter(),
    );
    builder.add_var(
        "enum_data",
        VarType::Logic,
        VarDirection::Input,
        Some("four_state_t"),
        None,
        Some(four_state_t),
        enum_data,
    );

    builder.pop_scope();
    builder.pop_scope();
    let hierarchy = builder.finish();

    assert!(hierarchy.lookup_scope(&["top"]).is_some());
    assert!(hierarchy.lookup_scope(&["top", "dut"]).is_some());
    let dut = hierarchy.lookup_scope(&["top", "dut"]).unwrap();
    assert_eq!(
        hierarchy[dut].component(&hierarchy).unwrap(),
        "MyCoolDesign"
    );
    assert!(hierarchy.lookup_var(&["top", "dut"], "clk").is_some());
    assert!(hierarchy.lookup_var(&["top", "dut"], "data").is_some());

    (hierarchy, clk, data)
}

fn assert_snapshot(
    source: &mut SignalSource,
    hierarchy: &Hierarchy,
    time_table: &[u64],
    clk: SignalRef,
    data: SignalRef,
    expected_clk: Vec<(u64, String)>,
    expected_data: Vec<(u64, String)>,
) {
    let loaded = source.load_signals(&[clk, data], hierarchy, false);
    let by_ref: HashMap<SignalRef, Signal> =
        loaded.into_iter().map(|s| (s.signal_ref(), s)).collect();

    let clk_signal = by_ref.get(&clk).unwrap();
    let data_signal = by_ref.get(&data).unwrap();

    assert_eq!(collect_bit_changes(clk_signal, time_table), expected_clk);
    assert_eq!(collect_bit_changes(data_signal, time_table), expected_data);
}

fn collect_bit_changes(signal: &Signal, time_table: &[u64]) -> Vec<(u64, String)> {
    signal
        .iter_changes()
        .map(|(time_idx, value)| {
            (
                time_table[time_idx as usize],
                value
                    .to_bit_string()
                    .expect("expected a bit-vector value for this test"),
            )
        })
        .collect()
}
