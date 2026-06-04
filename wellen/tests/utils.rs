// Copyright 2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use rustc_hash::FxHashMap;
use wellen::simple::Waveform;
use wellen::{Hierarchy, SignalRef, SignalValueRef, States, Time, TimeTableIdx, Var};

#[allow(dead_code)]
pub fn load_all_signals(our: &mut Waveform) {
    let all_signals = get_all_signals(our.hierarchy());
    our.load_signals(&all_signals);
}

#[allow(dead_code)]
pub fn get_all_signals(h: &Hierarchy) -> Vec<SignalRef> {
    h.signals().collect()
}

#[allow(dead_code)]
pub fn diff_signals(a: &mut Waveform, b: &mut Waveform, time_factor: u64) {
    // with the same time tables, comparisons become much easier!
    assert_eq!(time_factor, 1);
    assert_eq!(a.time_table(), b.time_table());

    let time_table = Vec::from_iter(a.time_table().iter().cloned());

    let all_signals_ghw = get_all_signals(a.hierarchy());
    let all_signals_fst = get_all_signals(b.hierarchy());
    assert_eq!(all_signals_ghw, all_signals_fst);

    let a_signals: Vec<_> = a.hierarchy().signals().collect();

    for signal in a_signals {
        let a_signal = a.get_signal(signal).unwrap();
        let b_signal = b.get_signal(signal).unwrap();

        for (idx, time) in time_table.iter().enumerate() {
            let offset_g = a_signal.get_offset(idx as TimeTableIdx);
            let offset_f = b_signal.get_offset(idx as TimeTableIdx);
            match (offset_g.clone(), offset_f.clone()) {
                (Some(og), Some(of)) => {
                    assert_eq!(og.elements, of.elements, "{signal:?} @ {time}");
                    let a_value = a_signal.get_value_at(&og, 0);
                    let b_value = b_signal.get_value_at(&of, 0);
                    diff_signal_value(*time, signal, a_value, b_value, None, a.hierarchy());
                }
                _ => assert_eq!(offset_g, offset_f),
            }
        }
    }
}

#[allow(dead_code)]
pub fn get_value<'a>(
    our: &'a Waveform,
    signal_ref: SignalRef,
    time_table_idx: usize,
    delta_counter: &mut FxHashMap<SignalRef, u16>,
) -> SignalValueRef<'a> {
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

#[allow(dead_code)]
pub fn get_final_value(
    our: &Waveform,
    signal_ref: SignalRef,
    time_table_idx: usize,
) -> SignalValueRef<'_> {
    get_maybe_final_value(our, signal_ref, time_table_idx).expect("failed to get value for signal")
}

#[allow(dead_code)]
pub fn get_maybe_final_value(
    our: &Waveform,
    signal_ref: SignalRef,
    time_table_idx: usize,
) -> Option<SignalValueRef<'_>> {
    let our_signal = our.get_signal(signal_ref).unwrap();
    let our_offset = our_signal.get_offset(time_table_idx as u32);
    our_offset.map(|of| our_signal.get_value_at(&of, of.elements - 1))
}

#[allow(dead_code)]
pub fn get_maybe_value_change_at(
    our: &Waveform,
    signal_ref: SignalRef,
    time_table_idx: usize,
) -> Option<SignalValueRef<'_>> {
    let our_signal = our.get_signal(signal_ref).unwrap();
    let our_offset = our_signal.get_offset(time_table_idx as u32)?;
    if our_offset.time_match {
        Some(our_signal.get_value_at(&our_offset, our_offset.elements - 1))
    } else {
        None
    }
}

pub fn diff_signal_value(
    time: Time,
    signal: SignalRef,
    a_value: SignalValueRef,
    b_value: SignalValueRef,
    a_signal_var: Option<&Var>,
    h: &Hierarchy,
) {
    ensure_minimal_format(a_value);
    ensure_minimal_format(b_value);
    match (a_value, b_value) {
        (SignalValueRef::String(gs), SignalValueRef::String(fs)) => {
            assert_eq!(gs, fs, "{signal:?} @ {time}");
        }
        (g_value, SignalValueRef::String(fs)) => {
            if let Some(signal_var) = a_signal_var {
                if let Some((_, mapping)) = signal_var.enum_type(h) {
                    // find enum value
                    let g_value_str = g_value.to_bit_string().unwrap();
                    let mut found = false;
                    for (bits, name) in mapping.iter() {
                        if **bits == g_value_str {
                            assert_eq!(*name, fs);
                            found = true;
                        }
                    }
                    assert!(
                        found,
                        "Could not find mapping for {g_value_str}\n{mapping:?}"
                    );
                } else {
                    println!(
                        "Ignoring: {signal:?} @ {time} = {:?} vs {}",
                        g_value.to_bit_string(),
                        fs
                    );
                }
            }
        }
        (SignalValueRef::Real(a_real), SignalValueRef::Real(b_real)) => {
            assert_eq!(a_real, b_real, "{signal:?} @ {time}");
        }
        (g_value, f_value) => {
            if g_value.is_event() {
                assert!(f_value.is_event())
            } else {
                assert_eq!(
                    g_value.to_bit_string(),
                    f_value.to_bit_string(),
                    "{signal:?} @ {time}"
                );
            }
        }
    }
}

#[allow(dead_code)]
/// Checks to make sure that 4 and 9 state signals are only used when they are required.
fn ensure_minimal_format(signal_value: SignalValueRef) {
    if let SignalValueRef::BitVec(bv) = signal_value {
        match bv.states() {
            States::Two => {}
            States::Four => {
                let value_str = bv.bit_string();
                assert!(
                    value_str.contains('x') || value_str.contains('z'),
                    "{value_str} does not need to be represented as a 4-state signal"
                )
            }
            States::Nine => {
                let value_str = bv.bit_string();
                assert!(
                    value_str.contains('h')
                        || value_str.contains('u')
                        || value_str.contains('w')
                        || value_str.contains('l')
                        || value_str.contains('-'),
                    "{value_str} does not need to be represented as a 9-state signal"
                )
            }
        }
    }
}
