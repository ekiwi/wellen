use std::collections::HashMap;

use crate::{
    signal_utils::all_changes, BitVectorBuilder, Signal, SignalRef, SignalValue, States,
    TimeTableIdx,
};

//TODO: support variable length signals -- currently only supports fixed length signals
pub trait LazySignal {
    fn value_at_idx(&self, time_idx: TimeTableIdx) -> Option<SignalValue>;
    fn time_indices(&self) -> &[TimeTableIdx];
    fn width(&self) -> u32;
    fn states(&self) -> States;
    fn to_signal(&self) -> Signal {
        let mut vec_builder = BitVectorBuilder::new(self.states(), self.width());

        for change in self.time_indices().iter().cloned() {
            let value = self.value_at_idx(change);
            if let Some(actual_val) = value {
                vec_builder.add_change(change, actual_val)
            }
        }
        vec_builder.finish(SignalRef::from_index(0xbeedad69).unwrap())
    }
}

pub struct SimpleLazy {
    signals: HashMap<String, Signal>,
    generator: Box<dyn Fn(&'_ HashMap<String, Signal>, TimeTableIdx) -> Option<SignalValue<'_>>>,
    all_times: Vec<TimeTableIdx>,
    width: u32,
}

impl SimpleLazy {
    pub fn new(
        signals: HashMap<String, Signal>,
        generator: Box<dyn Fn(&HashMap<String, Signal>, TimeTableIdx) -> Option<SignalValue>>,
        width: u32,
    ) -> SimpleLazy {
        let all_times = all_changes(signals.values());
        SimpleLazy {
            all_times,
            signals,
            generator,
            width,
        }
    }
}

impl LazySignal for SimpleLazy {
    fn states(&self) -> States {
        self.signals
            .values()
            .filter_map(|val| val.max_states())
            .max()
            .unwrap_or(States::Two)
    }
    fn value_at_idx(&self, time_idx: TimeTableIdx) -> Option<SignalValue> {
        let gen = &self.generator;
        gen(&self.signals, time_idx)
    }
    fn width(&self) -> u32 {
        self.width
    }
    fn time_indices(&self) -> &[TimeTableIdx] {
        &self.all_times
    }
}

/// A signal that only emits values from a "source" signal when an "enable" signal changes
///
/// The enable signal doesnt _need_ to be a single bit -- a "clocking" event happens on any change
pub struct VirtualEdgeTriggered<'a> {
    source: &'a Signal,
    virtual_enable: &'a Signal,
    all_times: Vec<TimeTableIdx>,
}

impl<'a> VirtualEdgeTriggered<'a> {
    pub fn new(source: &'a Signal, virtual_enable: &'a Signal) -> Option<Self> {
        if source.width().is_some() && source.max_states().is_some() {
            let all_times = all_changes(vec![source, virtual_enable]);
            Some(Self {
                source,
                virtual_enable,
                all_times,
            })
        } else {
            None
        }
    }
}
impl<'a> LazySignal for VirtualEdgeTriggered<'a> {
    fn value_at_idx(&self, time_idx: TimeTableIdx) -> Option<SignalValue> {
        if let Some(offset) = self.virtual_enable.get_offset(time_idx) {
            if offset.time_match {
                self.source
                    .get_offset(time_idx)
                    .map(|offset| self.source.get_value_at(&offset, 0))
            } else {
                None
            }
        } else {
            None
        }
    }
    fn time_indices(&self) -> &[TimeTableIdx] {
        self.all_times.as_slice()
    }
    fn width(&self) -> u32 {
        self.source.width().unwrap()
    }
    fn states(&self) -> States {
        self.source.max_states().unwrap()
    }
}
