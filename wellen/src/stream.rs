// Copyright 2025-26 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>
//! # Stream Interface
//! This interface is useful when you want to batch process waveform data instead
//! of displaying it in a waveform viewer.

use crate::signal::{DerivedBitVecSignal, SignalMap, States};
use crate::vcd::{VcdBitVecChange, decode_vcd_bit_vec_change};
use crate::wavemem::write_n_state_from_ascii;
use crate::{
    FileFormat, Hierarchy, LoadOptions, Real, Result, SignalEncoding, SignalRef, SignalValue,
    SignalValueRef, Time, WellenError, viewers,
};
use fst_reader::FstSignalValue;
use rustc_hash::FxHashSet;
use smallvec::{SmallVec, smallvec};
use std::fmt::{Debug, Formatter};
use std::io::{BufRead, Seek};

/// Read a waveform file. Reads only the header.
pub fn read_from_file<P: AsRef<std::path::Path>>(
    filename: P,
    options: &LoadOptions,
) -> Result<StreamingWaveform<std::io::BufReader<std::fs::File>>> {
    let file_format = viewers::open_and_detect_file_format(filename.as_ref());
    match file_format {
        FileFormat::Unknown => Err(WellenError::UnknownFileFormat),
        FileFormat::Vcd => {
            let (hierarchy, body, _body_len) =
                crate::vcd::read_header_from_file(filename, options)?;
            Ok(StreamingWaveform {
                hierarchy,
                body: viewers::ReadBodyData::Vcd(Box::new(body)),
            })
        }
        FileFormat::Ghw => {
            todo!("streaming for ghw")
        }
        FileFormat::Fst => {
            let (hierarchy, body) = crate::fst::read_header_from_file(filename, options)?;
            Ok(StreamingWaveform {
                hierarchy,
                body: viewers::ReadBodyData::Fst(Box::new(body)),
            })
        }
    }
}

/// Read from something that is not a file. Reads only the header.
pub fn read<R: BufRead + Seek + Send + Sync + 'static>(
    _input: R,
    _options: &LoadOptions,
) -> Result<StreamingWaveform<R>> {
    todo!("support streaming read from things that are not files")
}

/// Represents a waveform that was loaded for streaming.
pub struct StreamingWaveform<R: BufRead + Seek> {
    hierarchy: Hierarchy,
    body: viewers::ReadBodyData<R>,
}

impl<R: BufRead + Seek> Debug for StreamingWaveform<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "stream::Waveform(...)")
    }
}

/// Defines which time interval and signals to include when streaming value changes.
#[derive(Debug, Copy, Clone)]
pub struct Filter<'a> {
    /// First time change.
    pub start: Time,
    /// Last time change. `None` means until the end of the file.
    pub end: Option<Time>,
    /// `None` means all signals.
    pub signals: Option<&'a [SignalRef]>,
}

impl<'a> Filter<'a> {
    /// Include all value changes.
    pub fn all() -> Self {
        Filter {
            start: 0,
            end: None,
            signals: None,
        }
    }

    pub fn new(start: u64, end: u64, signals: &'a [SignalRef]) -> Self {
        Filter {
            start,
            end: Some(end),
            signals: Some(signals),
        }
    }

    pub fn include_signals(signals: &'a [SignalRef]) -> Self {
        Filter {
            start: 0,
            end: None,
            signals: Some(signals),
        }
    }

    pub fn includes_signal(&self, signal: SignalRef) -> bool {
        // if we do not have a singal slice, then that means that all signals are included
        self.signals.map(|s| s.contains(&signal)).unwrap_or(true)
    }
}

impl<R: BufRead + Seek> StreamingWaveform<R> {
    pub fn hierarchy(&self) -> &Hierarchy {
        &self.hierarchy
    }

    /// Calls the callback for each change of a signal.
    pub fn stream_changes(
        &mut self,
        filter: Filter,
        callback: impl FnMut(Time, SignalRef, SignalValueRef<'_>),
    ) -> Result<()> {
        let (mut dispatcher, maybe_augmented_filter) =
            StreamDispatcherOnChange::new(self.hierarchy(), &filter, callback);

        let sub_filter = maybe_augmented_filter
            .as_ref()
            .map(|refs| Filter::include_signals(refs))
            .unwrap_or(filter);

        self.do_stream(
            &sub_filter,
            StreamEncoder::new(self.hierarchy(), &sub_filter, |time, signal, value| {
                dispatcher.on_change(time, signal, value)
            }),
        )?;
        dispatcher.finish();
        Ok(())
    }

    pub fn stream_time_steps(
        &mut self,
        filter: Filter,
        callback: impl FnMut(Time, &SignalMap<SignalValue>),
    ) -> Result<()> {
        let (mut dispatcher, maybe_augmented_filter) =
            StreamDispatcherOnTimeStep::new(self.hierarchy(), &filter, callback);

        let sub_filter = maybe_augmented_filter
            .as_ref()
            .map(|refs| Filter::include_signals(refs))
            .unwrap_or(filter);

        self.do_stream(
            &sub_filter,
            StreamEncoder::new(self.hierarchy(), &sub_filter, |time, signal, value| {
                dispatcher.on_change(time, signal, value)
            }),
        )?;
        dispatcher.finish();
        Ok(())
    }

    fn do_stream<C: FnMut(Time, SignalRef, SignalValueRef<'_>)>(
        &mut self,
        filter: &Filter,
        enc: StreamEncoder<C>,
    ) -> Result<()> {
        // ensure that none of the signals are slices
        match &mut self.body {
            viewers::ReadBodyData::Vcd(data) => crate::vcd::stream_body(data, enc)?,
            viewers::ReadBodyData::Fst(data) => crate::fst::stream_body(data, enc, filter)?,
            viewers::ReadBodyData::Ghw(_) => panic!("streaming GHW files is not supported"),
        }
        Ok(())
    }
}

/// Takes on the role of the [Encoder] when streaming instead of encoding to
/// a wavemem.
pub(crate) struct StreamEncoder<C>
where
    C: FnMut(Time, SignalRef, SignalValueRef<'_>),
{
    callback: C,
    time: Option<Time>,
    skipping_time_step: bool,
    /// contains encoding for all _included_ signals (depending on the [Filter] provided)
    encoding: SignalMap<SignalEncoding>,
    buf: SignalValueBuffer,
}

/// Designed to work around the fact that we cannot have a reference to an element of the same struct.
/// Also simplifies ownership.
#[derive(Debug, Default)]
struct SignalValueBuffer {
    data: Vec<u8>,
    string: String,
    kind: SignalKind,
}

#[derive(Debug, Default)]
enum SignalKind {
    #[default]
    Event,
    String,
    BitVec(States, u32),
    Real(Real),
}

impl SignalValueBuffer {
    fn update_fst(&mut self, encoding: SignalEncoding, value: &FstSignalValue) {
        self.data.clear();
        self.string.clear();
        self.kind = match value {
            FstSignalValue::String(value) => match encoding {
                SignalEncoding::Event => {
                    debug_assert!(value.is_empty(), "events do not carry data");
                    SignalKind::Event
                }
                SignalEncoding::String => {
                    debug_assert!(self.string.is_empty());
                    self.string
                        .push_str(String::from_utf8_lossy(value).as_ref());
                    SignalKind::String
                }

                SignalEncoding::BitVector(len) => {
                    let width = len.get();

                    debug_assert_eq!(
                        value.len(),
                        width as usize,
                        "{}",
                        String::from_utf8_lossy(value)
                    );

                    let states = States::from_ascii(value).unwrap_or_else(|| {
                        panic!(
                            "Unexpected signal value: {}",
                            String::from_utf8_lossy(value)
                        )
                    });

                    // convert from ASCII characters to packed encoding
                    debug_assert!(self.data.is_empty());
                    write_n_state_from_ascii(states, value, &mut self.data, None);
                    SignalKind::BitVec(states, width)
                }
                SignalEncoding::Real => panic!(
                    "Expecting reals, but got: {}",
                    String::from_utf8_lossy(value)
                ),
                SignalEncoding::Unknown => unreachable!("Unknown signal encoding!"),
            },
            FstSignalValue::Real(value) => {
                debug_assert_eq!(encoding, SignalEncoding::Real);
                SignalKind::Real(*value)
            }
        };
    }

    fn update_vcd(&mut self, encoding: SignalEncoding, value: &[u8]) {
        self.data.clear();
        self.string.clear();
        self.kind = match encoding {
            SignalEncoding::Event => {
                debug_assert!(
                    value.len() <= 1,
                    "event changes carry no value, or a 1-bit value"
                );
                SignalKind::Event
            }
            SignalEncoding::BitVector(width) => {
                let (data, states) = decode_vcd_bit_vec_change(width, value);
                debug_assert!(self.data.is_empty());

                // put data into buffer
                match data {
                    VcdBitVecChange::SingleBit(bit_value) => {
                        self.data.push(bit_value.into());
                    }
                    VcdBitVecChange::MultiBit(data_to_write) => {
                        write_n_state_from_ascii(states, &data_to_write, &mut self.data, None);
                    }
                }
                SignalKind::BitVec(states, width.get())
            }
            SignalEncoding::String => {
                assert!(
                    matches!(value[0], b's' | b'S'),
                    "expected a string, not {}",
                    String::from_utf8_lossy(value)
                );
                let characters = &value[1..];
                debug_assert!(self.string.is_empty());
                self.string
                    .push_str(std::str::from_utf8(characters).unwrap());
                SignalKind::String
            }
            SignalEncoding::Real => {
                assert!(
                    matches!(value[0], b'r' | b'R'),
                    "expected a real, not {}",
                    String::from_utf8_lossy(value)
                );
                // parse float
                let float_value: Real = std::str::from_utf8(&value[1..])
                    .unwrap()
                    .parse::<Real>()
                    .unwrap();
                SignalKind::Real(float_value)
            }
            SignalEncoding::Unknown => unreachable!("Unknown signal encoding!"),
        };
    }
}

impl<'a> From<&'a SignalValueBuffer> for SignalValueRef<'a> {
    fn from(value: &'a SignalValueBuffer) -> Self {
        match value.kind {
            SignalKind::Event => SignalValueRef::Event,
            SignalKind::String => SignalValueRef::String(&value.string),
            SignalKind::BitVec(states, width) => {
                SignalValueRef::bit_vec(states, width, &value.data)
            }
            SignalKind::Real(value) => SignalValueRef::Real(value),
        }
    }
}

impl<C> StreamEncoder<C>
where
    C: FnMut(Time, SignalRef, SignalValueRef<'_>),
{
    pub(crate) fn new(hierarchy: &Hierarchy, filter: &Filter, callback: C) -> Self {
        // remember encoding information for all included signals
        let encoding = match filter.signals {
            None => hierarchy.signal_encodings().into(),
            Some([]) => SignalMap::sparse(),
            Some(signals) => {
                debug_assert!(
                    signals.iter().all(|s| !s.is_derived_signal()),
                    "derived signals are not supported in the StreamEncoder!"
                );
                SignalMap::from_iter(
                    signals
                        .iter()
                        .map(|&s| (s, hierarchy.get_signal_tpe(s).unwrap())),
                )
            }
        };

        Self {
            callback,
            time: Default::default(),
            skipping_time_step: false,
            encoding,
            buf: SignalValueBuffer::default(),
        }
    }

    fn get_encoding(&self, id: u64) -> SignalEncoding {
        self.encoding
            .get_index(id)
            .cloned()
            .unwrap_or(SignalEncoding::Unknown)
    }

    pub(crate) fn fst_value_change(&mut self, time: u64, id: u64, value: &FstSignalValue) {
        debug_assert!(
            !self.skipping_time_step,
            "fst reader should filter out time steps"
        );

        // emit a fake time step
        if self.time.is_none_or(|t| time > t) {
            self.time_change(time);
        }

        // check to see if the signal should be included
        let encoding = self.get_encoding(id);
        if encoding != SignalEncoding::Unknown {
            let signal_ref = SignalRef::from_index(id as usize).unwrap();
            self.buf.update_fst(encoding, value);
            (self.callback)(time, signal_ref, (&self.buf).into())
        }
    }

    pub(crate) fn vcd_value_change(&mut self, id: u64, value: &[u8]) {
        if self.skipping_time_step {
            return;
        }
        // check to see if the signal should be included
        let encoding = self.get_encoding(id);
        if encoding != SignalEncoding::Unknown {
            let signal_ref = SignalRef::from_index(id as usize).unwrap();
            let time = self.time.unwrap();
            self.buf.update_vcd(encoding, value);
            (self.callback)(time, signal_ref, (&self.buf).into())
        }
    }

    pub(crate) fn time_change(&mut self, time: u64) {
        // sanity check to make sure that time is increasing
        if let Some(prev_time) = self.time {
            match prev_time.cmp(&time) {
                std::cmp::Ordering::Equal => {
                    return; // ignore calls to time_change that do not actually change anything
                }
                std::cmp::Ordering::Greater => {
                    println!("WARN: time decreased from {prev_time} to {time}. Skipping!");
                    self.skipping_time_step = true;
                    return;
                }
                std::cmp::Ordering::Less => {
                    // this is the normal situation where we actually increment the time
                }
            }
        }

        // TODO: check filter to see if we are done or what!
        self.time = Some(time);
        self.skipping_time_step = false;
    }

    pub(crate) fn time_is_none(&self) -> bool {
        self.time.is_none()
    }

    pub fn finish(&mut self) {
        //  nothing to do right now
    }
}

/// Contains information derived from the filter for streaming.
/// Used by several stream dispatchers.
struct StreamDerivedSignalInfo {
    requested: Option<FxHashSet<SignalRef>>,
    to_derived: SignalMap<SmallVec<[SignalRef; 4]>>,
    transforms: SignalMap<DerivedBitVecSignal>,
    new_filter: Option<Vec<SignalRef>>,
}

impl StreamDerivedSignalInfo {
    fn compute(hierarchy: &Hierarchy, filter: &Filter) -> Self {
        // data for derived signals
        let mut transforms = SignalMap::sparse();

        // any extra signals that were not included in the original filter
        let mut extras = vec![];

        // remember encoding information for all included signals
        let requested = match filter.signals {
            None => {
                transforms = SignalMap::from_iter(
                    hierarchy.all_derived_signals().map(|(s, t)| (s, t.clone())),
                );
                None
            }
            Some([]) => Some(FxHashSet::default()),
            Some(signals) => {
                let requested = FxHashSet::from_iter(signals.iter().cloned());
                for &signal in signals {
                    if let Some(transform) = hierarchy.get_derived_signal(signal) {
                        debug_assert!(signal.is_derived_signal());
                        for &input in transform.inputs() {
                            if !requested.contains(&input) {
                                extras.push(input);
                            }
                        }
                        transforms.insert(signal, transform.clone());
                    } else {
                        debug_assert!(!signal.is_derived_signal());
                    }
                }
                Some(requested)
            }
        };

        // save all transforms
        let mut to_derived = SignalMap::sparse();
        for (&signal_ref, transform) in transforms.iter() {
            for &input in transform.inputs() {
                to_derived
                    .entry(input)
                    .or_insert_with(|| smallvec![])
                    .push(signal_ref);
            }
        }

        // construct a new filter if necessary
        let new_filter = if !extras.is_empty()
            && let Some(orig) = filter.signals
        {
            extras.sort();
            extras.dedup();
            // there are extra signals and the filter does not just include all signals
            let mut new_signals: Vec<_> = orig
                .iter()
                .filter(|s| !s.is_derived_signal())
                .cloned()
                .collect();
            new_signals.append(&mut extras);
            Some(new_signals)
        } else {
            None
        };

        Self {
            requested,
            to_derived,
            transforms,
            new_filter,
        }
    }
}

/// Handles derived signals for a callback that is invoked at every change.
/// Gets hooked into the [[StreamEncoder]] through a closure.
struct StreamDispatcherOnChange<C>
where
    C: FnMut(Time, SignalRef, SignalValueRef<'_>),
{
    callback: C,
    /// to track when a time change occures
    time: Option<Time>,
    /// signals that were requested by the user, None means all signals were requested
    requested: Option<FxHashSet<SignalRef>>,
    /// signal value cache, used for derived signals and their inputs
    values: SignalMap<SignalValue>,
    /// what signals are derived from the key?
    to_derived: SignalMap<SmallVec<[SignalRef; 4]>>,
    /// keep track of which derived signals had changes this time step
    has_changed: FxHashSet<SignalRef>,
    transforms: SignalMap<DerivedBitVecSignal>,
}

impl<C> StreamDispatcherOnChange<C>
where
    C: FnMut(Time, SignalRef, SignalValueRef<'_>),
{
    fn new(hierarchy: &Hierarchy, filter: &Filter, callback: C) -> (Self, Option<Vec<SignalRef>>) {
        let info = StreamDerivedSignalInfo::compute(hierarchy, filter);
        let new_filter = info.new_filter;
        let out = Self {
            callback,
            time: None,
            requested: info.requested,
            values: SignalMap::sparse(),
            to_derived: info.to_derived,
            has_changed: Default::default(),
            transforms: info.transforms,
        };

        (out, new_filter)
    }

    /// Will be called by the [[StreamEncoder]]
    fn on_change(&mut self, time: Time, signal: SignalRef, value: SignalValueRef) {
        // emit derived signal changes at the end of a a timestep
        if let Some(prev) = self.time
            && time > prev
        {
            self.emit_derived_signal_changes();
        }
        self.time = Some(time);
        self.update_derived(signal, value);
        self.dispatch_change(time, signal, value)
    }

    /// Checks whether the signal is an input to a derived signal and deals with that.
    #[inline]
    fn update_derived(&mut self, signal_ref: SignalRef, value: SignalValueRef) {
        if let Some(derived) = self.to_derived.get(&signal_ref) {
            self.values.insert(signal_ref, value.into());
            for &signal in derived.iter() {
                self.has_changed.insert(signal);
            }
        }
    }

    #[inline]
    fn dispatch_change(&mut self, time: Time, signal_ref: SignalRef, value: SignalValueRef) {
        if self
            .requested
            .as_ref()
            .map(|r| r.contains(&signal_ref))
            .unwrap_or(true)
        {
            (self.callback)(time, signal_ref, value);
        }
    }

    fn emit_derived_signal_changes(&mut self) {
        if !self.has_changed.is_empty() {
            let time = self
                .time
                .expect("time cannot be None when there are changes");
            for signal in self.has_changed.drain() {
                let t = &self.transforms.get(&signal).unwrap();
                let inputs: Vec<_> = t
                    .inputs()
                    .iter()
                    .map(|i| {
                        self.values
                            .get(i)
                            .map(|v| SignalValueRef::from(v).as_bit_vec().unwrap())
                    })
                    .collect();
                let value = t.on_change(&inputs);
                (self.callback)(time, signal, (&value).into());
            }
        }
    }

    /// Must be called at the end of a stream. Dispatches any pending derived signal changes.
    pub(crate) fn finish(&mut self) {
        self.emit_derived_signal_changes();
    }
}

/// Buffers signal changes and only invokes the call back once at the end of each time step.
/// Gets hooked into the [[StreamEncoder]] through a closure.
struct StreamDispatcherOnTimeStep<C>
where
    C: FnMut(Time, &SignalMap<SignalValue>),
{
    callback: C,
    /// to track when a time change occurres
    time: Option<Time>,
    /// most recent value of each signal
    values: SignalMap<SignalValue>,
    /// what signals are derived from the key?
    to_derived: SignalMap<SmallVec<[SignalRef; 4]>>,
    transforms: SignalMap<DerivedBitVecSignal>,
    /// keep track of which derived signals had changes this time step
    derived_input_has_changed: FxHashSet<SignalRef>,
    /// has there been a change since the last dispatch
    observed_change: bool,
}

impl<C> StreamDispatcherOnTimeStep<C>
where
    C: FnMut(Time, &SignalMap<SignalValue>),
{
    fn new(hierarchy: &Hierarchy, filter: &Filter, callback: C) -> (Self, Option<Vec<SignalRef>>) {
        let info = StreamDerivedSignalInfo::compute(hierarchy, filter);
        let new_filter = info.new_filter;

        let values = if filter.signals.is_none() {
            SignalMap::dense()
        } else {
            SignalMap::sparse()
        };

        let out = Self {
            callback,
            time: None,
            values,
            to_derived: info.to_derived,
            transforms: info.transforms,
            derived_input_has_changed: FxHashSet::default(),
            observed_change: false,
        };

        (out, new_filter)
    }

    /// Will be called by the [[StreamEncoder]]
    fn on_change(&mut self, time: Time, signal: SignalRef, value: SignalValueRef) {
        // emit derived signal changes at the end of a a timestep
        if let Some(prev) = self.time
            && time > prev
        {
            self.dispatch();
        }
        self.time = Some(time);

        let changed = self
            .values
            .get(&signal)
            .map(|old| SignalValueRef::from(old) != value)
            .unwrap_or(true);
        if changed {
            if time == 605000000 {
                println!(
                    "{time} {signal:?} {:?} -> {value:?}",
                    self.values.get(&signal)
                );
            }
            self.values.insert(signal, value.into());
            self.update_has_changed(signal);
        }
        if value.is_event() || changed {
            self.observed_change = true;
        }
    }

    fn dispatch(&mut self) {
        if self.observed_change {
            self.update_derived_signal_changes();
            let time = self
                .time
                .expect("dispatch should only be called when we know the time");
            (self.callback)(time, &self.values);
            self.observed_change = false;
        }
    }

    #[inline]
    fn update_has_changed(&mut self, signal_ref: SignalRef) {
        if let Some(derived) = self.to_derived.get(&signal_ref) {
            for &signal in derived.iter() {
                self.derived_input_has_changed.insert(signal);
            }
        }
    }

    fn update_derived_signal_changes(&mut self) {
        if !self.derived_input_has_changed.is_empty() {
            for signal in self.derived_input_has_changed.drain() {
                let t = &self.transforms.get(&signal).unwrap();
                let inputs: Vec<_> = t
                    .inputs()
                    .iter()
                    .map(|i| {
                        self.values
                            .get(i)
                            .map(|v| SignalValueRef::from(v).as_bit_vec().unwrap())
                    })
                    .collect();
                let value = t.on_change(&inputs);
                self.values.insert(signal, value.into());
            }
        }
    }

    /// Must be called at the end of a stream. Dispatches any pending derived signal changes.
    pub(crate) fn finish(&mut self) {
        // time can be none if we never observed a single change
        if self.time.is_some() {
            self.dispatch();
        }
    }
}
