// Copyright 2025-26 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>
//! # Stream Interface
//! This interface is useful when you want to batch process waveform data instead
//! of displaying it in a waveform viewer.

use crate::signal::{DerivedBitVecSignal, States};
use crate::vcd::{VcdBitVecChange, decode_vcd_bit_vec_change};
use crate::wavemem::write_n_state_from_ascii;
use crate::{
    FileFormat, Hierarchy, LoadOptions, Real, Result, SignalEncoding, SignalRef, SignalValue,
    SignalValueRef, Time, WellenError, viewers,
};
use fst_reader::FstSignalValue;
use rustc_hash::{FxHashMap, FxHashSet};
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
}

impl<R: BufRead + Seek> StreamingWaveform<R> {
    pub fn hierarchy(&self) -> &Hierarchy {
        &self.hierarchy
    }

    /// Calls the callback for each change of a signal.
    pub fn stream_changes(
        &mut self,
        filter: &Filter,
        callback: impl FnMut(Time, SignalRef, SignalValueRef<'_>),
    ) -> Result<()> {
        // ensure that none of the signals are slices
        match &mut self.body {
            viewers::ReadBodyData::Vcd(data) => {
                crate::vcd::stream_body(data, &self.hierarchy, filter, callback)?
            }
            viewers::ReadBodyData::Fst(data) => {
                crate::fst::stream_body(data, &self.hierarchy, filter, callback)?
            }
            viewers::ReadBodyData::Ghw(_) => panic!("streaming GHW files is not supported"),
        }
        Ok(())
    }

    pub fn stream_time_steps(&mut self, filter: &Filter) -> Result<()> {
        todo!()
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
    encoding: Vec<SignalEncoding>,
    buf: Vec<u8>,
    /// signal value cache, used for derived signals and their inputs
    values: FxHashMap<SignalRef, SignalValue>,
    /// what signals are derived from the key?
    to_derived: FxHashMap<SignalRef, SmallVec<[SignalRef; 4]>>,
    /// keep track of which derived signals had changes this time step
    has_changed: FxHashSet<SignalRef>,
    transforms: FxHashMap<SignalRef, DerivedBitVecSignal>,
}

impl<C> StreamEncoder<C>
where
    C: FnMut(Time, SignalRef, SignalValueRef<'_>),
{
    pub(crate) fn new(hierarchy: &Hierarchy, filter: &Filter, callback: C) -> Self {
        // data for derived signals
        let mut transforms = FxHashMap::default();

        // remember encoding information for all included signals
        let mut encoding = match filter.signals {
            None => {
                // collect info for derived signals
                for (signal_ref, transform) in hierarchy.all_derived_signals() {
                    transforms.insert(signal_ref, transform.clone());
                }

                // all signals
                hierarchy.signal_encodings().into()
            }
            Some([]) => {
                // nothing
                vec![]
            }
            Some(signals) => {
                let max_index = signals.iter().map(|r| r.index()).max().unwrap();
                let mut enc = vec![SignalEncoding::Unknown; max_index + 1];
                for &signal in signals {
                    if let Some(transform) = hierarchy.get_derived_signal(signal) {
                        debug_assert!(signal.is_derived_signal());
                        transforms.insert(signal, transform.clone());
                    } else {
                        debug_assert!(!signal.is_derived_signal());
                        enc[signal.index()] = hierarchy.get_signal_tpe(signal).unwrap();
                    }
                }
                enc
            }
        };

        let mut to_derived = FxHashMap::default();
        for (&signal_ref, transform) in transforms.iter() {
            for &input in transform.inputs() {
                to_derived
                    .entry(input)
                    .or_insert_with(|| smallvec![])
                    .push(signal_ref);
                encoding[input.index()] = hierarchy.get_signal_tpe(input).unwrap();
            }
        }

        Self {
            callback,
            time: Default::default(),
            skipping_time_step: false,
            encoding,
            buf: Vec::with_capacity(128),
            values: FxHashMap::default(),
            to_derived,
            has_changed: Default::default(),
            transforms,
        }
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
        let tpe = self
            .encoding
            .get(id as usize)
            .cloned()
            .unwrap_or(SignalEncoding::Unknown);
        #[allow(unused_assignments)]
        let mut maybe_str = None;
        if tpe != SignalEncoding::Unknown {
            let signal_ref = SignalRef::from_index(id as usize).unwrap();
            let signal_value = match value {
                FstSignalValue::String(value) => match tpe {
                    SignalEncoding::Event => {
                        debug_assert!(value.is_empty(), "events do not carry data");
                        SignalValueRef::Event
                    }
                    SignalEncoding::String => {
                        maybe_str = Some(String::from_utf8_lossy(value));
                        SignalValueRef::String(maybe_str.as_ref().unwrap())
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
                        self.buf.clear();
                        write_n_state_from_ascii(states, value, &mut self.buf, None);
                        SignalValueRef::bit_vec(states, width, &self.buf)
                    }
                    SignalEncoding::Real => panic!(
                        "Expecting reals, but got: {}",
                        String::from_utf8_lossy(value)
                    ),
                    SignalEncoding::Unknown => unreachable!("Unknown signal encoding!"),
                },
                FstSignalValue::Real(value) => {
                    debug_assert_eq!(tpe, SignalEncoding::Real);
                    SignalValueRef::Real(*value)
                }
            };

            if let Some(derived) = self.to_derived.get(&signal_ref) {
                self.values.insert(signal_ref, signal_value.into());
                for &signal in derived.iter() {
                    self.has_changed.insert(signal);
                }
            }
            (self.callback)(time, signal_ref, signal_value);
        }
    }

    pub(crate) fn vcd_value_change(&mut self, id: u64, value: &[u8]) {
        if self.skipping_time_step {
            return;
        }
        // check to see if the signal should be included
        let tpe = self
            .encoding
            .get(id as usize)
            .cloned()
            .unwrap_or(SignalEncoding::Unknown);
        if tpe != SignalEncoding::Unknown {
            let signal_ref = SignalRef::from_index(id as usize).unwrap();
            let time = self.time.unwrap();
            self.buf.clear();
            let signal_value = match tpe {
                SignalEncoding::Event => {
                    debug_assert!(
                        value.len() <= 1,
                        "event changes carry no value, or a 1-bit value"
                    );
                    SignalValueRef::Event
                }
                SignalEncoding::BitVector(width) => {
                    let (data, states) = decode_vcd_bit_vec_change(width, value);

                    // put data into buffer
                    match data {
                        VcdBitVecChange::SingleBit(bit_value) => {
                            self.buf.push(bit_value.into());
                        }
                        VcdBitVecChange::MultiBit(data_to_write) => {
                            write_n_state_from_ascii(states, &data_to_write, &mut self.buf, None);
                        }
                    }
                    SignalValueRef::bit_vec(states, width.get(), &self.buf)
                }
                SignalEncoding::String => {
                    assert!(
                        matches!(value[0], b's' | b'S'),
                        "expected a string, not {}",
                        String::from_utf8_lossy(value)
                    );
                    let characters = &value[1..];
                    SignalValueRef::String(std::str::from_utf8(characters).unwrap())
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
                    SignalValueRef::Real(float_value)
                }
                SignalEncoding::Unknown => unreachable!("Unknown signal encoding!"),
            };

            if let Some(derived) = self.to_derived.get(&signal_ref) {
                self.values.insert(signal_ref, signal_value.into());
                for &signal in derived.iter() {
                    self.has_changed.insert(signal);
                }
            }
            (self.callback)(time, signal_ref, signal_value);
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
        // Emit derived signals.
        self.emit_derived_signal_changes();

        // TODO: check filter to see if we are done or what!
        self.time = Some(time);
        self.skipping_time_step = false;
    }

    /// Must be called at the end of a stream. Dispatches any pending derived signal changes.
    pub(crate) fn finish(&mut self) {
        self.emit_derived_signal_changes();
    }

    fn emit_derived_signal_changes(&mut self) {
        if !self.has_changed.is_empty() {
            let time = self
                .time
                .expect("time cannot be None when there are changes");
            for signal in self.has_changed.drain() {
                let t = &self.transforms[&signal];
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

    pub(crate) fn time_is_none(&self) -> bool {
        self.time.is_none()
    }
}
