// Copyright 2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>
//! # Stream Interface
//! This interface is useful when you want to batch process waveform data instead
//! of displaying it in a waveform viewer.

use crate::vcd::{VcdBitVecChange, decode_vcd_bit_vec_change};
use crate::wavemem::{States, check_states, write_n_state};
use crate::{
    FileFormat, Hierarchy, LoadOptions, Real, Result, SignalEncoding, SignalRef, SignalValue, Time,
    WellenError, viewers,
};
use fst_reader::FstSignalValue;
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

    pub fn stream(
        &mut self,
        filter: &Filter,
        callback: impl FnMut(Time, SignalRef, SignalValue<'_>),
    ) -> Result<()> {
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
}

/// Takes on the role of the [Encoder] when streaming instead of encoding to
/// a wavemem.
pub(crate) struct StreamEncoder<C>
where
    C: FnMut(Time, SignalRef, SignalValue<'_>),
{
    callback: C,
    time: Option<Time>,
    skipping_time_step: bool,
    /// contains encoding for all _included_ signals (depending on the [Filter] provided)
    encoding: Vec<Option<SignalEncoding>>,
    buf: Vec<u8>,
}

impl<C> StreamEncoder<C>
where
    C: FnMut(Time, SignalRef, SignalValue<'_>),
{
    pub(crate) fn new(hierarchy: &Hierarchy, filter: &Filter, callback: C) -> Self {
        // remember encoding information for all included signals
        let encoding = match filter.signals {
            None => {
                // all signals
                hierarchy
                    .get_unique_signals_vars()
                    .into_iter()
                    .map(|var| {
                        Some(match var {
                            None => SignalEncoding::String, // we do not know!
                            Some(var) => var.signal_encoding(),
                        })
                    })
                    .collect::<Vec<_>>()
            }
            Some([]) => {
                // nothing
                vec![]
            }
            Some(signals) => {
                let max_index = signals.iter().map(|r| r.index()).max().unwrap();
                let mut enc = vec![None; max_index + 1];
                for &signal in signals {
                    enc[signal.index()] = hierarchy.get_signal_tpe(signal);
                }
                enc
            }
        };

        Self {
            callback,
            time: Default::default(),
            skipping_time_step: false,
            encoding,
            buf: Vec::with_capacity(128),
        }
    }

    pub(crate) fn fst_value_change(&mut self, time: u64, id: u64, value: &FstSignalValue) {
        debug_assert!(
            !self.skipping_time_step,
            "fst reader should filter out time steps"
        );

        // check to see if the signal should be included
        let maybe_tpe = self.encoding.get(id as usize).and_then(|a| a.as_ref());
        #[allow(unused_assignments)]
        let mut maybe_str = None;
        if let Some(tpe) = maybe_tpe.cloned() {
            let signal_ref = SignalRef::from_index(id as usize).unwrap();
            let signal_value = match value {
                FstSignalValue::String(value) => match tpe {
                    SignalEncoding::Event => {
                        debug_assert!(value.is_empty(), "events do not carry data");
                        SignalValue::Event
                    }
                    SignalEncoding::String => {
                        maybe_str = Some(String::from_utf8_lossy(value));
                        SignalValue::String(maybe_str.as_ref().unwrap())
                    }

                    SignalEncoding::BitVector(len) => {
                        let bits = len.get();

                        debug_assert_eq!(
                            value.len(),
                            bits as usize,
                            "{}",
                            String::from_utf8_lossy(value)
                        );

                        let states = check_states(value).unwrap_or_else(|| {
                            panic!(
                                "Unexpected signal value: {}",
                                String::from_utf8_lossy(value)
                            )
                        });

                        // convert from ASCII characters to packed encoding
                        self.buf.clear();
                        write_n_state(states, value, &mut self.buf, None);

                        match states {
                            States::Two => SignalValue::Binary(&self.buf, bits),
                            States::Four => SignalValue::FourValue(&self.buf, bits),
                            States::Nine => SignalValue::NineValue(&self.buf, bits),
                        }
                    }
                    SignalEncoding::Real => panic!(
                        "Expecting reals, but got: {}",
                        String::from_utf8_lossy(value)
                    ),
                },
                FstSignalValue::Real(value) => {
                    debug_assert_eq!(tpe, SignalEncoding::Real);
                    SignalValue::Real(*value)
                }
            };

            (self.callback)(time, signal_ref, signal_value);
        }
    }

    pub(crate) fn vcd_value_change(&mut self, id: u64, value: &[u8]) {
        if self.skipping_time_step {
            return;
        }
        // check to see if the signal should be included
        let maybe_tpe = self.encoding.get(id as usize).and_then(|a| a.as_ref());
        if let Some(tpe) = maybe_tpe {
            let signal_ref = SignalRef::from_index(id as usize).unwrap();
            let time = self.time.unwrap();
            self.buf.clear();
            let signal_value = match tpe {
                SignalEncoding::Event => {
                    debug_assert!(
                        value.len() <= 1,
                        "event changes carry no value, or a 1-bit value"
                    );
                    SignalValue::Event
                }
                &SignalEncoding::BitVector(len) => {
                    let (data, states) = decode_vcd_bit_vec_change(len, value);

                    // put data into buffer
                    match data {
                        VcdBitVecChange::SingleBit(bit_value) => {
                            self.buf.push(bit_value);
                        }
                        VcdBitVecChange::MultiBit(data_to_write) => {
                            write_n_state(states, &data_to_write, &mut self.buf, None);
                        }
                    }

                    // construct signal data based on number of states
                    match states {
                        States::Two => SignalValue::Binary(&self.buf, len.get()),
                        States::Four => SignalValue::FourValue(&self.buf, len.get()),
                        States::Nine => SignalValue::NineValue(&self.buf, len.get()),
                    }
                }
                SignalEncoding::String => {
                    assert!(
                        matches!(value[0], b's' | b'S'),
                        "expected a string, not {}",
                        String::from_utf8_lossy(value)
                    );
                    let characters = &value[1..];
                    SignalValue::String(std::str::from_utf8(characters).unwrap())
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
                    SignalValue::Real(float_value)
                }
            };

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
        // TODO: check filter to see if we are done or what!
        self.time = Some(time);
        self.skipping_time_step = false;
    }

    pub(crate) fn time_is_none(&self) -> bool {
        self.time.is_none()
    }
}
