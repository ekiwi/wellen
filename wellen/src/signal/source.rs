// Copyright 2023-2024 The Regents of the University of California
// Copyright 2024-2026 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use crate::signal::transform::slice_signal;
use crate::{Hierarchy, Signal, SignalEncoding, SignalRef};

pub trait SignalSourceImplementation: Sync + Send {
    /// Loads new signals.
    /// Many implementations take advantage of loading multiple signals at a time.
    fn load_signals(
        &mut self,
        ids: &[SignalRef],
        types: &[SignalEncoding],
        multi_threaded: bool,
    ) -> Vec<Signal>;
    /// Print memory size / speed statistics.
    fn print_statistics(&self);
}

pub struct SignalSource {
    inner: Box<dyn SignalSourceImplementation>,
}

impl SignalSource {
    pub fn new(inner: Box<dyn SignalSourceImplementation + Send + Sync>) -> Self {
        Self { inner }
    }

    /// Loads new signals.
    /// Many implementations take advantage of loading multiple signals at a time.
    pub fn load_signals(
        &mut self,
        ids: &[SignalRef],
        hierarchy: &Hierarchy,
        multi_threaded: bool,
    ) -> Vec<(SignalRef, Signal)> {
        // sort and dedup ids
        let mut ids = Vec::from_iter(ids.iter().cloned());
        ids.sort();
        ids.dedup();

        // replace any aliases by their source signal
        let orig_ids = ids.clone();
        let mut is_alias = vec![false; ids.len()];
        for (ii, id) in ids.iter_mut().enumerate() {
            if let Some(slice) = hierarchy.get_slice_info(*id) {
                *id = slice.sliced_signal;
                is_alias[ii] = true;
            }
        }

        // collect meta data
        let types: Vec<_> = ids
            .iter()
            .map(|i| hierarchy.get_signal_tpe(*i).unwrap())
            .collect();
        let signals = self.inner.load_signals(&ids, &types, multi_threaded);
        // the signal source must always return the correct number of signals!
        assert_eq!(signals.len(), ids.len());
        let mut out = Vec::with_capacity(orig_ids.len());
        for ((id, is_alias), signal) in orig_ids.iter().zip(is_alias.iter()).zip(signals) {
            if *is_alias {
                let slice = hierarchy.get_slice_info(*id).unwrap();
                let sliced = slice_signal(*id, &signal, slice.msb, slice.lsb);
                out.push((*id, sliced));
            } else {
                out.push((*id, signal));
            }
        }
        out
    }

    /// Print memory size / speed statistics.
    pub fn print_statistics(&self) {
        self.inner.print_statistics();
    }
}
