// Copyright 2023-2024 The Regents of the University of California
// Copyright 2024-2026 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use crate::signal::SignalTransform;
use crate::{Hierarchy, Signal, SignalEncoding, SignalRef};
use rustc_hash::FxHashMap;

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
    /// A note on derived signals: this function will load the underlying signals and compute the
    /// derived signals. However, since we do not have access to signals that were loaded
    /// in a previous call to this function, performance may be suboptimal.
    /// Thus, if you can, consider handling derived signals in the caller and only passing
    /// non-derived signals that are needed to this function.
    pub fn load_signals(
        &mut self,
        ids: &[SignalRef],
        hierarchy: &Hierarchy,
        multi_threaded: bool,
    ) -> Vec<Signal> {
        let (derived, signals): (Vec<_>, Vec<_>) = ids.iter().partition(|s| s.is_derived_signal());
        let mut out = self.load_non_derived_signals(signals, hierarchy, multi_threaded);
        let mut others = self.load_derived_signals(derived, hierarchy, multi_threaded);
        out.append(&mut others);
        out
    }

    fn load_non_derived_signals(
        &mut self,
        mut ids: Vec<SignalRef>,
        hierarchy: &Hierarchy,
        multi_threaded: bool,
    ) -> Vec<Signal> {
        ids.sort();
        ids.dedup();
        debug_assert!(ids.iter().all(|s| !s.is_derived_signal()));
        let enc: Vec<_> = ids
            .iter()
            .map(|i| hierarchy.get_signal_tpe(*i).unwrap())
            .collect();
        let signals = self.inner.load_signals(&ids, &enc, multi_threaded);
        assert_eq!(
            signals.len(),
            ids.len(),
            "the signal source must always return the correct number of signals!"
        );
        signals
    }

    fn load_derived_signals(
        &mut self,
        mut ids: Vec<SignalRef>,
        hierarchy: &Hierarchy,
        multi_threaded: bool,
    ) -> Vec<Signal> {
        ids.sort();
        ids.dedup();
        debug_assert!(ids.iter().all(|s| s.is_derived_signal()));

        let transforms: Vec<_> = ids
            .iter()
            .map(|s| hierarchy.get_derived_signal(*s).unwrap())
            .collect();
        let underlying_ids: Vec<_> = transforms
            .iter()
            .flat_map(|d| d.inputs())
            .cloned()
            .collect();
        let _underlying_signals: FxHashMap<_, _> = self
            .load_non_derived_signals(underlying_ids, hierarchy, multi_threaded)
            .into_iter()
            .map(|s| (s.idx, s))
            .collect();
        ids.into_iter()
            .zip(transforms)
            .map(|(_s, _transform)| todo!("apply transform to signal!"))
            .collect()
    }

    /// Print memory size / speed statistics.
    pub fn print_statistics(&self) {
        self.inner.print_statistics();
    }
}
