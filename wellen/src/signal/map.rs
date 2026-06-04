// Copyright 2026 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use crate::SignalRef;
use rustc_hash::FxHashMap;
use std::fmt::{Debug, Formatter};
use std::ops::Index;

/// A map that is indexed by SignalRef.
pub struct SignalMap<V>(Smi<V>);

/// Signal Map Implementation
enum Smi<V> {
    Dense(Vec<Option<V>>),
    Sparse(FxHashMap<SignalRef, V>),
}

impl<V> SignalMap<V> {
    pub fn dense() -> Self {
        Self(Smi::Dense(vec![]))
    }

    pub fn sparse() -> Self {
        Self(Smi::Sparse(FxHashMap::default()))
    }

    pub fn from_iter<T: IntoIterator<Item = (SignalRef, V)>>(iter: T) -> Self {
        Self(Smi::Sparse(FxHashMap::from_iter(iter)))
    }

    pub fn insert(&mut self, k: SignalRef, v: V) -> Option<V> {
        match &mut self.0 {
            Smi::Dense(vec) => {
                let index = k.index();
                if vec.len() <= index {
                    vec.resize_with(index + 1, || None);
                }
                let mut opt_v = Some(v);
                std::mem::swap(&mut vec[index], &mut opt_v);
                opt_v
            }
            Smi::Sparse(m) => m.insert(k, v),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&SignalRef, &V)> {
        match &self.0 {
            Smi::Dense(_) => todo!(),
            Smi::Sparse(m) => m.iter(),
        }
    }

    /// like `get`, but the key is a [[u64]] index which will be interpreted as a SignalRef
    pub fn get_index(&self, k: u64) -> Option<&V> {
        match &self.0 {
            Smi::Dense(v) => v.get(k as usize).and_then(|o| o.as_ref()),
            Smi::Sparse(m) => m.get(&SignalRef::from_index(k as usize).unwrap()),
        }
    }

    pub fn get(&self, k: &SignalRef) -> Option<&V> {
        match &self.0 {
            Smi::Dense(v) => v.get(k.index()).and_then(|o| o.as_ref()),
            Smi::Sparse(m) => m.get(k),
        }
    }

    pub fn entry(&mut self, key: SignalRef) -> Entry<'_, V> {
        match &mut self.0 {
            Smi::Dense(v) => Entry(EI::Dense(v, key.index())),
            Smi::Sparse(m) => Entry(EI::Sparse(m.entry(key))),
        }
    }
}

impl<V> Index<SignalRef> for SignalMap<V> {
    type Output = V;

    fn index(&self, index: SignalRef) -> &Self::Output {
        match &self.0 {
            Smi::Dense(v) => v[index.index()].as_ref().unwrap(),
            Smi::Sparse(m) => &m[&index],
        }
    }
}

impl<V: Clone> Clone for SignalMap<V> {
    fn clone(&self) -> Self {
        match &self.0 {
            Smi::Dense(v) => Self(Smi::Dense(v.clone())),
            Smi::Sparse(m) => Self(Smi::Sparse(m.clone())),
        }
    }
}

impl<V> From<Vec<Option<V>>> for SignalMap<V> {
    fn from(value: Vec<Option<V>>) -> Self {
        Self(Smi::Dense(value))
    }
}

impl<V: Clone> From<&[V]> for SignalMap<V> {
    fn from(value: &[V]) -> Self {
        Self(Smi::Dense(value.iter().cloned().map(Some).collect()))
    }
}

impl<V> From<FxHashMap<SignalRef, V>> for SignalMap<V> {
    fn from(value: FxHashMap<SignalRef, V>) -> Self {
        Self(Smi::Sparse(value))
    }
}

impl<V: Debug> Debug for SignalMap<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Smi::Dense(v) => v.fmt(f),
            Smi::Sparse(m) => m.fmt(f),
        }
    }
}

pub struct Entry<'a, V>(EI<'a, V>);

enum EI<'a, V> {
    Dense(&'a mut Vec<Option<V>>, usize),
    Sparse(std::collections::hash_map::Entry<'a, SignalRef, V>),
}

impl<'a, V> Entry<'a, V> {
    pub fn or_insert_with<F: FnOnce() -> V>(self, default: F) -> &'a mut V {
        match self.0 {
            EI::Dense(vec, index) => {
                if vec.len() <= index {
                    vec.resize_with(index + 1, || None);
                }
                if vec[index].is_none() {
                    vec[index] = Some(default());
                }
                vec[index].as_mut().unwrap()
            }
            EI::Sparse(e) => e.or_insert_with(default),
        }
    }
}
