// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use std::cmp::Ordering;

/// A dense map implemented by a Vec. Get will return the default value, if the index has not been
/// written, but is in range.
pub struct DenseHashMap<V>
where
    V: Default + Clone,
{
    entries: Vec<V>,
}

impl<V> Default for DenseHashMap<V>
where
    V: Default + Clone,
{
    fn default() -> Self {
        DenseHashMap {
            entries: Vec::default(),
        }
    }
}

impl<V> DenseHashMap<V>
where
    V: Default + Clone,
{
    pub fn with_capacity(capacity: usize) -> Self {
        DenseHashMap {
            entries: Vec::with_capacity(capacity),
        }
    }

    pub fn insert(&mut self, key: usize, value: V) {
        match key.cmp(&self.entries.len()) {
            Ordering::Less => {
                self.entries[key] = value;
            }
            Ordering::Equal => {
                self.entries.push(value);
            }
            Ordering::Greater => {
                self.entries.resize(key, V::default());
                self.entries.push(value);
            }
        }
    }
    pub fn get(&self, key: usize) -> Option<&V> {
        self.entries.get(key)
    }
    pub fn get_mut(&mut self, key: usize) -> Option<&mut V> {
        self.entries.get_mut(key)
    }
    pub fn get_or_else_create_mut(&mut self, key: usize) -> &mut V {
        if self.entries.len() <= key {
            self.entries.resize(key + 1, V::default());
        }
        &mut self.entries[key]
    }
    pub fn into_vec(self) -> Vec<V> {
        self.entries
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}
