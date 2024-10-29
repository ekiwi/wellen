use std::{
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
};

use crate::{Signal, TimeTableIdx};

#[derive(Debug, Eq)]
pub struct ChangesWithIdx<'a> {
    arr: &'a [TimeTableIdx],
    idx: usize,
}

impl<'a> PartialEq for ChangesWithIdx<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.get_item() == other.get_item()
    }
}

impl<'a> PartialOrd for ChangesWithIdx<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.get_item().partial_cmp(&other.get_item())
    }
}

impl<'a> Ord for ChangesWithIdx<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.get_item().cmp(&other.get_item())
    }
}

impl<'a> ChangesWithIdx<'a> {
    pub fn new(arr: &'a [TimeTableIdx], idx: usize) -> Self {
        Self { arr, idx }
    }

    pub fn get_item(&self) -> TimeTableIdx {
        self.arr[self.idx]
    }
}

pub fn all_changes<'a>(arrays: impl IntoIterator<Item = &'a Signal>) -> Vec<TimeTableIdx> {
    merge_indices(arrays.into_iter().map(|sig| sig.time_indices()).collect())
}

fn merge_indices(arrays: Vec<&[TimeTableIdx]>) -> Vec<TimeTableIdx> {
    let mut sorted = vec![];

    let mut heap = BinaryHeap::with_capacity(arrays.len());
    for arr in arrays {
        let item = ChangesWithIdx::new(arr, 0);
        heap.push(Reverse(item));
    }

    while !heap.is_empty() {
        let mut it = heap.pop().unwrap();
        sorted.push(it.0.get_item());
        it.0.idx += 1;
        if it.0.idx < it.0.arr.len() {
            heap.push(it)
        }
    }

    sorted
}
