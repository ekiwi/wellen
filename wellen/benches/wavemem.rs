// Copyright 2024 The Regents of the University of California
// Copyright 2024-2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

#[cfg(feature = "benchmark")]
use criterion::{Criterion, criterion_group, criterion_main};
#[cfg(feature = "benchmark")]
use wellen::check_states_pub;

#[cfg(feature = "benchmark")]
fn criterion_benchmark(c: &mut Criterion) {
    let input = b"xx1010101UUuuHHHh";
    c.bench_function("check_state", |b| b.iter(|| check_states_pub(input)));
}

#[cfg(feature = "benchmark")]
criterion_group!(benches, criterion_benchmark);
#[cfg(feature = "benchmark")]
criterion_main!(benches);
