// Copyright 2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use waveform::check_states_pub;

fn criterion_benchmark(c: &mut Criterion) {
    let input = b"xx1010101UUuuHHHh";
    c.bench_function("check_state", |b| b.iter(|| check_states_pub(input)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
