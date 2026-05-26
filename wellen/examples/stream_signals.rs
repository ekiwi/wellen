// Copyright 2023 The Regents of the University of California
// Copyright 2024-2026 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use clap::{Parser, ValueEnum};
use wellen::*;

#[derive(Parser, Debug)]
#[command(name = "stream_signals")]
#[command(author = "Kevin Laeufer <laeufer@berkeley.edu>")]
#[command(version)]
#[command(about = "Streams signals from a waveform file. Mostly used for benchmarking.", long_about = None)]
struct Args {
    #[arg(value_name = "WAVE", index = 1)]
    filename: String,
    #[arg(long)]
    single_thread: bool,
    #[arg(long, value_enum)]
    mode: Mode,
}

#[derive(Debug, Copy, Clone, Default, ValueEnum)]
enum Mode {
    #[default]
    OnChange,
    TimeStep,
}

fn main() {
    let args = Args::parse();

    let load_opts = LoadOptions {
        multi_thread: !args.single_thread,
        remove_scopes_with_empty_name: false,
    };

    // load header
    let header_start = std::time::Instant::now();
    let mut stream =
        wellen::stream::read_from_file(&args.filename, &load_opts).expect("Failed to load file!");
    let header_load_duration = header_start.elapsed();
    println!(
        "It took {:?} to load the header of {}",
        header_load_duration, args.filename
    );

    let filter = wellen::stream::Filter::all();

    // stream
    let stream_start = std::time::Instant::now();
    match args.mode {
        Mode::OnChange => {
            let mut count = 0u64;

            stream
                .stream_changes(filter, |_time, _signal, _value| {
                    count += 1;
                })
                .expect("stream failed");

            let duration = stream_start.elapsed();
            println!("It took {duration:?} to stream {count} signal changes.");
        }
        Mode::TimeStep => {
            let mut count = 0u64;

            stream
                .stream_time_steps(filter, |_time, _values| {
                    count += 1;
                })
                .expect("stream failed");

            let duration = stream_start.elapsed();
            println!("It took {duration:?} to stream {count} time steps.");
        }
    }
}
