// Copyright 2023 The Regents of the University of California
// Copyright 2024-2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use bytesize::ByteSize;
use clap::Parser;
use indicatif::ProgressStyle;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use wellen::*;

#[derive(Parser, Debug)]
#[command(name = "loadfst")]
#[command(author = "Kevin Laeufer <laeufer@berkeley.edu>")]
#[command(version)]
#[command(about = "Loads a FST file into a representation suitable for fast access.", long_about = None)]
struct Args {
    #[arg(value_name = "FSTFILE", index = 1)]
    filename: String,
    #[arg(
        long,
        help = "only parse the file, but do not actually load the signals"
    )]
    skip_load: bool,
    #[arg(long)]
    single_thread: bool,
}

fn print_size_of_full_vs_reduced_names(hierarchy: &Hierarchy) {
    let total_num_elements = hierarchy.iter_vars().len() + hierarchy.iter_scopes().len();
    let reduced_size = hierarchy
        .iter_scopes()
        .map(|s| s.name(hierarchy).bytes().len())
        .sum::<usize>()
        + hierarchy
            .iter_vars()
            .map(|v| v.name(hierarchy).bytes().len())
            .sum::<usize>();
    // to compute full names efficiently, we do need to save a 16-bit parent pointer which takes some space
    let _parent_overhead = std::mem::size_of::<u16>() * total_num_elements;
    let full_size = hierarchy
        .iter_scopes()
        .map(|s| s.full_name(hierarchy).bytes().len())
        .sum::<usize>()
        + hierarchy
            .iter_vars()
            .map(|v| v.full_name(hierarchy).bytes().len())
            .sum::<usize>();
    let string_overhead = std::mem::size_of::<String>() * total_num_elements;

    println!("Full vs. partial strings. (Ignoring interning)");
    println!(
        "Saving only the local names uses {}.",
        ByteSize::b((reduced_size + string_overhead) as u64)
    );
    println!(
        "Saving full names would use {}.",
        ByteSize::b((full_size + string_overhead) as u64)
    );
    println!(
        "We saved {}. (actual saving is larger because of interning)",
        ByteSize::b((full_size - reduced_size) as u64)
    )
}

fn main() {
    let args = Args::parse();

    let load_opts = LoadOptions {
        multi_thread: !args.single_thread,
        remove_scopes_with_empty_name: false,
    };

    // load header
    let header_start = std::time::Instant::now();
    let header =
        viewers::read_header_from_file(&args.filename, &load_opts).expect("Failed to load file!");
    let header_load_duration = header_start.elapsed();
    println!(
        "It took {:?} to load the header of {}",
        header_load_duration, args.filename
    );

    // create body progress indicator
    let body_len = header.body_len;
    let (body_progress, progress) = if body_len == 0 {
        (None, None)
    } else {
        let p = Arc::new(AtomicU64::new(0));
        let p_out = p.clone();
        let done = Arc::new(AtomicBool::new(false));
        let done_out = done.clone();
        let ten_millis = std::time::Duration::from_millis(10);
        let t = thread::spawn(move || {
            let bar = indicatif::ProgressBar::new(body_len);
            bar.set_style(
                ProgressStyle::with_template(
                    "[{elapsed_precise}] {bar:40.cyan/blue} {decimal_bytes} ({percent_precise}%)",
                )
                .unwrap(),
            );
            loop {
                // always update
                let new_value = p.load(Ordering::SeqCst);
                bar.set_position(new_value);
                thread::sleep(ten_millis);
                // see if we are done
                let now_done = done.load(Ordering::SeqCst);
                if now_done {
                    if bar.position() != body_len {
                        println!(
                            "WARN: Final progress value was: {}, expected {}",
                            bar.position(),
                            body_len
                        );
                    }
                    bar.finish_and_clear();
                    break;
                }
            }
        });

        (Some(p_out), Some((done_out, t)))
    };

    // load body
    let hierarchy = header.hierarchy;
    let body_start = std::time::Instant::now();
    let body =
        viewers::read_body(header.body, &hierarchy, body_progress).expect("Failed to load body!");
    let body_load_duration = body_start.elapsed();
    println!(
        "It took {:?} to load the body of {}",
        body_load_duration, args.filename
    );
    if let Some((done, t)) = progress {
        done.store(true, Ordering::SeqCst);
        t.join().unwrap();
    }
    let mut wave_source = body.source;

    wave_source.print_statistics();

    println!(
        "The hierarchy takes up at least {} of memory.",
        ByteSize::b(hierarchy.size_in_memory() as u64)
    );
    print_size_of_full_vs_reduced_names(&hierarchy);

    if args.skip_load {
        return;
    }

    // load every signal individually
    let mut signal_load_times = Vec::new();
    let mut signal_sizes = Vec::new();
    let signal_load_start = std::time::Instant::now();
    for var in hierarchy.get_unique_signals_vars().iter().flatten() {
        let _signal_name: String = var.full_name(&hierarchy);
        let ids = [var.signal_ref(); 1];
        let start = std::time::Instant::now();
        let loaded = wave_source.load_signals(&ids, &hierarchy, load_opts.multi_thread);
        let load_time = start.elapsed();
        assert_eq!(loaded.len(), ids.len());
        let (loaded_id, loaded_signal) = loaded.into_iter().next().unwrap();
        assert_eq!(loaded_id, ids[0]);
        let bytes_in_mem = loaded_signal.size_in_memory();
        signal_load_times.push(load_time);
        signal_sizes.push(bytes_in_mem);
    }
    let signal_load_total_duration = signal_load_start.elapsed();
    println!(
        "It took {:?} to load all signals. (and drop them)",
        signal_load_total_duration
    );

    let average_signal_load_time =
        signal_load_times.iter().sum::<std::time::Duration>() / signal_load_times.len() as u32;
    let max_signal_load_time = signal_load_times.iter().max().unwrap();
    let min_signal_load_time = signal_load_times.iter().min().unwrap();
    println!(
        "Loading a signal takes: {:?}..{:?} (avg. {:?})",
        min_signal_load_time, max_signal_load_time, average_signal_load_time
    );

    let total_signal_size = signal_sizes.iter().sum::<usize>();
    let average_signal_size = total_signal_size / signal_sizes.len();
    let max_signal_size = *signal_sizes.iter().max().unwrap();
    let min_signal_size = *signal_sizes.iter().min().unwrap();
    println!(
        "All signals together take up {}",
        ByteSize::b(total_signal_size as u64)
    );
    println!(
        "Signal take up {}..{} (avg. {})",
        ByteSize::b(min_signal_size as u64),
        ByteSize::b(max_signal_size as u64),
        ByteSize::b(average_signal_size as u64)
    )
}
