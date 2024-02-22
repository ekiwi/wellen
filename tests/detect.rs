// Copyright 2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use std::io::BufReader;
use std::path::{Path, PathBuf};
use wellen::{detect_file_format, FileFormat};

fn find_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap().filter_map(Result::ok) {
        let entry_path = entry.path();
        if entry_path.is_dir() {
            let mut sub = find_files(&entry_path);
            out.append(&mut sub);
        } else if entry_path.is_file() {
            out.push(entry_path);
        }
    }
    out.sort();
    out
}

#[test]
fn test_detect_file_format() {
    let files = find_files(Path::new("inputs/"));
    assert!(files.len() > 10);
    for filename in files {
        let f = std::fs::File::open(filename.clone())
            .unwrap_or_else(|_| panic!("Failed to open {:?}", filename));
        let mut reader = BufReader::new(f);
        let format = detect_file_format(&mut reader);
        let filename_str = filename.to_str().unwrap();
        match format {
            FileFormat::Vcd => {
                assert!(filename_str.ends_with(".vcd"), "{filename_str}");
            }
            FileFormat::Fst => {
                assert!(filename_str.ends_with(".fst"), "{filename_str}");
            }
            FileFormat::Ghw => {
                assert!(filename_str.ends_with(".ghw"), "{filename_str}");
            }
            FileFormat::Unknown => {
                // this file ends in fst, but does not seem to be a valid fst
                let ignore = filename_str.ends_with("libsigrok.vcd.fst");
                if !ignore {
                    assert!(!filename_str.ends_with(".vcd"), "{filename_str}");
                    assert!(!filename_str.ends_with(".fst"), "{filename_str}");
                    assert!(!filename_str.ends_with(".ghw"), "{filename_str}");
                }
            }
        }
    }
}
