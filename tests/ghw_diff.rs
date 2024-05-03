// Copyright 2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>
//
// comparing the output of the ghwdump tool and wellen

use std::path::PathBuf;
use std::sync::Once;

static COMPILE_GHW_DUMP: Once = Once::new();

const GHW_DUMP_DIR: &str = "ext/ghw";

fn compile_ghwdump() {
    let stat = std::process::Command::new("gcc")
        .args(["ghwdump.c", "libghw.c", "-o", "ghwdump"])
        .current_dir(GHW_DUMP_DIR)
        .status()
        .expect("failed to compile ghwdump");
    assert!(stat.success(), "failed to compile ghwdump");
}

fn run_ghwdump(cmd: &str, filename: &str) -> String {
    // make sure ghwdump is compiled
    COMPILE_GHW_DUMP.call_once(|| compile_ghwdump());
    let mut ghw_cmd = PathBuf::from(GHW_DUMP_DIR);
    ghw_cmd.push("ghwdump");

    // run ghwdump to collect expected data
    String::from_utf8(
        std::process::Command::new(ghw_cmd)
            .args([cmd, filename])
            .output()
            .expect("failed to run ghwdump")
            .stdout,
    )
    .expect("failed to read output")
}

fn diff_test(filename: &str) {
    // open file with wellen
    let mut waves = wellen::simple::read(filename).expect("failed to parse GHW file");

    // collect expected data
    let signals = run_ghwdump("-s", filename);
    let hierarchy = run_ghwdump("-h", filename);

    // println!("Signals:\n{}\n\nHierarchy:\n{}", signals, hierarchy);
}

#[test]
fn diff_ghdl_oscar_test() {
    diff_test("inputs/ghdl/oscar/test.ghw");
}

/// https://github.com/ekiwi/wellen/issues/12
#[test]
fn diff_wellen_issue_12() {
    diff_test("inputs/ghdl/wellen_issue_12.ghw");
}
