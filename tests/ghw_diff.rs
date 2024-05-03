// Copyright 2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>
//
// comparing the output of the ghwdump tool and wellen


use std::sync::Once;

static COMPILE_GHW_DUMP: Once = Once::new();


fn compile_ghwdump() {

}

fn run_ghwdump(filename: &str) {
    // make sure ghwdump is compiled
    COMPILE_GHW_DUMP.call_once(|| compile_ghwdump());
}


#[test]
fn diff_ghdl_oscar_test() {
    run_ghwdump("inputs/ghdl/oscar/test.ghw");
}
