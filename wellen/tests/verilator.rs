// Copyright 2026 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>
//
// test Verilator specific behaviors

use wellen::simple::read;
use wellen::{Hierarchy, Scope, Var, VarType};

#[test]
fn test_new_verilator_fst_attributes() {
    let filename = "inputs/verilator/verilator-pull-7255-t_trace_complex_structs_cc_fst.fst";
    let waves = read(filename).expect("failed to parse");
    let h = waves.hierarchy();

    // v_arrp : arrp_t
    let v_arrp = &h[h.lookup_var(&[&"top", &"t"], &"v_arrp").unwrap()];
    check_arrp_t_var(h, v_arrp);

    // v_arrp_arrp : arrp_arrp_t
    let v_arrp_arrp = &h[h.lookup_scope(&[&"top", &"t", &"v_arrp_arrp"]).unwrap()];
    check_arrp_arrp_t_scope(h, v_arrp_arrp);

    // v_arru : arru_t
    let v_arru = &h[h.lookup_scope(&[&"top", &"t", &"v_arru"]).unwrap()];
    check_arru_t_scope(h, v_arru);

    // v_arru_arru : arru_arru_t
    let v_arru_arru = &h[h.lookup_scope(&[&"top", &"t", &"v_arru_arru"]).unwrap()];
    check_arru_arru_t_scope(h, v_arru_arru);

    // v_arru_arrp : arru_arrp_t
    let v_arru_arrp = &h[h.lookup_scope(&[&"top", &"t", &"v_arru_arrp"]).unwrap()];
    check_arru_arrp_t_scope(h, v_arru_arrp);
}

/// A packed array of bits: `typedef bit [2:1] arrp_t;`
fn check_arrp_t_var(_h: &Hierarchy, var: &Var) {
    let index = var.index().expect("should have an index");
    assert_eq!(index.msb(), 2);
    assert_eq!(index.lsb(), 1);
    assert_eq!(var.var_type(), VarType::Bit);
}

/// A packed array of a packed array: `typedef arrp_t [4:3] arrp_arrp_t;`
fn check_arrp_arrp_t_scope(h: &Hierarchy, scope: &Scope) {
    assert!(scope.is_packed_array());
    assert_no_scopes(h, scope);
    let elements = child_vars(h, scope);
    assert_eq!(elements.len(), 2);
    assert_eq!(elements[0].name(h), "[4]");
    check_arrp_t_var(h, elements[0]);
    assert_eq!(elements[1].name(h), "[3]");
    check_arrp_t_var(h, elements[1]);
}

/// An unpacked array of a packed array: `typedef arrp_t arru_arrp_t[4:3];`
fn check_arru_arrp_t_scope(h: &Hierarchy, scope: &Scope) {
    assert!(scope.is_unpacked_array());
    assert_no_scopes(h, scope);
    let elements = child_vars(h, scope);
    assert_eq!(elements.len(), 2);
    assert_eq!(elements[0].name(h), "[4]");
    check_arrp_t_var(h, elements[0]);
    assert_eq!(elements[1].name(h), "[3]");
    check_arrp_t_var(h, elements[1]);
}

/// An unpacked array of an unpacked array: `typedef arru_t arru_arru_t[4:3];`
fn check_arru_arru_t_scope(h: &Hierarchy, scope: &Scope) {
    assert!(scope.is_unpacked_array());
    assert_no_vars(h, scope);
    let elements = child_scopes(h, scope);
    assert_eq!(elements.len(), 2);
    assert_eq!(elements[0].name(h), "[4]");
    check_arru_t_scope(h, elements[0]);
    assert_eq!(elements[1].name(h), "[3]");
    check_arru_t_scope(h, elements[1]);
}

/// Unpacked bit array: `typedef bit arru_t[2:1];`
fn check_arru_t_scope(h: &Hierarchy, scope: &Scope) {
    assert!(scope.is_unpacked_array());
    assert_no_scopes(h, scope);
    let elements = child_vars(h, scope);
    assert_eq!(
        elements.len(),
        2,
        "entries of unpacked arrays should never be merged!"
    );
    assert_eq!(elements[0].name(h), "[2]");
    assert_eq!(elements[0].length(h).unwrap(), 1);
    assert_eq!(elements[0].var_type(), VarType::Bit);
    assert_eq!(elements[1].name(h), "[1]");
    assert_eq!(elements[1].length(h).unwrap(), 1);
    assert_eq!(elements[1].var_type(), VarType::Bit);
}

fn child_scopes<'a>(h: &'a Hierarchy, scope: &Scope) -> Vec<&'a Scope> {
    scope.scopes(h).map(|v| &h[v]).collect()
}

fn child_vars<'a>(h: &'a Hierarchy, scope: &Scope) -> Vec<&'a Var> {
    scope.vars(h).map(|v| &h[v]).collect()
}

fn assert_no_scopes(h: &Hierarchy, scope: &Scope) {
    let child_scopes: Vec<_> = scope.scopes(h).map(|v| h[v].name(h)).collect();
    assert!(
        child_scopes.is_empty(),
        "unexpected child scopes: {:?}",
        child_scopes
    );
}

fn assert_no_vars(h: &Hierarchy, scope: &Scope) {
    let child_vars: Vec<_> = scope.vars(h).map(|v| h[v].name(h)).collect();
    assert!(
        child_vars.is_empty(),
        "unexpected child vars: {:?}",
        child_vars
    );
}
