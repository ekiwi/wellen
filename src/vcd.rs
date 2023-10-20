// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use crate::hierarchy::*;
use crate::values::*;
use std::io::BufRead;
use vcd::Command;

pub fn read(filename: &str) -> (Hierarchy, Values) {
    let input = std::fs::File::open(filename).expect("failed to open input file!");
    let mut parser = vcd::Parser::new(std::io::BufReader::new(input));
    let hierarchy = read_hierarchy(&mut parser);
    let values = read_values(&mut parser, &hierarchy);
    (hierarchy, values)
}

fn read_hierarchy<R: BufRead>(parser: &mut vcd::Parser<R>) -> Hierarchy {
    let header = parser.parse_header().unwrap();
    let mut h = HierarchyBuilder::default();
    for item in header.items {
        add_item(item, &mut h);
    }
    h.print_statistics();
    h.finish()
}

fn convert_scope_tpe(tpe: vcd::ScopeType) -> ScopeType {
    match tpe {
        vcd::ScopeType::Module => ScopeType::Module,
        _ => ScopeType::Todo,
    }
}

fn convert_var_tpe(tpe: vcd::VarType) -> VarType {
    match tpe {
        vcd::VarType::Wire => VarType::Wire,
        _ => VarType::Todo,
    }
}

fn add_item(item: vcd::ScopeItem, h: &mut HierarchyBuilder) {
    match item {
        vcd::ScopeItem::Scope(scope) => {
            h.add_scope(scope.identifier, convert_scope_tpe(scope.scope_type));
            for item in scope.items {
                add_item(item, h);
            }
            h.pop_scope();
        }
        vcd::ScopeItem::Var(var) => {
            h.add_var(
                var.reference,
                convert_var_tpe(var.var_type),
                VarDirection::Todo,
                var.size,
                0, // problem: we cannot get to the actual ID because it is private!
            );
        }
        vcd::ScopeItem::Comment(_) => {} // ignore comments
        _ => {}
    }
}

fn read_values<R: BufRead>(parser: &mut vcd::Parser<R>, hierarchy: &Hierarchy) -> Values {
    let mut v = ValueBuilder::default();
    for var in hierarchy.iter_vars() {
        v.add_signal(var.handle(), var.length())
    }

    for command_result in parser {
        let command = command_result?;
        use vcd::Command::*;
        match command {
            ChangeScalar(i, v) => {
                todo!()
            }

            _ => {}
        }
    }
    v.print_statistics();
    v.finish()
}
