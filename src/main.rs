// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

mod dense;
mod hierarchy;

use crate::hierarchy::*;
use clap::Parser;
use fst_native::{FstHierarchyEntry, FstReader, FstScopeType, FstVarDirection, FstVarType};
use std::collections::HashMap;
use std::io::{BufRead, Read, Seek};

#[derive(Parser, Debug)]
#[command(name = "loadfst")]
#[command(author = "Kevin Laeufer <laeufer@berkeley.edu>")]
#[command(version)]
#[command(about = "Loads a FST file into a representation suitable for fast access.", long_about = None)]
struct Args {
    #[arg(value_name = "FSTFILE", index = 1)]
    filename: String,
}

fn convert_scope_tpe(tpe: FstScopeType) -> ScopeType {
    match tpe {
        FstScopeType::Module => ScopeType::Module,
        _ => ScopeType::Todo,
    }
}

fn convert_var_tpe(tpe: FstVarType) -> VarType {
    match tpe {
        FstVarType::Wire => VarType::Wire,
        _ => VarType::Todo,
    }
}

fn convert_var_direction(tpe: FstVarDirection) -> VarDirection {
    match tpe {
        FstVarDirection::Input => VarDirection::Input,
        _ => VarDirection::Todo,
    }
}

fn read_hierarchy<F: BufRead + Seek>(reader: &mut FstReader<F>) -> Hierarchy {
    let mut h = HierarchyBuilder::default();
    let mut path_names = HashMap::new();

    let cb = |entry: FstHierarchyEntry| {
        match entry {
            FstHierarchyEntry::Scope {
                tpe,
                name,
                ..
            } => {
                h.add_scope(name, convert_scope_tpe(tpe));
            }
            FstHierarchyEntry::UpScope => h.pop_scope(),
            FstHierarchyEntry::Var {
                tpe,
                direction,
                name,
                length,
                handle,
                ..
            } => {
                h.add_var(name,
                    convert_var_tpe(tpe),
                    convert_var_direction(direction),
                    length,
                    handle.get_index() as u32,
                );
            }
            FstHierarchyEntry::PathName { id, name } => {
                path_names.insert(id, name);
            }
            FstHierarchyEntry::SourceStem { .. } => todo!(),
            FstHierarchyEntry::Comment { .. } => todo!(),
            FstHierarchyEntry::EnumTable { .. } => todo!(),
            FstHierarchyEntry::EnumTableRef { .. } => todo!(),
            FstHierarchyEntry::AttributeEnd => todo!(),
        };
    };
    reader.read_hierarchy(cb).unwrap();
    h.print_statistics();
    h.finish()
}

fn main() {
    let args = Args::parse();
    let input = std::fs::File::open(args.filename).expect("failed to open input file!");
    let mut reader = fst_native::FstReader::open(std::io::BufReader::new(input)).unwrap();
    let hierarchy = read_hierarchy(&mut reader);
    println!("The hierarchy takes up at least {} bytes of memory.", estimate_hierarchy_size(&hierarchy));
}
