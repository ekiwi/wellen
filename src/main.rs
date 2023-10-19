// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

mod dense;
mod hierarchy;
mod values;

use crate::hierarchy::*;
use crate::values::*;
use bytesize::ByteSize;
use clap::Parser;
use fst_native::{
    FstFilter, FstHierarchyEntry, FstReader, FstScopeType, FstSignalHandle, FstSignalValue,
    FstVarDirection, FstVarType,
};
use std::collections::HashMap;
use std::io::{BufRead, Seek};

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
            FstHierarchyEntry::Scope { tpe, name, .. } => {
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
                h.add_var(
                    name,
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
    let parent_overhead = std::mem::size_of::<u16>() * total_num_elements;
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

fn read_values<F: BufRead + Seek>(reader: &mut FstReader<F>, hierarchy: &Hierarchy) -> Values {
    let mut v = ValueBuilder::default();
    for var in hierarchy.iter_vars() {
        v.add_signal(var.handle(), var.length())
    }

    let cb = |time: u64, handle: FstSignalHandle, value: FstSignalValue| match value {
        FstSignalValue::String(value) => {
            v.value_and_time(handle.get_index() as SignalHandle, time, value.as_bytes());
        }
        FstSignalValue::Real(value) => {
            v.value_and_time(
                handle.get_index() as SignalHandle,
                time,
                &value.to_be_bytes(),
            );
        }
    };
    let filter = FstFilter::all();
    reader.read_signals(&filter, cb).unwrap();
    v.print_statistics();
    v.finish()
}

fn main() {
    let args = Args::parse();
    let input = std::fs::File::open(args.filename).expect("failed to open input file!");
    let mut reader = fst_native::FstReader::open(std::io::BufReader::new(input)).unwrap();
    let hierarchy = read_hierarchy(&mut reader);
    println!(
        "The hierarchy takes up at least {} of memory.",
        ByteSize::b(estimate_hierarchy_size(&hierarchy) as u64)
    );
    print_size_of_full_vs_reduced_names(&hierarchy);
    read_values(&mut reader, &hierarchy);
}
