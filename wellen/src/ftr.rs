// Copyright 2025 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use crate::hierarchy::{HierarchyBuilder, HierarchyStringId};
use crate::{FileFormat, Hierarchy, LoadOptions, ScopeType, VarType};
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::io::{BufRead, Seek};

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FtrError {
    #[error("[ftr] {0}")]
    Parsing(String),
}

pub type Result<T> = std::result::Result<T, FtrError>;

/// Checks header to see if we are dealing with a FTR file.
pub fn is_ftr(input: &mut (impl BufRead + Seek)) -> bool {
    ftr_parser::parse::is_ftr(input)
}

pub fn read_header_from_file<P: AsRef<std::path::Path>>(
    filename: P,
    options: &LoadOptions,
) -> Result<(
    Hierarchy,
    ReadBodyContinuation<std::io::BufReader<std::fs::File>>,
    u64,
)> {
    let r = match ftr_parser::parse::parse_ftr(filename.as_ref().to_path_buf()) {
        Ok(r) => r,
        Err(e) => return Err(FtrError::Parsing(e.to_string())),
    };

    let mut hb = HierarchyBuilder::new(FileFormat::Ftr);

    // create string to hierarchy string id mapping
    let to_str_id = FxHashMap::from_iter(r.str_dict.iter().flat_map(|(_, s)| {
        // we split onm `.` since our hierarchy never saves the full name
        let inner: Vec<_> = s
            .split('.')
            .map(|name| {
                let name = name.to_string();
                let id = hb.add_string(name.clone());
                (name, id)
            })
            .collect();
        inner
    }));

    // streams are like scopes
    let mut stream_ids: Vec<_> = r.tx_streams.keys().collect();
    stream_ids.sort();
    for stream_id in stream_ids {
        add_stream(
            &mut hb,
            &to_str_id,
            &r.tx_generators,
            &r.tx_streams[stream_id],
        );
    }

    println!("GENERATORS");
    for (id, gen) in r.tx_generators.iter() {
        println!("{id}: {:?}", gen);
    }

    println!("STREAMS");
    for (id, stream) in r.tx_streams.iter() {
        println!("{id}: {:?}", stream);
    }

    println!("STRINGS");
    for (id, name) in r.str_dict.iter() {
        println!("{id}: {}", name);
    }

    todo!();
    // let cont = ReadBodyContinuation {
    //     input,
    // };
}

fn add_stream(
    hb: &mut HierarchyBuilder,
    to_str_id: &FxHashMap<String, HierarchyStringId>,
    generators: &HashMap<usize, ftr_parser::types::TxGenerator>,
    stream: &ftr_parser::types::TxStream,
) {
    // split up the name and create scope hierarchy
    let parts: Vec<_> = stream.name.split('.').map(|s| to_str_id[s]).collect();
    for (ii, &name) in parts.iter().enumerate() {
        let is_last = ii == parts.len() - 1;
        let component = if is_last {
            Some(to_str_id[&stream.kind])
        } else {
            None
        };
        hb.add_scope(name, component, ScopeType::Stream, None, None, false);
    }

    // TODO: how do we treat the generators?

    // pop all scopes
    hb.pop_scopes(parts.len());
}

pub struct ReadBodyContinuation<R: BufRead + Seek> {
    input: R,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_test() {
        read_header_from_file("inputs/ftr/my_db.ftr", &LoadOptions::default()).unwrap();
    }
}
