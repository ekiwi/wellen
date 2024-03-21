// Copyright 2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>
//
// contains functions to serialize and de-serialize Hierarchy and Signal
// We write these manually instead of using serde in order to be able to provide some
// backwards compatibility.

use crate::Hierarchy;
use std::io::{Write, Read};

pub fn serialize_hierarchy(hierarchy: &Hierarchy, out: &mut impl Write) -> std::io::Result<()> {


    todo!();
}

pub fn deserialize_hierarchy(input: &mut impl Read) -> std::io::Result<Hierarchy> {


    todo!();
}