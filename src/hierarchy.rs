// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>
//
// Space efficient format for a wavedump hierarchy.

use bytesize::ByteSize;
use std::num::{NonZeroU16, NonZeroU32, NonZeroU64};
use string_interner::Symbol;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Timescale {
    pub factor: u32,
    pub unit: TimescaleUnit,
}

impl Timescale {
    pub fn new(factor: u32, unit: TimescaleUnit) -> Self {
        Timescale { factor, unit }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimescaleUnit {
    FemtoSeconds,
    PicoSeconds,
    NanoSeconds,
    MicroSeconds,
    MilliSeconds,
    Seconds,
    Unknown,
}

impl TimescaleUnit {
    pub fn to_exponent(&self) -> Option<i8> {
        match &self {
            TimescaleUnit::FemtoSeconds => Some(-15),
            TimescaleUnit::PicoSeconds => Some(-12),
            TimescaleUnit::NanoSeconds => Some(-9),
            TimescaleUnit::MicroSeconds => Some(-6),
            TimescaleUnit::MilliSeconds => Some(-3),
            TimescaleUnit::Seconds => Some(0),
            TimescaleUnit::Unknown => None,
        }
    }
}

/// Uniquely identifies a variable in the hierarchy.
/// Replaces the old `SignalRef`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarRef(NonZeroU32);

impl VarRef {
    #[inline]
    fn from_index(index: usize) -> Option<Self> {
        match NonZeroU32::new(index as u32 + 1) {
            None => None,
            Some(value) => Some(VarRef(value)),
        }
    }

    #[inline]
    fn index(&self) -> usize {
        (self.0.get() - 1) as usize
    }
}

impl Default for VarRef {
    fn default() -> Self {
        Self::from_index(0).unwrap()
    }
}

/// Uniquely identifies a scope in the hierarchy.
/// Replaces the old `ModuleRef`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScopeRef(NonZeroU16);

impl ScopeRef {
    #[inline]
    fn from_index(index: usize) -> Option<Self> {
        match NonZeroU16::new(index as u16 + 1) {
            None => None,
            Some(value) => Some(Self(value)),
        }
    }

    #[inline]
    fn index(&self) -> usize {
        (self.0.get() - 1) as usize
    }
}

impl Default for ScopeRef {
    fn default() -> Self {
        Self::from_index(0).unwrap()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HierarchyStringId(NonZeroU32);

impl HierarchyStringId {
    #[inline]
    fn from_interner(sym: StringInternerSym) -> Self {
        let value = (sym.to_usize() as u32) + 1;
        HierarchyStringId(NonZeroU32::new(value).unwrap())
    }

    #[inline]
    fn index(&self) -> usize {
        (self.0.get() - 1) as usize
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ScopeType {
    Module,
    Task,
    Function,
    Begin,
    Fork,
}

#[derive(Debug, Clone, Copy)]
pub enum VarType {
    Event,
    Integer,
    Parameter,
    Real,
    Reg,
    Supply0,
    Supply1,
    Time,
    Tri,
    TriAnd,
    TriOr,
    TriReg,
    Tri0,
    Tri1,
    WAnd,
    Wire,
    WOr,
    String,
}

#[derive(Debug, Clone, Copy)]
pub enum VarDirection {
    Input,
    Todo, // placeholder tpe
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct VarIndex(NonZeroU64);

impl VarIndex {
    pub(crate) fn new(msb: i32, lsb: i32) -> Self {
        assert!((lsb as u32) < u32::MAX);
        let value = ((msb as u64) << 32) | (lsb as u64);
        Self(NonZeroU64::new(value + 1).unwrap())
    }

    pub fn msb(&self) -> i32 {
        let value = self.0.get() - 1;
        (value >> 32) as i32
    }

    pub fn lsb(&self) -> i32 {
        let value = self.0.get() - 1;
        (value & (u32::MAX as u64)) as i32
    }
}

/// Signal identifier in the waveform (VCD, FST, etc.) file.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub struct SignalRef(NonZeroU32);

impl SignalRef {
    #[inline]
    pub fn from_index(index: usize) -> Option<Self> {
        match NonZeroU32::new(index as u32 + 1) {
            None => None,
            Some(value) => Some(Self(value)),
        }
    }

    #[inline]
    pub(crate) fn index(&self) -> usize {
        (self.0.get() - 1) as usize
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SignalType {
    String,
    Real,
    BitVector(u32, Option<VarIndex>),
}

impl SignalType {
    pub fn from_uint(len: u32, index: Option<VarIndex>) -> Self {
        SignalType::BitVector(len, index)
    }
}

#[derive(Debug, Clone)]
pub struct Var {
    name: HierarchyStringId,
    var_tpe: VarType,
    direction: VarDirection,
    signal_tpe: SignalType,
    signal_idx: SignalRef,
    parent: Option<ScopeRef>,
    next: Option<HierarchyItemId>,
}

const SCOPE_SEPARATOR: char = '.';

impl Var {
    /// Local name of the variable.
    #[inline]
    pub fn name<'a>(&self, hierarchy: &'a Hierarchy) -> &'a str {
        hierarchy.get_str(self.name)
    }

    /// Full hierarchical name of the variable.
    pub fn full_name<'a>(&self, hierarchy: &Hierarchy) -> String {
        match self.parent {
            None => self.name(hierarchy).to_string(),
            Some(parent) => {
                let mut out = hierarchy.get(parent).full_name(hierarchy);
                out.push(SCOPE_SEPARATOR);
                out.push_str(self.name(hierarchy));
                out
            }
        }
    }

    pub fn var_type(&self) -> VarType {
        self.var_tpe
    }
    pub fn direction(&self) -> VarDirection {
        self.direction
    }
    pub fn index(&self) -> Option<VarIndex> {
        match &self.signal_tpe {
            SignalType::BitVector(_, index) => *index,
            _ => None,
        }
    }
    pub fn signal_idx(&self) -> SignalRef {
        self.signal_idx
    }
    pub fn length(&self) -> Option<u32> {
        match &self.signal_tpe {
            SignalType::String => None,
            SignalType::Real => None,
            SignalType::BitVector(len, _) => Some(*len),
        }
    }
    pub fn is_real(&self) -> bool {
        matches!(self.signal_tpe, SignalType::Real)
    }
    pub fn is_string(&self) -> bool {
        matches!(self.signal_tpe, SignalType::String)
    }
    pub fn is_bit_vector(&self) -> bool {
        matches!(self.signal_tpe, SignalType::BitVector(_, _))
    }
    pub fn is_1bit(&self) -> bool {
        match self.length() {
            Some(l) => l == 1,
            _ => false,
        }
    }
    pub(crate) fn signal_tpe(&self) -> SignalType {
        self.signal_tpe
    }
}

#[derive(Debug, Clone, Copy)]
enum HierarchyItemId {
    Scope(ScopeRef),
    Var(VarRef),
}

#[derive(Debug, Clone, Copy)]
pub enum HierarchyItem<'a> {
    Scope(&'a Scope),
    Var(&'a Var),
}

#[derive(Debug)]
pub struct Scope {
    name: HierarchyStringId,
    tpe: ScopeType,
    child: Option<HierarchyItemId>,
    parent: Option<ScopeRef>,
    next: Option<HierarchyItemId>,
}

impl Scope {
    /// Local name of the scope.
    pub fn name<'a>(&self, hierarchy: &'a Hierarchy) -> &'a str {
        hierarchy.get_str(self.name)
    }

    /// Full hierarchical name of the scope.
    pub fn full_name<'a>(&self, hierarchy: &Hierarchy) -> String {
        let mut parents = Vec::new();
        let mut parent = self.parent;
        while let Some(id) = parent {
            parents.push(id);
            parent = hierarchy.get(id).parent;
        }
        let mut out: String = String::with_capacity((parents.len() + 1) * 5);
        for parent_id in parents.iter().rev() {
            out.push_str(hierarchy.get(*parent_id).name(hierarchy));
            out.push(SCOPE_SEPARATOR)
        }
        out.push_str(self.name(hierarchy));
        out
    }

    pub fn scope_type(&self) -> ScopeType {
        self.tpe
    }

    pub fn items<'a>(&'a self, hierarchy: &'a Hierarchy) -> HierarchyItemIterator<'a> {
        let start = self.child.map(|c| hierarchy.get_item(c));
        HierarchyItemIterator::new(hierarchy, start)
    }

    pub fn vars<'a>(&'a self, hierarchy: &'a Hierarchy) -> HierarchyVarIterator<'a> {
        HierarchyVarIterator {
            underlying: self.items(hierarchy),
        }
    }
}

pub struct HierarchyItemIterator<'a> {
    hierarchy: &'a Hierarchy,
    item: Option<HierarchyItem<'a>>,
    is_first: bool,
}

impl<'a> HierarchyItemIterator<'a> {
    fn new(hierarchy: &'a Hierarchy, item: Option<HierarchyItem<'a>>) -> Self {
        HierarchyItemIterator {
            hierarchy,
            item,
            is_first: true,
        }
    }

    fn get_next(item: HierarchyItem) -> Option<HierarchyItemId> {
        match item {
            HierarchyItem::Scope(scope) => scope.next,
            HierarchyItem::Var(var) => var.next,
        }
    }
}

impl<'a> Iterator for HierarchyItemIterator<'a> {
    type Item = HierarchyItem<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.item {
            None => None, // this iterator is done!
            Some(item) => {
                if self.is_first {
                    self.is_first = false;
                    Some(item)
                } else {
                    match Self::get_next(item) {
                        None => {
                            self.item = None;
                            None
                        }
                        Some(HierarchyItemId::Scope(scope_id)) => {
                            let new_scope = self.hierarchy.get(scope_id);
                            self.item = Some(HierarchyItem::Scope(new_scope));
                            Some(HierarchyItem::Scope(new_scope))
                        }
                        Some(HierarchyItemId::Var(var_id)) => {
                            let var = self.hierarchy.get(var_id);
                            self.item = Some(HierarchyItem::Var(var));
                            Some(HierarchyItem::Var(var))
                        }
                    }
                }
            }
        }
    }
}

pub struct HierarchyVarIterator<'a> {
    underlying: HierarchyItemIterator<'a>,
}

impl<'a> Iterator for HierarchyVarIterator<'a> {
    type Item = &'a Var;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.underlying.next() {
                None => return None,
                Some(HierarchyItem::Var(var)) => return Some(var),
                Some(_) => {} // continue
            }
        }
    }
}

pub struct Hierarchy {
    vars: Vec<Var>,
    scopes: Vec<Scope>,
    first_item: Option<HierarchyItemId>,
    strings: Vec<String>,
    signal_idx_to_var: Vec<Option<VarRef>>,
    meta: HierarchyMetaData,
}

struct HierarchyMetaData {
    timescale: Option<Timescale>,
    date: String,
    version: String,
    comments: Vec<String>,
}

impl Default for HierarchyMetaData {
    fn default() -> Self {
        HierarchyMetaData {
            timescale: None,
            date: "".to_string(),
            version: "".to_string(),
            comments: Vec::default(),
        }
    }
}

// public implementation
impl Hierarchy {
    /// Returns an iterator over all variables (at all levels).
    pub fn iter_vars(&self) -> std::slice::Iter<'_, Var> {
        self.vars.iter()
    }

    /// Returns an iterator over all scopes (at all levels).
    pub fn iter_scopes(&self) -> std::slice::Iter<'_, Scope> {
        self.scopes.iter()
    }

    /// Returns an iterator over all top-level scopes and variables.
    pub fn items(&self) -> HierarchyItemIterator {
        let start = self.first_item.map(|id| self.get_item(id));
        HierarchyItemIterator::new(&self, start)
    }

    /// Returns the first scope that was declared in the underlying file.
    pub fn first_scope(&self) -> Option<&Scope> {
        self.scopes.first()
    }

    /// Returns one variable per unique signal in the order of signal handles.
    /// The value will be None if there is no var pointing to the given handle.
    pub fn get_unique_signals_vars(&self) -> Vec<Option<Var>> {
        let mut out = Vec::with_capacity(self.signal_idx_to_var.len());
        for maybe_var_id in self.signal_idx_to_var.iter() {
            if let Some(var_id) = maybe_var_id {
                out.push(Some((*self.get(*var_id)).clone()));
            } else {
                out.push(None)
            }
        }
        out
    }

    /// Size of the Hierarchy in bytes.
    pub fn size_in_memory(&self) -> usize {
        let var_size = self.vars.capacity() * std::mem::size_of::<Var>();
        let scope_size = self.scopes.capacity() * std::mem::size_of::<Scope>();
        let string_size = self.strings.capacity() * std::mem::size_of::<String>()
            + self
                .strings
                .iter()
                .map(|s| s.as_bytes().len())
                .sum::<usize>();
        let handle_lookup_size = self.signal_idx_to_var.capacity() * std::mem::size_of::<VarRef>();
        var_size + scope_size + string_size + handle_lookup_size + std::mem::size_of::<Hierarchy>()
    }

    pub fn date(&self) -> &str {
        &self.meta.date
    }
    pub fn version(&self) -> &str {
        &self.meta.version
    }
    pub fn timescale(&self) -> Option<Timescale> {
        self.meta.timescale
    }
}

impl Hierarchy {
    pub(crate) fn num_unique_signals(&self) -> usize {
        self.signal_idx_to_var.len()
    }

    /// Retrieves the length of a signal identified by its id by looking up a
    /// variable that refers to the signal.
    pub(crate) fn get_signal_tpe(&self, signal_idx: SignalRef) -> Option<SignalType> {
        let var_id = (*self.signal_idx_to_var.get(signal_idx.index())?)?;
        Some(self.get(var_id).signal_tpe)
    }
}

// private implementation
impl Hierarchy {
    fn get_str(&self, id: HierarchyStringId) -> &str {
        &self.strings[id.index()]
    }

    fn get_item(&self, id: HierarchyItemId) -> HierarchyItem {
        match id {
            HierarchyItemId::Scope(id) => HierarchyItem::Scope(self.get(id)),
            HierarchyItemId::Var(id) => HierarchyItem::Var(self.get(id)),
        }
    }
}

pub trait GetItem<R, I> {
    fn get(&self, id: R) -> &I;
}

impl GetItem<ScopeRef, Scope> for Hierarchy {
    fn get(&self, id: ScopeRef) -> &Scope {
        &self.scopes[id.index()]
    }
}

impl GetItem<VarRef, Var> for Hierarchy {
    fn get(&self, id: VarRef) -> &Var {
        &self.vars[id.index()]
    }
}

struct ScopeStackEntry {
    scope_id: usize,
    last_child: Option<HierarchyItemId>,
}

type StringInternerSym = string_interner::symbol::SymbolU32;
type StringInternerBackend = string_interner::backend::StringBackend<StringInternerSym>;
type StringInterner = string_interner::StringInterner<StringInternerBackend>;

/// we use this to get rid of all the String hashes and safe memory
fn interner_to_vec(interner: StringInterner) -> Vec<String> {
    let mut out = Vec::with_capacity(interner.len());
    for (id, entry) in interner.into_iter() {
        assert_eq!(id.to_usize(), out.len(), "cannot convert to Vec!");
        out.push(entry.to_string());
    }
    out
}

pub struct HierarchyBuilder {
    vars: Vec<Var>,
    scopes: Vec<Scope>,
    first_item: Option<HierarchyItemId>,
    scope_stack: Vec<ScopeStackEntry>,
    strings: StringInterner,
    handle_to_node: Vec<Option<VarRef>>,
    meta: HierarchyMetaData,
    // some statistics
    duplicate_string_count: usize,
    duplicate_string_size: usize,
}

impl Default for HierarchyBuilder {
    fn default() -> Self {
        // we start with a fake entry in the scope stack to keep track of multiple items in the top scope
        let scope_stack = vec![ScopeStackEntry {
            scope_id: usize::MAX,
            last_child: None,
        }];
        HierarchyBuilder {
            vars: Vec::default(),
            scopes: Vec::default(),
            first_item: None,
            scope_stack,
            strings: StringInterner::default(),
            handle_to_node: Vec::default(),
            meta: HierarchyMetaData::default(),
            duplicate_string_count: 0,
            duplicate_string_size: 0,
        }
    }
}

impl HierarchyBuilder {
    pub fn finish(self) -> Hierarchy {
        Hierarchy {
            vars: self.vars,
            scopes: self.scopes,
            first_item: self.first_item,
            strings: interner_to_vec(self.strings),
            signal_idx_to_var: self.handle_to_node,
            meta: self.meta,
        }
    }

    #[allow(dead_code)]
    pub fn print_statistics(&self) {
        println!("Duplicate strings: {}", self.duplicate_string_count);
        println!(
            "Memory saved by interning strings: {}",
            ByteSize::b(self.duplicate_string_size as u64),
        );
    }

    fn add_string(&mut self, value: String) -> HierarchyStringId {
        // collect some statistics
        if self.strings.get(&value).is_some() {
            self.duplicate_string_count += 1;
            self.duplicate_string_size += value.as_bytes().len() + std::mem::size_of::<String>();
        }

        // do the actual interning
        let sym = self.strings.get_or_intern(value);
        HierarchyStringId::from_interner(sym)
    }

    /// adds a variable or scope to the hierarchy tree
    fn add_to_hierarchy_tree(&mut self, node_id: HierarchyItemId) -> Option<ScopeRef> {
        let entry = self.scope_stack.last_mut().unwrap();
        let parent = entry.scope_id;
        let fake_top_scope_parent = parent == usize::MAX;
        match entry.last_child {
            Some(HierarchyItemId::Var(child)) => {
                // add pointer to new node from last child
                assert!(self.vars[child.index()].next.is_none());
                self.vars[child.index()].next = Some(node_id);
            }
            Some(HierarchyItemId::Scope(child)) => {
                // add pointer to new node from last child
                assert!(self.scopes[child.index()].next.is_none());
                self.scopes[child.index()].next = Some(node_id);
            }
            None => {
                if !fake_top_scope_parent {
                    // otherwise we need to add a pointer from the parent
                    assert!(self.scopes[parent].child.is_none());
                    self.scopes[parent].child = Some(node_id);
                }
            }
        }
        // the new node is now the last child
        entry.last_child = Some(node_id);
        // return the parent id if we had a real parent and we aren't at the top scope
        if fake_top_scope_parent {
            None
        } else {
            Some(ScopeRef::from_index(parent).unwrap())
        }
    }

    pub fn add_scope(&mut self, name: String, tpe: ScopeType) {
        let node_id = self.scopes.len();
        let wrapped_id = HierarchyItemId::Scope(ScopeRef::from_index(node_id).unwrap());
        if self.first_item.is_none() {
            self.first_item = Some(wrapped_id);
        }
        let parent = self.add_to_hierarchy_tree(wrapped_id);

        // new active scope
        self.scope_stack.push(ScopeStackEntry {
            scope_id: node_id,
            last_child: None,
        });

        // now we can build the node data structure and store it
        let node = Scope {
            parent,
            child: None,
            next: None,
            name: self.add_string(name),
            tpe,
        };
        self.scopes.push(node);
    }

    pub fn add_var(
        &mut self,
        name: String,
        tpe: VarType,
        direction: VarDirection,
        raw_length: u32,
        index: Option<VarIndex>,
        signal_idx: SignalRef,
    ) {
        let node_id = self.vars.len();
        let var_id = VarRef::from_index(node_id).unwrap();
        let wrapped_id = HierarchyItemId::Var(var_id);
        if self.first_item.is_none() {
            self.first_item = Some(wrapped_id);
        }
        let parent = self.add_to_hierarchy_tree(wrapped_id);

        // add lookup
        let handle_idx = signal_idx.index();
        if self.handle_to_node.len() <= handle_idx {
            self.handle_to_node.resize(handle_idx + 1, None);
        }
        self.handle_to_node[handle_idx] = Some(var_id);

        // for strings, the length is always flexible
        let signal_tpe = match tpe {
            VarType::String => SignalType::String,
            VarType::Real => SignalType::Real,
            _ => SignalType::from_uint(raw_length, index),
        };

        // now we can build the node data structure and store it
        let node = Var {
            parent,
            name: self.add_string(name),
            var_tpe: tpe,
            direction,
            signal_tpe,
            signal_idx,
            next: None,
        };
        self.vars.push(node);
    }

    pub fn pop_scope(&mut self) {
        self.scope_stack.pop().unwrap();
    }

    pub fn set_date(&mut self, value: String) {
        assert!(
            self.meta.date.is_empty(),
            "Duplicate dates: {} vs {}",
            self.meta.date,
            value
        );
        self.meta.date = value;
    }

    pub fn set_version(&mut self, value: String) {
        assert!(
            self.meta.version.is_empty(),
            "Duplicate versions: {} vs {}",
            self.meta.version,
            value
        );
        self.meta.version = value;
    }

    pub fn set_timescale(&mut self, value: Timescale) {
        assert!(
            self.meta.timescale.is_none(),
            "Duplicate timescales: {:?} vs {:?}",
            self.meta.timescale.unwrap(),
            value
        );
        self.meta.timescale = Some(value);
    }

    pub fn add_comment(&mut self, comment: String) {
        self.meta.comments.push(comment);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sizes() {
        // unfortunately this one is pretty big
        assert_eq!(std::mem::size_of::<HierarchyItemId>(), 8);

        // 4 byte length, 8 byte index + tag + padding
        assert_eq!(std::mem::size_of::<SignalType>(), 16);

        // Var
        assert_eq!(
            std::mem::size_of::<Var>(),
            std::mem::size_of::<HierarchyStringId>() // name
                + 1 // tpe
                + 1 // direction
                + 16 // signal tpe
                + std::mem::size_of::<SignalRef>() // handle
                + std::mem::size_of::<ScopeRef>() // parent
                + std::mem::size_of::<HierarchyItemId>() // next
                + 4 // padding
        );
        // currently this all comes out to 40 bytes (~= 5x 64-bit pointers)
        assert_eq!(std::mem::size_of::<Var>(), 40);

        // Scope
        assert_eq!(
            std::mem::size_of::<Scope>(),
            std::mem::size_of::<HierarchyStringId>() // name
                + 1 // tpe
                + std::mem::size_of::<HierarchyItemId>() // child
                + std::mem::size_of::<ScopeRef>() // parent
                + std::mem::size_of::<HierarchyItemId>() // next
                + 1 // padding
        );
        // currently this all comes out to 24 bytes (~= 3x 64-bit pointers)
        assert_eq!(std::mem::size_of::<Scope>(), 24);

        // for comparison: one string is 24 bytes for the struct alone (ignoring heap allocation)
        assert_eq!(std::mem::size_of::<String>(), 24);
    }
}
