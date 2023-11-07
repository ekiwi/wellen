// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>
//
// Space efficient format for a wavedump hierarchy.

use bytesize::ByteSize;
use std::num::{NonZeroU16, NonZeroU32};
use string_interner::Symbol;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Timescale {
    pub factor: u32,
    pub unit: TimescaleUnit,
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
    Todo, // placeholder tpe
}

#[derive(Debug, Clone, Copy)]
pub enum VarType {
    Wire,
    Todo, // placeholder tpe
}

#[derive(Debug, Clone, Copy)]
pub enum VarDirection {
    Input,
    Todo, // placeholder tpe
}

/// Signal identifier in the waveform (VCD, FST, etc.) file.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub struct SignalRef(NonZeroU32);

impl SignalRef {
    #[inline]
    pub(crate) fn from_index(index: usize) -> Option<Self> {
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

#[derive(Debug, Clone, Copy)]
pub enum SignalLength {
    Variable,
    Fixed(NonZeroU32),
}

impl Default for SignalLength {
    fn default() -> Self {
        SignalLength::Variable
    }
}

impl SignalLength {
    pub fn from_uint(len: u32) -> Self {
        match NonZeroU32::new(len) {
            None => SignalLength::Variable,
            Some(value) => SignalLength::Fixed(value),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Var {
    name: HierarchyStringId,
    tpe: VarType,
    direction: VarDirection,
    length: SignalLength,
    signal_idx: SignalRef,
    parent: ScopeRef,
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
        let mut out = hierarchy.get_scope(self.parent).full_name(hierarchy);
        out.push(SCOPE_SEPARATOR);
        out.push_str(self.name(hierarchy));
        out
    }

    pub fn var_type(&self) -> VarType {
        self.tpe
    }
    pub fn direction(&self) -> VarDirection {
        self.direction
    }
    pub fn signal_idx(&self) -> SignalRef {
        self.signal_idx
    }
    pub fn length(&self) -> SignalLength {
        self.length
    }
    pub fn is_real(&self) -> bool {
        todo!()
    }
    pub fn is_string(&self) -> bool {
        todo!()
    }
    pub fn is_bit_vector(&self) -> bool {
        true
    }
    pub fn is_1bit(&self) -> bool {
        match self.length {
            SignalLength::Fixed(l) => l.get() == 1,
            _ => false,
        }
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
            parent = hierarchy.get_scope(id).parent;
        }
        let mut out: String = String::with_capacity((parents.len() + 1) * 5);
        for parent_id in parents.iter().rev() {
            out.push_str(hierarchy.get_scope(*parent_id).name(hierarchy));
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
                            let new_scope = self.hierarchy.get_scope(scope_id);
                            self.item = Some(HierarchyItem::Scope(new_scope));
                            Some(HierarchyItem::Scope(new_scope))
                        }
                        Some(HierarchyItemId::Var(var_id)) => {
                            let var = self.hierarchy.get_var(var_id);
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
    strings: Vec<String>,
    signal_idx_to_var: Vec<Option<VarRef>>,
}

// public implementation
impl Hierarchy {
    // TODO: add custom iterator that guarantees a traversal in topological order
    pub fn iter_vars(&self) -> std::slice::Iter<'_, Var> {
        self.vars.iter()
    }

    pub fn iter_scopes(&self) -> std::slice::Iter<'_, Scope> {
        self.scopes.iter()
    }

    pub fn items(&self) -> HierarchyItemIterator {
        let start = self.scopes.first().map(|s| HierarchyItem::Scope(s));
        HierarchyItemIterator::new(&self, start)
    }

    pub(crate) fn num_vars(&self) -> usize {
        self.vars.len()
    }

    pub(crate) fn num_unique_signals(&self) -> usize {
        self.signal_idx_to_var.len()
    }

    /// Retrieves the length of a signal identified by its id by looking up a
    /// variable that refers to the signal.
    pub(crate) fn get_signal_length(&self, signal_idx: SignalRef) -> Option<SignalLength> {
        let var_id = (*self.signal_idx_to_var.get(signal_idx.index())?)?;
        Some(self.get_var(var_id).length)
    }

    /// Returns one variable per unique signal in the order of signal handles.
    /// The value will be None if there is no var pointing to the given handle.
    pub fn get_unique_signals_vars(&self) -> Vec<Option<Var>> {
        let mut out = Vec::with_capacity(self.signal_idx_to_var.len());
        for maybe_var_id in self.signal_idx_to_var.iter() {
            if let Some(var_id) = maybe_var_id {
                out.push(Some((*self.get_var(*var_id)).clone()));
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
}

// private implementation
impl Hierarchy {
    fn get_str(&self, id: HierarchyStringId) -> &str {
        &self.strings[id.index()]
    }

    fn get_scope(&self, id: ScopeRef) -> &Scope {
        &self.scopes[id.index()]
    }

    fn get_var(&self, id: VarRef) -> &Var {
        &self.vars[id.index()]
    }

    fn get_item(&self, id: HierarchyItemId) -> HierarchyItem {
        match id {
            HierarchyItemId::Scope(id) => HierarchyItem::Scope(self.get_scope(id)),
            HierarchyItemId::Var(id) => HierarchyItem::Var(self.get_var(id)),
        }
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
    scope_stack: Vec<ScopeStackEntry>,
    strings: StringInterner,
    handle_to_node: Vec<Option<VarRef>>,
    // some statistics
    duplicate_string_count: usize,
    duplicate_string_size: usize,
}

impl Default for HierarchyBuilder {
    fn default() -> Self {
        HierarchyBuilder {
            vars: Vec::default(),
            scopes: Vec::default(),
            scope_stack: Vec::default(),
            strings: StringInterner::default(),
            handle_to_node: Vec::default(),
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
            strings: interner_to_vec(self.strings),
            signal_idx_to_var: self.handle_to_node,
        }
    }

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
        match self.scope_stack.last_mut() {
            Some(entry) => {
                let parent = entry.scope_id;
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
                        // otherwise we need to add a pointer from the parent
                        assert!(self.scopes[parent].child.is_none());
                        self.scopes[parent].child = Some(node_id);
                    }
                }
                // the new node is now the last child
                entry.last_child = Some(node_id);
                // return the parent id
                Some(ScopeRef::from_index(parent).unwrap())
            }
            None => None,
        }
    }

    pub fn add_scope(&mut self, name: String, tpe: ScopeType) {
        let node_id = self.scopes.len();
        let wrapped_id = HierarchyItemId::Scope(ScopeRef::from_index(node_id).unwrap());
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
        length: u32,
        signal_idx: SignalRef,
    ) {
        let node_id = self.vars.len();
        let var_id = VarRef::from_index(node_id).unwrap();
        let wrapped_id = HierarchyItemId::Var(var_id);
        let parent = self.add_to_hierarchy_tree(wrapped_id);
        assert!(
            parent.is_some(),
            "Vars cannot be at the top of the hierarchy (not supported for now)"
        );

        // add lookup
        let handle_idx = signal_idx.index();
        if self.handle_to_node.len() <= handle_idx {
            self.handle_to_node.resize(handle_idx + 1, None);
        }
        self.handle_to_node[handle_idx] = Some(var_id);

        // now we can build the node data structure and store it
        let node = Var {
            parent: parent.unwrap(),
            name: self.add_string(name),
            tpe,
            direction,
            length: SignalLength::from_uint(length),
            signal_idx,
            next: None,
        };
        self.vars.push(node);
    }

    pub fn pop_scope(&mut self) {
        self.scope_stack.pop().unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sizes() {
        // unfortunately this one is pretty big
        assert_eq!(std::mem::size_of::<HierarchyItemId>(), 8);

        // Var
        assert_eq!(
            std::mem::size_of::<Var>(),
            std::mem::size_of::<HierarchyStringId>() // name
                + 1 // tpe
                + 1 // direction
                + 4 // length
                + std::mem::size_of::<SignalRef>() // handle
                + std::mem::size_of::<ScopeRef>() // parent
                + std::mem::size_of::<HierarchyItemId>() // next
        );
        // currently this all comes out to 24 bytes (~= 3x 64-bit pointers)
        assert_eq!(std::mem::size_of::<Var>(), 24);

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
