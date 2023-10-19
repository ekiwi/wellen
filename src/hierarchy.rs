// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use crate::dense::DenseHashMap;
use bytesize::ByteSize;
use std::num::{NonZeroU16, NonZeroU32};
use string_interner::Symbol;

#[derive(Debug, Clone, Copy, PartialEq)]
struct HierarchyVarId(NonZeroU32);

impl HierarchyVarId {
    #[inline]
    fn from_index(index: usize) -> Option<Self> {
        match NonZeroU32::new(index as u32 + 1) {
            None => None,
            Some(value) => Some(HierarchyVarId(value)),
        }
    }

    #[inline]
    fn index(&self) -> usize {
        (self.0.get() - 1) as usize
    }
}

impl Default for HierarchyVarId {
    fn default() -> Self {
        Self::from_index(0).unwrap()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct HierarchyScopeId(NonZeroU16);

impl HierarchyScopeId {
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

impl Default for HierarchyScopeId {
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

pub enum ScopeType {
    Module,
    Todo, // placeholder tpe
}

pub enum VarType {
    Wire,
    Todo, // placeholder tpe
}

pub enum VarDirection {
    Input,
    Todo, // placeholder tpe
}

/// Signal identifier.
pub type SignalHandle = u32;

pub struct Var {
    name: HierarchyStringId,
    tpe: VarType,
    direction: VarDirection,
    length: u32,
    handle: SignalHandle,
    parent: HierarchyScopeId,
    next: Option<HierarchyEntryId>,
}

const SCOPE_SEPARATOR: char = '.';

impl Var {
    /// Local name of the variable.
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
}

// TODO: rename to HierarchyNodeId
#[derive(Clone, Copy)]
enum HierarchyEntryId {
    Scope(HierarchyScopeId),
    Var(HierarchyVarId),
}

pub struct Scope {
    name: HierarchyStringId,
    tpe: ScopeType,
    child: Option<HierarchyEntryId>,
    parent: Option<HierarchyScopeId>,
    next: Option<HierarchyEntryId>,
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
}

pub struct Hierarchy {
    vars: Vec<Var>,
    scopes: Vec<Scope>,
    strings: Vec<String>,
    handle_to_var: Vec<HierarchyVarId>,
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
}

// private implementation
impl Hierarchy {
    fn get_str(&self, id: HierarchyStringId) -> &str {
        &self.strings[id.index()]
    }

    fn get_scope(&self, id: HierarchyScopeId) -> &Scope {
        &self.scopes[id.index()]
    }
}

/// Estimates how much memory the hierarchy uses.
pub fn estimate_hierarchy_size(hierarchy: &Hierarchy) -> usize {
    let var_size = hierarchy.vars.capacity() * std::mem::size_of::<Var>();
    let scope_size = hierarchy.scopes.capacity() * std::mem::size_of::<Scope>();
    let string_size = hierarchy.strings.capacity() * std::mem::size_of::<String>()
        + hierarchy
            .strings
            .iter()
            .map(|s| s.as_bytes().len())
            .sum::<usize>();
    let handle_lookup_size =
        hierarchy.handle_to_var.capacity() * std::mem::size_of::<HierarchyVarId>();
    var_size + scope_size + string_size + handle_lookup_size + std::mem::size_of::<Hierarchy>()
}

struct ScopeStackEntry {
    scope_id: usize,
    last_child: Option<HierarchyEntryId>,
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
    handle_to_node: DenseHashMap<HierarchyVarId>,
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
            handle_to_node: DenseHashMap::default(),
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
            handle_to_var: self.handle_to_node.into_vec(),
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
    fn add_to_hierarchy_tree(&mut self, node_id: HierarchyEntryId) -> Option<HierarchyScopeId> {
        match self.scope_stack.last_mut() {
            Some(entry) => {
                let parent = entry.scope_id;
                match entry.last_child {
                    Some(HierarchyEntryId::Var(child)) => {
                        // add pointer to new node from last child
                        assert!(self.vars[child.index()].next.is_none());
                        self.vars[child.index()].next = Some(node_id);
                    }
                    Some(HierarchyEntryId::Scope(child)) => {
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
                Some(HierarchyScopeId::from_index(parent).unwrap())
            }
            None => None,
        }
    }

    pub fn add_scope(&mut self, name: String, tpe: ScopeType) {
        let node_id = self.scopes.len();
        let wrapped_id = HierarchyEntryId::Scope(HierarchyScopeId::from_index(node_id).unwrap());
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
        handle: SignalHandle,
    ) {
        let node_id = self.vars.len();
        let var_id = HierarchyVarId::from_index(node_id).unwrap();
        let wrapped_id = HierarchyEntryId::Var(var_id);
        let parent = self.add_to_hierarchy_tree(wrapped_id);

        // add lookup
        self.handle_to_node.insert(handle as usize, var_id);

        // now we can build the node data structure and store it
        let node = Var {
            parent: parent.unwrap(),
            name: self.add_string(name),
            tpe,
            direction,
            length,
            handle,
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
        assert_eq!(std::mem::size_of::<HierarchyEntryId>(), 8);

        // Var
        assert_eq!(
            std::mem::size_of::<Var>(),
            std::mem::size_of::<HierarchyStringId>() // name
                + 1 // tpe
                + 1 // direction
                + 4 // length
                + std::mem::size_of::<SignalHandle>() // handle
                + std::mem::size_of::<HierarchyScopeId>() // parent
                + std::mem::size_of::<HierarchyEntryId>() // next
        );
        // currently this all comes out to 24 bytes (~= 3x 64-bit pointers)
        assert_eq!(std::mem::size_of::<Var>(), 24);

        // Scope
        assert_eq!(
            std::mem::size_of::<Scope>(),
            std::mem::size_of::<HierarchyStringId>() // name
                + 1 // tpe
                + std::mem::size_of::<HierarchyEntryId>() // child
                + std::mem::size_of::<HierarchyScopeId>() // parent
                + std::mem::size_of::<HierarchyEntryId>() // next
                + 1 // padding
        );
        // currently this all comes out to 24 bytes (~= 3x 64-bit pointers)
        assert_eq!(std::mem::size_of::<Scope>(), 24);

        // for comparison: one string is 24 bytes for the struct alone (ignoring heap allocation)
        assert_eq!(std::mem::size_of::<String>(), 24);
    }
}
