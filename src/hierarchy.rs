// Copyright 2023-2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>
//
// Space efficient format for a wavedump hierarchy.

use crate::FileFormat;
use std::num::{NonZeroU16, NonZeroU32, NonZeroU64};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
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
pub(crate) struct HierarchyStringId(NonZeroU32);

impl HierarchyStringId {
    #[inline]
    fn from_index(index: usize) -> Self {
        let value = (index + 1) as u32;
        HierarchyStringId(NonZeroU32::new(value).unwrap())
    }

    #[inline]
    fn index(&self) -> usize {
        (self.0.get() - 1) as usize
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ScopeType {
    // VCD Scope Types
    Module,
    Task,
    Function,
    Begin,
    Fork,
    Generate,
    Struct,
    Union,
    Class,
    Interface,
    Package,
    Program,
    // VHDL
    VhdlArchitecture,
    VhdlProcedure,
    VhdlFunction,
    VhdlRecord,
    VhdlProcess,
    VhdlBlock,
    VhdlForGenerate,
    VhdlIfGenerate,
    VhdlGenerate,
    VhdlPackage,
    // from GHW
    GhwGeneric,
    VhdlArray,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarType {
    // VCD
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
    Port,
    SparseArray,
    RealTime,
    // System Verilog
    Bit,
    Logic,
    Int,
    ShortInt,
    LongInt,
    Byte,
    Enum,
    ShortReal,
    // VHDL (these are the types emitted by GHDL)
    Boolean,
    BitVector,
    StdLogic,
    StdLogicVector,
    StdULogic,
    StdULogicVector,
}

/// Signal directions of a variable. Currently these have the exact same meaning as in the FST format.
/// For VCD inputs, all variables will be marked as `VarDirection::Unknown` since no direction information is included.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum VarDirection {
    Unknown,
    Implicit,
    Input,
    Output,
    InOut,
    Buffer,
    Linkage,
}

impl VarDirection {
    pub(crate) fn vcd_default() -> Self {
        VarDirection::Unknown
    }
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
    BitVector(NonZeroU32, Option<VarIndex>),
}

impl SignalType {
    pub fn from_uint(len: u32, index: Option<VarIndex>) -> Self {
        match NonZeroU32::new(len) {
            // a zero length signal should be represented as a 1-bit signal
            None => SignalType::BitVector(NonZeroU32::new(1).unwrap(), index),
            Some(value) => SignalType::BitVector(value, index),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Var {
    name: HierarchyStringId,
    var_tpe: VarType,
    direction: VarDirection,
    signal_tpe: SignalType,
    signal_idx: SignalRef,
    enum_type: Option<EnumTypeId>,
    vhdl_type_name: Option<HierarchyStringId>,
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
    pub fn enum_type<'a>(
        &self,
        hierarchy: &'a Hierarchy,
    ) -> Option<(&'a str, Vec<(&'a str, &'a str)>)> {
        self.enum_type.map(|id| hierarchy.get_enum_type(id))
    }

    #[inline]
    pub fn vhdl_type_name<'a>(&self, hierarchy: &'a Hierarchy) -> Option<&'a str> {
        self.vhdl_type_name.map(|i| hierarchy.get_str(i))
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
    pub fn signal_ref(&self) -> SignalRef {
        self.signal_idx
    }
    pub fn length(&self) -> Option<u32> {
        match &self.signal_tpe {
            SignalType::String => None,
            SignalType::Real => None,
            SignalType::BitVector(len, _) => Some(len.get()),
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
    /// Some wave formats supply the name of the component, e.g., of the module that was instantiated.
    component: Option<HierarchyStringId>,
    tpe: ScopeType,
    declaration_source: Option<SourceLocId>,
    instance_source: Option<SourceLocId>,
    child: Option<HierarchyItemId>,
    parent: Option<ScopeRef>,
    next: Option<HierarchyItemId>,
}

impl Scope {
    /// Local name of the scope.
    pub fn name<'a>(&self, hierarchy: &'a Hierarchy) -> &'a str {
        hierarchy.get_str(self.name)
    }

    /// Local name of the component, e.g., the name of the module that was instantiated.
    pub fn component<'a>(&self, hierarchy: &'a Hierarchy) -> Option<&'a str> {
        self.component.map(|n| hierarchy.get_str(n))
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

    pub fn source_loc<'a>(&self, hierarchy: &'a Hierarchy) -> Option<(&'a str, u64)> {
        self.declaration_source
            .map(|id| hierarchy.get_source_loc(id))
    }

    pub fn instantiation_source_loc<'a>(&self, hierarchy: &'a Hierarchy) -> Option<(&'a str, u64)> {
        self.instance_source.map(|id| hierarchy.get_source_loc(id))
    }

    pub fn items<'a>(&'a self, hierarchy: &'a Hierarchy) -> HierarchyItemIterator<'a> {
        HierarchyItemIterator::new(hierarchy, self.child)
    }

    pub fn vars<'a>(&'a self, hierarchy: &'a Hierarchy) -> HierarchyVarRefIterator<'a> {
        HierarchyVarRefIterator {
            underlying: HierarchyItemIdIterator::new(hierarchy, self.child),
        }
    }

    pub fn scopes<'a>(&'a self, hierarchy: &'a Hierarchy) -> HierarchyScopeRefIterator<'a> {
        HierarchyScopeRefIterator {
            underlying: HierarchyItemIdIterator::new(hierarchy, self.child),
        }
    }
}

struct HierarchyItemIdIterator<'a> {
    hierarchy: &'a Hierarchy,
    item: Option<HierarchyItemId>,
    is_first: bool,
}

impl<'a> HierarchyItemIdIterator<'a> {
    fn new(hierarchy: &'a Hierarchy, item: Option<HierarchyItemId>) -> Self {
        Self {
            hierarchy,
            item,
            is_first: true,
        }
    }

    fn get_next(&self, item: HierarchyItemId) -> Option<HierarchyItemId> {
        match self.hierarchy.get_item(item) {
            HierarchyItem::Scope(scope) => scope.next,
            HierarchyItem::Var(var) => var.next,
        }
    }
}

impl<'a> Iterator for HierarchyItemIdIterator<'a> {
    type Item = HierarchyItemId;

    fn next(&mut self) -> Option<Self::Item> {
        match self.item {
            None => None, // this iterator is done!
            Some(item) => {
                if self.is_first {
                    self.is_first = false;
                    Some(item)
                } else {
                    self.item = self.get_next(item);
                    self.item
                }
            }
        }
    }
}

pub struct HierarchyItemIterator<'a> {
    inner: HierarchyItemIdIterator<'a>,
}

impl<'a> HierarchyItemIterator<'a> {
    fn new(hierarchy: &'a Hierarchy, item: Option<HierarchyItemId>) -> Self {
        Self {
            inner: HierarchyItemIdIterator::new(hierarchy, item),
        }
    }
}

impl<'a> Iterator for HierarchyItemIterator<'a> {
    type Item = HierarchyItem<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|item_id| self.inner.hierarchy.get_item(item_id))
    }
}

pub struct HierarchyVarRefIterator<'a> {
    underlying: HierarchyItemIdIterator<'a>,
}

impl<'a> Iterator for HierarchyVarRefIterator<'a> {
    type Item = VarRef;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.underlying.next() {
                None => return None,
                Some(HierarchyItemId::Var(var)) => return Some(var),
                Some(_) => {} // continue
            }
        }
    }
}

pub struct HierarchyScopeRefIterator<'a> {
    underlying: HierarchyItemIdIterator<'a>,
}

impl<'a> Iterator for HierarchyScopeRefIterator<'a> {
    type Item = ScopeRef;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.underlying.next() {
                None => return None,
                Some(HierarchyItemId::Scope(scope)) => return Some(scope),
                Some(_) => {} // continue
            }
        }
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub(crate) struct SourceLocId(NonZeroU16);

impl SourceLocId {
    #[inline]
    fn from_index(index: usize) -> Self {
        let value = (index + 1) as u16;
        SourceLocId(NonZeroU16::new(value).unwrap())
    }

    #[inline]
    fn index(&self) -> usize {
        (self.0.get() - 1) as usize
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
struct SourceLoc {
    path: HierarchyStringId,
    line: u64,
    is_instantiation: bool,
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub(crate) struct EnumTypeId(NonZeroU16);

impl EnumTypeId {
    #[inline]
    fn from_index(index: usize) -> Self {
        let value = (index + 1) as u16;
        EnumTypeId(NonZeroU16::new(value).unwrap())
    }

    #[inline]
    fn index(&self) -> usize {
        (self.0.get() - 1) as usize
    }
}

#[derive(Debug, PartialEq, Clone)]
struct EnumType {
    name: HierarchyStringId,
    mapping: Vec<(HierarchyStringId, HierarchyStringId)>,
}

pub struct Hierarchy {
    vars: Vec<Var>,
    scopes: Vec<Scope>,
    first_item: Option<HierarchyItemId>,
    strings: Vec<String>,
    source_locs: Vec<SourceLoc>,
    enums: Vec<EnumType>,
    signal_idx_to_var: Vec<Option<VarRef>>,
    meta: HierarchyMetaData,
}

struct HierarchyMetaData {
    timescale: Option<Timescale>,
    date: String,
    version: String,
    comments: Vec<String>,
    file_format: FileFormat,
}

impl HierarchyMetaData {
    fn new(file_format: FileFormat) -> Self {
        HierarchyMetaData {
            timescale: None,
            date: "".to_string(),
            version: "".to_string(),
            comments: Vec::default(),
            file_format,
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
        HierarchyItemIterator::new(&self, self.first_item)
    }

    /// Returns an iterator over references to all top-level scopes.
    pub fn scopes(&self) -> HierarchyScopeRefIterator {
        HierarchyScopeRefIterator {
            underlying: HierarchyItemIdIterator::new(&self, self.first_item),
        }
    }

    /// Returns an iterator over references to all top-level variables.
    pub fn vars(&self) -> HierarchyVarRefIterator {
        HierarchyVarRefIterator {
            underlying: HierarchyItemIdIterator::new(&self, self.first_item),
        }
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
    pub fn file_format(&self) -> FileFormat {
        self.meta.file_format
    }

    pub fn lookup_scope<N: AsRef<str>>(&self, names: &[N]) -> Option<ScopeRef> {
        let prefix = names.first()?.as_ref();
        let mut scope = self.scopes().find(|s| self.get(*s).name(&self) == prefix)?;
        for name in names.iter().skip(1) {
            scope = self
                .get(scope)
                .scopes(&self)
                .find(|s| self.get(*s).name(&self) == name.as_ref())?;
        }
        Some(scope)
    }

    pub fn lookup_var<N: AsRef<str>>(&self, path: &[N], name: &N) -> Option<VarRef> {
        match path {
            [] => self
                .vars()
                .find(|v| self.get(*v).name(&self) == name.as_ref()),
            scopes => {
                let scope = self.get(self.lookup_scope(scopes)?);
                scope
                    .vars(&self)
                    .find(|v| self.get(*v).name(&self) == name.as_ref())
            }
        }
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

    fn get_source_loc(&self, id: SourceLocId) -> (&str, u64) {
        let loc = &self.source_locs[id.index()];
        (self.get_str(loc.path), loc.line)
    }

    fn get_enum_type(&self, id: EnumTypeId) -> (&str, Vec<(&str, &str)>) {
        let enum_tpe = &self.enums[id.index()];
        let name = self.get_str(enum_tpe.name);
        let mapping = enum_tpe
            .mapping
            .iter()
            .map(|(a, b)| (self.get_str(*a), self.get_str(*b)))
            .collect::<Vec<_>>();
        (name, mapping)
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
    /// indicates that this scope is being flattened and all operations should be done on the parent instead
    flattened: bool,
}

pub struct HierarchyBuilder {
    vars: Vec<Var>,
    scopes: Vec<Scope>,
    first_item: Option<HierarchyItemId>,
    scope_stack: Vec<ScopeStackEntry>,
    strings: Vec<String>,
    source_locs: Vec<SourceLoc>,
    enums: Vec<EnumType>,
    handle_to_node: Vec<Option<VarRef>>,
    meta: HierarchyMetaData,
}

const EMPTY_STRING: HierarchyStringId = HierarchyStringId(unsafe { NonZeroU32::new_unchecked(1) });

impl HierarchyBuilder {
    pub(crate) fn new(file_type: FileFormat) -> Self {
        // we start with a fake entry in the scope stack to keep track of multiple items in the top scope
        let scope_stack = vec![ScopeStackEntry {
            scope_id: usize::MAX,
            last_child: None,
            flattened: false,
        }];
        HierarchyBuilder {
            vars: Vec::default(),
            scopes: Vec::default(),
            first_item: None,
            scope_stack,
            strings: vec!["".to_string()], // string 0 is ""
            source_locs: Vec::default(),
            enums: Vec::default(),
            handle_to_node: Vec::default(),
            meta: HierarchyMetaData::new(file_type),
        }
    }
}

impl HierarchyBuilder {
    pub fn finish(mut self) -> Hierarchy {
        self.vars.shrink_to_fit();
        self.scopes.shrink_to_fit();
        self.strings.shrink_to_fit();
        self.source_locs.shrink_to_fit();
        self.enums.shrink_to_fit();
        self.handle_to_node.shrink_to_fit();
        Hierarchy {
            vars: self.vars,
            scopes: self.scopes,
            first_item: self.first_item,
            strings: self.strings,
            source_locs: self.source_locs,
            enums: self.enums,
            signal_idx_to_var: self.handle_to_node,
            meta: self.meta,
        }
    }

    pub(crate) fn add_string(&mut self, value: String) -> HierarchyStringId {
        if value.is_empty() {
            return EMPTY_STRING;
        }
        // we assign each string a unique ID, currently we make no effort to avoid saving the same string twice
        let sym = HierarchyStringId::from_index(self.strings.len());
        self.strings.push(value);
        debug_assert_eq!(self.strings.len(), sym.index() + 1);
        sym
    }

    pub(crate) fn get_str(&self, id: HierarchyStringId) -> &str {
        &self.strings[id.index()]
    }

    pub(crate) fn add_source_loc(
        &mut self,
        path: HierarchyStringId,
        line: u64,
        is_instantiation: bool,
    ) -> SourceLocId {
        let sym = SourceLocId::from_index(self.source_locs.len());
        self.source_locs.push(SourceLoc {
            path,
            line,
            is_instantiation,
        });
        sym
    }

    pub(crate) fn add_enum_type(
        &mut self,
        name: HierarchyStringId,
        mapping: Vec<(HierarchyStringId, HierarchyStringId)>,
    ) -> EnumTypeId {
        let sym = EnumTypeId::from_index(self.enums.len());
        self.enums.push(EnumType { name, mapping });
        sym
    }

    /// adds a variable or scope to the hierarchy tree
    fn add_to_hierarchy_tree(&mut self, node_id: HierarchyItemId) -> Option<ScopeRef> {
        let entry = find_parent_scope(&mut self.scope_stack);
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

    pub fn add_scope(
        &mut self,
        name: HierarchyStringId,
        component: Option<HierarchyStringId>,
        tpe: ScopeType,
        declaration_source: Option<SourceLocId>,
        instance_source: Option<SourceLocId>,
        flatten: bool,
    ) {
        if flatten {
            self.scope_stack.push(ScopeStackEntry {
                scope_id: usize::MAX,
                last_child: None,
                flattened: true,
            });
        } else {
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
                flattened: false,
            });

            // empty component name is treated the same as none
            let component = match component {
                None => None,
                Some(name) => {
                    if name == EMPTY_STRING {
                        None
                    } else {
                        Some(name)
                    }
                }
            };

            // now we can build the node data structure and store it
            let node = Scope {
                parent,
                child: None,
                next: None,
                name,
                component,
                tpe,
                declaration_source,
                instance_source,
            };
            self.scopes.push(node);
        }
    }

    pub fn add_var(
        &mut self,
        name: HierarchyStringId,
        tpe: VarType,
        direction: VarDirection,
        raw_length: u32,
        index: Option<VarIndex>,
        signal_idx: SignalRef,
        enum_type: Option<EnumTypeId>,
        vhdl_type_name: Option<HierarchyStringId>,
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
            name,
            var_tpe: tpe,
            direction,
            signal_tpe,
            signal_idx,
            enum_type,
            next: None,
            vhdl_type_name,
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

/// finds the first not flattened parent scope
fn find_parent_scope(scope_stack: &mut Vec<ScopeStackEntry>) -> &mut ScopeStackEntry {
    let mut index = scope_stack.len() - 1;
    loop {
        if scope_stack[index].flattened {
            index -= 1;
        } else {
            return &mut scope_stack[index];
        }
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
                + std::mem::size_of::<Option<EnumTypeId>>() // enum type
                + std::mem::size_of::<HierarchyStringId>() // VHDL type name
                + std::mem::size_of::<SignalRef>() // handle
                + std::mem::size_of::<ScopeRef>() // parent
                + std::mem::size_of::<HierarchyItemId>() // next
                + 6 // padding
        );
        // currently this all comes out to 48 bytes (~= 6x 64-bit pointers)
        assert_eq!(std::mem::size_of::<Var>(), 48);

        // Scope
        assert_eq!(
            std::mem::size_of::<Scope>(),
            std::mem::size_of::<HierarchyStringId>() // name
                + std::mem::size_of::<HierarchyStringId>() // component name
                + 1 // tpe
                + 2 // source info
                + 2 // source info
                + std::mem::size_of::<HierarchyItemId>() // child
                + std::mem::size_of::<ScopeRef>() // parent
                + std::mem::size_of::<HierarchyItemId>() // next
                + 1 // padding
        );
        // currently this all comes out to 32 bytes (~= 4x 64-bit pointers)
        assert_eq!(std::mem::size_of::<Scope>(), 32);

        // for comparison: one string is 24 bytes for the struct alone (ignoring heap allocation)
        assert_eq!(std::mem::size_of::<String>(), 24);
    }
}
