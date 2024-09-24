// Copyright 2023-2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>
//
// Space efficient format for a wavedump hierarchy.

use crate::FileFormat;
use std::collections::HashMap;
use std::num::{NonZeroI32, NonZeroU16, NonZeroU32};

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub struct VarRef(NonZeroU32);

impl VarRef {
    #[inline]
    fn from_index(index: usize) -> Option<Self> {
        NonZeroU32::new(index as u32 + 1).map(VarRef)
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
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub struct ScopeRef(NonZeroU32);

impl ScopeRef {
    #[inline]
    fn from_index(index: usize) -> Option<Self> {
        NonZeroU32::new(index as u32 + 1).map(Self)
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
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub struct HierarchyStringId(NonZeroU32);

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
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
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
    pub fn vcd_default() -> Self {
        VarDirection::Unknown
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[repr(packed(4))]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub struct VarIndex {
    lsb: i64,
    width: NonZeroI32,
}

const DEFAULT_ZERO_REPLACEMENT: NonZeroI32 = unsafe { NonZeroI32::new_unchecked(i32::MIN) };

impl VarIndex {
    pub fn new(msb: i64, lsb: i64) -> Self {
        let width = NonZeroI32::new((msb - lsb) as i32).unwrap_or(DEFAULT_ZERO_REPLACEMENT);
        Self { lsb, width }
    }

    #[inline]
    pub fn msb(&self) -> i64 {
        if self.width == DEFAULT_ZERO_REPLACEMENT {
            self.lsb()
        } else {
            self.width.get() as i64 + self.lsb()
        }
    }

    #[inline]
    pub fn lsb(&self) -> i64 {
        self.lsb
    }
}

/// Signal identifier in the waveform (VCD, FST, etc.) file.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub struct SignalRef(NonZeroU32);

impl SignalRef {
    #[inline]
    pub fn from_index(index: usize) -> Option<Self> {
        NonZeroU32::new(index as u32 + 1).map(Self)
    }

    #[inline]
    pub fn index(&self) -> usize {
        (self.0.get() - 1) as usize
    }
}

/// Specifies how the underlying signal of a variable is encoded.
/// This is different from the `VarType` which tries to correspond to the variable type in the
/// source HDL code.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub enum SignalEncoding {
    /// encoded as variable length strings
    String,
    /// encoded as 64-bit floating point values
    Real,
    /// encoded as a fixed width bit-vector
    BitVector(NonZeroU32),
}

impl SignalEncoding {
    pub fn bit_vec_of_len(len: u32) -> Self {
        match NonZeroU32::new(len) {
            // a zero length signal should be represented as a 1-bit signal
            None => SignalEncoding::BitVector(NonZeroU32::new(1).unwrap()),
            Some(value) => SignalEncoding::BitVector(value),
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub struct Var {
    name: HierarchyStringId,
    var_tpe: VarType,
    direction: VarDirection,
    signal_encoding: SignalEncoding,
    index: Option<VarIndex>,
    signal_idx: SignalRef,
    enum_type: Option<EnumTypeId>,
    vhdl_type_name: Option<HierarchyStringId>,
    parent: Option<ScopeRef>,
    next: Option<HierarchyItemId>,
}

/// Represents a slice of another signal identified by its `SignalRef`.
/// This is helpful for formats like GHW where some signals are directly defined as
/// slices of other signals, and thus we only save the data of the larger signal.
#[derive(Debug, Copy, Clone)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub struct SignalSlice {
    pub msb: u32,
    pub lsb: u32,
    pub sliced_signal: SignalRef,
}

const SCOPE_SEPARATOR: char = '.';

impl Var {
    /// Local name of the variable.
    #[inline]
    pub fn name<'a>(&self, hierarchy: &'a Hierarchy) -> &'a str {
        hierarchy.get_str(self.name)
    }

    /// Full hierarchical name of the variable.
    pub fn full_name(&self, hierarchy: &Hierarchy) -> String {
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
        self.index
    }
    pub fn signal_ref(&self) -> SignalRef {
        self.signal_idx
    }
    pub fn length(&self) -> Option<u32> {
        match &self.signal_encoding {
            SignalEncoding::String => None,
            SignalEncoding::Real => None,
            SignalEncoding::BitVector(len) => Some(len.get()),
        }
    }
    pub fn is_real(&self) -> bool {
        matches!(self.signal_encoding, SignalEncoding::Real)
    }
    pub fn is_string(&self) -> bool {
        matches!(self.signal_encoding, SignalEncoding::String)
    }
    pub fn is_bit_vector(&self) -> bool {
        matches!(self.signal_encoding, SignalEncoding::BitVector(_))
    }
    pub fn is_1bit(&self) -> bool {
        match self.length() {
            Some(l) => l == 1,
            _ => false,
        }
    }
    pub fn signal_encoding(&self) -> SignalEncoding {
        self.signal_encoding
    }
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
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
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
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
    pub fn full_name(&self, hierarchy: &Hierarchy) -> String {
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
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub struct SourceLocId(NonZeroU16);

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
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
struct SourceLoc {
    path: HierarchyStringId,
    line: u64,
    is_instantiation: bool,
}

#[derive(Debug, PartialEq, Copy, Clone)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub struct EnumTypeId(NonZeroU16);

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
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
struct EnumType {
    name: HierarchyStringId,
    mapping: Vec<(HierarchyStringId, HierarchyStringId)>,
}

#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub struct Hierarchy {
    vars: Vec<Var>,
    scopes: Vec<Scope>,
    first_item: Option<HierarchyItemId>,
    strings: Vec<String>,
    source_locs: Vec<SourceLoc>,
    enums: Vec<EnumType>,
    signal_idx_to_var: Vec<Option<VarRef>>,
    meta: HierarchyMetaData,
    slices: HashMap<SignalRef, SignalSlice>,
}

#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
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
        HierarchyItemIterator::new(self, self.first_item)
    }

    /// Returns an iterator over references to all top-level scopes.
    pub fn scopes(&self) -> HierarchyScopeRefIterator {
        HierarchyScopeRefIterator {
            underlying: HierarchyItemIdIterator::new(self, self.first_item),
        }
    }

    /// Returns an iterator over references to all top-level variables.
    pub fn vars(&self) -> HierarchyVarRefIterator {
        HierarchyVarRefIterator {
            underlying: HierarchyItemIdIterator::new(self, self.first_item),
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
        let mut scope = self.scopes().find(|s| self.get(*s).name(self) == prefix)?;
        for name in names.iter().skip(1) {
            scope = self
                .get(scope)
                .scopes(self)
                .find(|s| self.get(*s).name(self) == name.as_ref())?;
        }
        Some(scope)
    }

    pub fn lookup_var<N: AsRef<str>>(&self, path: &[N], name: &N) -> Option<VarRef> {
        match path {
            [] => self
                .vars()
                .find(|v| self.get(*v).name(self) == name.as_ref()),
            scopes => {
                let scope = self.get(self.lookup_scope(scopes)?);
                scope
                    .vars(self)
                    .find(|v| self.get(*v).name(self) == name.as_ref())
            }
        }
    }
}

impl Hierarchy {
    pub fn num_unique_signals(&self) -> usize {
        self.signal_idx_to_var.len()
    }

    /// Retrieves the length of a signal identified by its id by looking up a
    /// variable that refers to the signal.
    pub fn get_signal_tpe(&self, signal_idx: SignalRef) -> Option<SignalEncoding> {
        let var_id = (*self.signal_idx_to_var.get(signal_idx.index())?)?;
        Some(self.get(var_id).signal_encoding)
    }

    pub fn get_slice_info(&self, signal_idx: SignalRef) -> Option<SignalSlice> {
        self.slices.get(&signal_idx).copied()
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
    slices: HashMap<SignalRef, SignalSlice>,
}

const EMPTY_STRING: HierarchyStringId = HierarchyStringId(unsafe { NonZeroU32::new_unchecked(1) });

impl HierarchyBuilder {
    pub fn new(file_type: FileFormat) -> Self {
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
            slices: HashMap::default(),
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
        self.slices.shrink_to_fit();
        Hierarchy {
            vars: self.vars,
            scopes: self.scopes,
            first_item: self.first_item,
            strings: self.strings,
            source_locs: self.source_locs,
            enums: self.enums,
            signal_idx_to_var: self.handle_to_node,
            meta: self.meta,
            slices: self.slices,
        }
    }

    pub fn add_string(&mut self, value: String) -> HierarchyStringId {
        if value.is_empty() {
            return EMPTY_STRING;
        }
        // we assign each string a unique ID, currently we make no effort to avoid saving the same string twice
        let sym = HierarchyStringId::from_index(self.strings.len());
        self.strings.push(value);
        debug_assert_eq!(self.strings.len(), sym.index() + 1);
        sym
    }

    pub fn get_str(&self, id: HierarchyStringId) -> &str {
        &self.strings[id.index()]
    }

    pub fn add_source_loc(
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

    pub fn add_enum_type(
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
        let entry_pos = find_parent_scope(&self.scope_stack);
        let entry = &mut self.scope_stack[entry_pos];
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

    /// Checks to see if a scope of the same name already exists.
    fn find_duplicate_scope(&self, name_id: HierarchyStringId) -> Option<ScopeRef> {
        let name = self.get_str(name_id);

        let parent = &self.scope_stack[find_parent_scope(&self.scope_stack)];
        let mut maybe_item = if parent.scope_id == usize::MAX {
            // we are on the top
            self.first_item
        } else {
            let parent_scope = &self.scopes[parent.scope_id];
            parent_scope.child
        };

        while let Some(item) = maybe_item {
            if let HierarchyItemId::Scope(other) = item {
                let scope = &self.scopes[other.index()];
                let other_name = self.get_str(scope.name);
                if other_name == name {
                    // duplicate found!
                    return Some(other);
                }
            }
            maybe_item = self.get_next(item);
        }
        // no duplicate found
        None
    }

    fn get_next(&self, item: HierarchyItemId) -> Option<HierarchyItemId> {
        match item {
            HierarchyItemId::Scope(scope_ref) => self.scopes[scope_ref.index()].next,
            HierarchyItemId::Var(var_ref) => self.vars[var_ref.index()].next,
        }
    }

    fn find_last_child(&self, scope: ScopeRef) -> Option<HierarchyItemId> {
        if let Some(mut child) = self.scopes[scope.index()].child {
            while let Some(next) = self.get_next(child) {
                child = next;
            }
            Some(child)
        } else {
            None
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
        // check to see if there is a scope of the same name already
        // if so we just activate that scope instead of adding a new one
        if let Some(duplicate) = self.find_duplicate_scope(name) {
            let last_child = self.find_last_child(duplicate);
            self.scope_stack.push(ScopeStackEntry {
                scope_id: duplicate.index(),
                last_child,
                flattened: false,
            })
        } else if flatten {
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
            let component = component.filter(|&name| name != EMPTY_STRING);

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

    /// Helper function for adding scopes that were generated from a nested array name in VCD or FST.
    #[inline]
    pub fn add_array_scopes(&mut self, names: Vec<String>) {
        for name in names {
            let name_id = self.add_string(name);
            self.add_scope(name_id, None, ScopeType::VhdlArray, None, None, false);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_var(
        &mut self,
        name: HierarchyStringId,
        tpe: VarType,
        signal_tpe: SignalEncoding,
        direction: VarDirection,
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

        // now we can build the node data structure and store it
        let node = Var {
            parent,
            name,
            var_tpe: tpe,
            index,
            direction,
            signal_encoding: signal_tpe,
            signal_idx,
            enum_type,
            next: None,
            vhdl_type_name,
        };
        self.vars.push(node);
    }

    #[inline]
    pub fn pop_scopes(&mut self, num: usize) {
        for _ in 0..num {
            self.pop_scope();
        }
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

    pub fn add_slice(
        &mut self,
        signal_ref: SignalRef,
        msb: u32,
        lsb: u32,
        sliced_signal: SignalRef,
    ) {
        debug_assert!(msb >= lsb);
        debug_assert!(!self.slices.contains_key(&signal_ref));
        self.slices.insert(
            signal_ref,
            SignalSlice {
                msb,
                lsb,
                sliced_signal,
            },
        );
    }
}

/// finds the first not flattened parent scope
fn find_parent_scope(scope_stack: &[ScopeStackEntry]) -> usize {
    let mut index = scope_stack.len() - 1;
    loop {
        if scope_stack[index].flattened {
            index -= 1;
        } else {
            return index;
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

        // 4 byte length + tag + padding
        assert_eq!(std::mem::size_of::<SignalEncoding>(), 8);

        // var index is packed in order to take up 12 bytes and contains a NonZero field to allow
        // for zero cost optioning
        assert_eq!(
            std::mem::size_of::<VarIndex>(),
            std::mem::size_of::<Option<VarIndex>>()
        );
        assert_eq!(std::mem::size_of::<VarIndex>(), 12);

        // Var
        assert_eq!(
            std::mem::size_of::<Var>(),
            std::mem::size_of::<HierarchyStringId>()        // name
                + std::mem::size_of::<VarType>()            // var_tpe
                + std::mem::size_of::<VarDirection>()       // direction
                + std::mem::size_of::<SignalEncoding>()     // signal_encoding
                + std::mem::size_of::<Option<VarIndex>>()   // index
                + std::mem::size_of::<SignalRef>()          // signal_idx
                + std::mem::size_of::<Option<EnumTypeId>>() // enum type
                + std::mem::size_of::<HierarchyStringId>()  // VHDL type name
                + std::mem::size_of::<Option<ScopeRef>>()   // parent
                + std::mem::size_of::<HierarchyItemId>()    // next
                + 2 // padding
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

    #[test]
    fn test_var_index() {
        let msb = 15;
        let lsb = -1;
        let index = VarIndex::new(msb, lsb);
        assert_eq!(index.lsb(), lsb);
        assert_eq!(index.msb(), msb);
    }
}
