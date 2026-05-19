// Copyright 2023-2024 The Regents of the University of California
// Copyright 2024-2026 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use crate::FileFormat;
use crate::signal::DerivedBitVecSignal;
use indexmap::IndexSet;
use rustc_hash::{FxBuildHasher, FxHashMap};
use std::fmt::{Debug, Formatter};
use std::num::{NonZeroI32, NonZeroU16, NonZeroU32};
use std::ops::{Index, IndexMut};

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
    ZeptoSeconds,
    AttoSeconds,
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
            TimescaleUnit::ZeptoSeconds => Some(-21),
            TimescaleUnit::AttoSeconds => Some(-18),
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub struct VarRef(NonZeroU32);

impl VarRef {
    #[inline]
    pub fn from_index(index: usize) -> Option<Self> {
        NonZeroU32::new(index as u32 + 1).map(VarRef)
    }

    #[inline]
    pub fn index(&self) -> usize {
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
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub struct ScopeRef(NonZeroU32);

impl ScopeRef {
    #[inline]
    pub const fn from_index(index: usize) -> Option<Self> {
        match NonZeroU32::new(index as u32 + 1) {
            None => None,
            Some(v) => Some(Self(v)),
        }
    }

    #[inline]
    pub fn index(&self) -> usize {
        (self.0.get() - 1) as usize
    }
}

impl Default for ScopeRef {
    fn default() -> Self {
        Self::from_index(0).unwrap()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
#[non_exhaustive]
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
    // for questa sim
    Unknown,
    // SystemVerilog
    Clocking,
    SvArray,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum ScopePackInfo {
    Packed,
    Unpacked,
    Sparse,
    TaggedPacked,
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
    RealParameter,
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
///
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

#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(Rust, packed(4))]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub struct VarIndex {
    lsb: i64,
    width: NonZeroI32,
}

impl Debug for VarIndex {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "VarIndex([{}:{}])", self.msb(), self.lsb())
    }
}

const DEFAULT_ZERO_REPLACEMENT: NonZeroI32 = NonZeroI32::new(i32::MIN).unwrap();

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
            i64::from(self.width.get()) + self.lsb()
        }
    }

    #[inline]
    pub fn lsb(&self) -> i64 {
        self.lsb
    }

    #[inline]
    pub fn width(&self) -> u32 {
        if self.width == DEFAULT_ZERO_REPLACEMENT {
            1
        } else {
            self.width.get().unsigned_abs() + 1
        }
    }
}

/// Signal identifier in the waveform (VCD, FST, etc.) file.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub struct SignalRef(NonZeroU32);

/// The upper bit is used for indicating whether the signal is derived.
const MAX_INDEX: usize = ((u32::MAX as usize) >> 2) - 1;
const INDEX_MASK: u32 = u32::MAX >> 1;

impl SignalRef {
    #[inline]
    pub fn from_index(index: usize) -> Option<Self> {
        if index > MAX_INDEX {
            None
        } else {
            Some(Self(NonZeroU32::new(index as u32 + 1).unwrap()))
        }
    }

    #[inline]
    pub fn derived_from_index(index: usize) -> Option<Self> {
        if index > MAX_INDEX {
            None
        } else {
            let value = (index as u32 + 1) | (1u32 << 31);
            Some(Self(NonZeroU32::new(value).unwrap()))
        }
    }

    /// Generates a dummy ID for a derived signal.
    #[inline]
    pub(crate) fn derived_max() -> Self {
        Self::derived_from_index(MAX_INDEX).unwrap()
    }

    /// A derived signal does not actually exist in the original waveform trace.
    #[inline]
    pub fn is_derived_signal(&self) -> bool {
        self.0.get() >> 31 == 1
    }

    #[inline]
    pub fn index(&self) -> usize {
        ((self.0.get() & INDEX_MASK) - 1) as usize
    }

    #[inline]
    pub fn to_derived(&self) -> Self {
        let value = self.0.get() | (1u32 << 31);
        Self(NonZeroU32::new(value).unwrap())
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
    /// essentially a bit vector of size 0
    Event,
    /// the encoding was never supplied
    Unknown,
}

impl SignalEncoding {
    pub fn bit_vec_of_len(len: u32) -> Self {
        match NonZeroU32::new(len) {
            // a zero length signal should be represented as a 1-bit signal
            None => SignalEncoding::BitVector(NonZeroU32::new(1).unwrap()),
            Some(value) => SignalEncoding::BitVector(value),
        }
    }

    pub fn length(&self) -> Option<u32> {
        match &self {
            SignalEncoding::String | SignalEncoding::Real | SignalEncoding::Unknown => None,
            SignalEncoding::Event => Some(0),
            SignalEncoding::BitVector(len) => Some(len.get()),
        }
    }

    pub fn is_real(&self) -> bool {
        matches!(self, SignalEncoding::Real)
    }
    pub fn is_string(&self) -> bool {
        matches!(self, SignalEncoding::String)
    }
    pub fn is_bit_vector(&self) -> bool {
        matches!(self, SignalEncoding::BitVector(_))
    }

    pub fn is_1bit(&self) -> bool {
        match self.length() {
            Some(l) => l == 1,
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub struct Var {
    name: HierarchyStringId,
    var_tpe: VarType,
    direction: VarDirection,
    index: Option<VarIndex>,
    signal_idx: SignalRef,
    enum_type: Option<EnumTypeId>,
    vhdl_type_name: Option<HierarchyStringId>,
    parent: Option<ScopeRef>,
    next: Option<ScopeOrVarRef>,
}

const SCOPE_SEPARATOR: char = '.';

impl Var {
    /// Local name of the variable.
    #[inline]
    pub fn name<'a>(&self, hierarchy: &'a Hierarchy) -> &'a str {
        &hierarchy[self.name]
    }

    /// Full hierarchical name of the variable.
    pub fn full_name(&self, hierarchy: &Hierarchy) -> String {
        match self.parent {
            None => self.name(hierarchy).to_string(),
            Some(parent) => {
                let mut out = hierarchy[parent].full_name(hierarchy);
                out.push(SCOPE_SEPARATOR);
                out.push_str(self.name(hierarchy));
                out
            }
        }
    }

    #[inline]
    pub fn var_type(&self) -> VarType {
        self.var_tpe
    }

    #[inline]
    pub fn enum_type<'a>(
        &self,
        hierarchy: &'a Hierarchy,
    ) -> Option<(&'a str, Vec<(&'a str, &'a str)>)> {
        self.enum_type.map(|id| hierarchy.get_enum_type(id))
    }

    #[inline]
    pub fn vhdl_type_name<'a>(&self, hierarchy: &'a Hierarchy) -> Option<&'a str> {
        self.vhdl_type_name.map(|i| &hierarchy[i])
    }

    #[inline]
    pub fn direction(&self) -> VarDirection {
        self.direction
    }

    #[inline]
    pub fn index(&self) -> Option<VarIndex> {
        self.index
    }

    #[inline]
    pub fn signal_ref(&self) -> SignalRef {
        self.signal_idx
    }

    #[inline]
    pub fn length(&self, h: &Hierarchy) -> Option<u32> {
        self.signal_encoding(h).length()
    }

    #[inline]
    pub fn is_real(&self, h: &Hierarchy) -> bool {
        self.signal_encoding(h).is_real()
    }

    #[inline]
    pub fn is_string(&self, h: &Hierarchy) -> bool {
        self.signal_encoding(h).is_string()
    }

    #[inline]
    pub fn is_bit_vector(&self, h: &Hierarchy) -> bool {
        self.signal_encoding(h).is_bit_vector()
    }

    #[inline]
    pub fn is_1bit(&self, h: &Hierarchy) -> bool {
        self.signal_encoding(h).is_1bit()
    }

    #[inline]
    pub fn signal_encoding(&self, h: &Hierarchy) -> SignalEncoding {
        h[self.signal_idx]
    }
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub enum ScopeOrVarRef {
    Scope(ScopeRef),
    Var(VarRef),
}

impl ScopeOrVarRef {
    pub fn deref<'a>(&self, h: &'a Hierarchy) -> ScopeOrVar<'a> {
        h.get_item(*self)
    }
}

impl From<ScopeRef> for ScopeOrVarRef {
    fn from(value: ScopeRef) -> Self {
        Self::Scope(value)
    }
}

impl From<VarRef> for ScopeOrVarRef {
    fn from(value: VarRef) -> Self {
        Self::Var(value)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ScopeOrVar<'a> {
    Scope(&'a Scope),
    Var(&'a Var),
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub struct Scope {
    name: HierarchyStringId,
    /// Some wave formats supply the name of the component, e.g., of the module that was instantiated.
    component: Option<HierarchyStringId>,
    tpe: ScopeType,
    pack: Option<ScopePackInfo>,
    declaration_source: Option<SourceLocId>,
    instance_source: Option<SourceLocId>,
    child: Option<ScopeOrVarRef>,
    parent: Option<ScopeRef>,
    next: Option<ScopeOrVarRef>,
}

impl Scope {
    /// Local name of the scope.
    pub fn name<'a>(&self, hierarchy: &'a Hierarchy) -> &'a str {
        &hierarchy[self.name]
    }

    /// Local name of the component, e.g., the name of the module that was instantiated.
    pub fn component<'a>(&self, hierarchy: &'a Hierarchy) -> Option<&'a str> {
        self.component.map(|n| &hierarchy[n])
    }

    /// Full hierarchical name of the scope.
    pub fn full_name(&self, hierarchy: &Hierarchy) -> String {
        let mut parents = Vec::new();
        let mut parent = self.parent;
        while let Some(id) = parent {
            parents.push(id);
            parent = hierarchy[id].parent;
        }
        let mut out: String = String::with_capacity((parents.len() + 1) * 5);
        for parent_id in parents.iter().rev() {
            out.push_str(hierarchy[*parent_id].name(hierarchy));
            out.push(SCOPE_SEPARATOR)
        }
        out.push_str(self.name(hierarchy));
        out
    }

    pub fn scope_type(&self) -> ScopeType {
        self.tpe
    }

    pub fn pack_info(&self) -> Option<ScopePackInfo> {
        self.pack
    }

    pub fn source_loc<'a>(&self, hierarchy: &'a Hierarchy) -> Option<(&'a str, u64)> {
        self.declaration_source
            .map(|id| hierarchy.get_source_loc(id))
    }

    pub fn instantiation_source_loc<'a>(&self, hierarchy: &'a Hierarchy) -> Option<(&'a str, u64)> {
        self.instance_source.map(|id| hierarchy.get_source_loc(id))
    }

    pub fn items<'a>(
        &'a self,
        hierarchy: &'a Hierarchy,
    ) -> impl Iterator<Item = ScopeOrVarRef> + 'a {
        HierarchyItemIdIterator::new(hierarchy, self.child)
    }

    pub fn vars<'a>(&'a self, hierarchy: &'a Hierarchy) -> impl Iterator<Item = VarRef> + 'a {
        to_var_ref_iterator(HierarchyItemIdIterator::new(hierarchy, self.child))
    }

    pub fn scopes<'a>(&'a self, hierarchy: &'a Hierarchy) -> impl Iterator<Item = ScopeRef> + 'a {
        to_scope_ref_iterator(HierarchyItemIdIterator::new(hierarchy, self.child))
    }
}

struct HierarchyItemIdIterator<'a> {
    hierarchy: &'a Hierarchy,
    item: Option<ScopeOrVarRef>,
    is_first: bool,
}

impl<'a> HierarchyItemIdIterator<'a> {
    fn new(hierarchy: &'a Hierarchy, item: Option<ScopeOrVarRef>) -> Self {
        Self {
            hierarchy,
            item,
            is_first: true,
        }
    }

    fn get_next(&self, item: ScopeOrVarRef) -> Option<ScopeOrVarRef> {
        match self.hierarchy.get_item(item) {
            ScopeOrVar::Scope(scope) => scope.next,
            ScopeOrVar::Var(var) => var.next,
        }
    }
}

impl Iterator for HierarchyItemIdIterator<'_> {
    type Item = ScopeOrVarRef;

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

fn to_var_ref_iterator(iter: impl Iterator<Item = ScopeOrVarRef>) -> impl Iterator<Item = VarRef> {
    iter.flat_map(|i| match i {
        ScopeOrVarRef::Scope(_) => None,
        ScopeOrVarRef::Var(v) => Some(v),
    })
}

fn to_scope_ref_iterator(
    iter: impl Iterator<Item = ScopeOrVarRef>,
) -> impl Iterator<Item = ScopeRef> {
    iter.flat_map(|i| match i {
        ScopeOrVarRef::Scope(s) => Some(s),
        ScopeOrVarRef::Var(_) => None,
    })
}

#[derive(Debug, PartialEq, Copy, Clone)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub struct SourceLocId(NonZeroU32);

impl SourceLocId {
    #[inline]
    fn from_index(index: usize) -> Self {
        let value = (index + 1) as u32;
        SourceLocId(NonZeroU32::new(value).unwrap())
    }

    #[inline]
    fn index(self) -> usize {
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
    fn index(self) -> usize {
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
    strings: Vec<String>,
    source_locs: Vec<SourceLoc>,
    enums: Vec<EnumType>,
    signal_encodings: Vec<SignalEncoding>,
    meta: HierarchyMetaData,
    signal_derivations: FxHashMap<SignalRef, DerivedBitVecSignal>,
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
    pub fn all_vars(&self) -> impl Iterator<Item = &Var> + '_ {
        self.vars.iter()
    }

    /// Returns an iterator over all scopes (at all levels).
    pub fn all_scopes(&self) -> impl Iterator<Item = &Scope> + '_ {
        self.scopes.iter()
    }

    /// Retrieves the first item inside the implicit fake top scope
    fn first_item(&self) -> Option<ScopeOrVarRef> {
        debug_assert!(self.scopes[FAKE_TOP_SCOPE.index()].next.is_none());
        self.scopes[FAKE_TOP_SCOPE.index()].child
    }

    /// Returns an iterator over references to all top-level scopes and variables.
    pub fn items(&self) -> impl Iterator<Item = ScopeOrVarRef> + '_ {
        HierarchyItemIdIterator::new(self, self.first_item())
    }

    /// Returns an iterator over references to all top-level scopes.
    pub fn scopes(&self) -> impl Iterator<Item = ScopeRef> + '_ {
        to_scope_ref_iterator(HierarchyItemIdIterator::new(self, self.first_item()))
    }

    /// Returns an iterator over references to all top-level variables.
    pub fn vars(&self) -> impl Iterator<Item = VarRef> + '_ {
        to_var_ref_iterator(HierarchyItemIdIterator::new(self, self.first_item()))
    }

    /// Returns the first scope that was declared in the underlying file.
    pub fn first_scope(&self) -> Option<&Scope> {
        // we need to skip the fake top scope
        self.scopes.get(1)
    }

    /// Encoding for all signals. The position of a signal encoding can be mapped to a SignalRef.
    pub fn signal_encodings(&self) -> &[SignalEncoding] {
        &self.signal_encodings
    }

    /// Iterate over all signal references with a known type.
    pub fn signals(&self) -> impl Iterator<Item = SignalRef> {
        self.signal_encodings()
            .iter()
            .enumerate()
            .filter(|(_, enc)| **enc != SignalEncoding::Unknown)
            .map(|(ii, _)| SignalRef::from_index(ii).unwrap())
    }

    /// Size of the Hierarchy in bytes.
    pub fn size_in_memory(&self) -> usize {
        let var_size = self.vars.capacity() * std::mem::size_of::<Var>();
        let scope_size = self.scopes.capacity() * std::mem::size_of::<Scope>();
        let string_size = self.strings.capacity() * std::mem::size_of::<String>()
            + self.strings.iter().map(|s| s.len()).sum::<usize>();
        let signal_encodings_size =
            self.signal_encodings.capacity() * std::mem::size_of::<SignalEncoding>();
        var_size
            + scope_size
            + string_size
            + signal_encodings_size
            + std::mem::size_of::<Hierarchy>()
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
        let mut scope = self.scopes().find(|s| self[*s].name(self) == prefix)?;
        for name in names.iter().skip(1) {
            scope = self[scope]
                .scopes(self)
                .find(|s| self[*s].name(self) == name.as_ref())?;
        }
        Some(scope)
    }

    pub fn lookup_var<N: AsRef<str>>(&self, path: &[N], name: N) -> Option<VarRef> {
        self.lookup_var_with_index(path, name, &None)
    }

    pub fn lookup_var_with_index<N: AsRef<str>>(
        &self,
        path: &[N],
        name: N,
        index: &Option<VarIndex>,
    ) -> Option<VarRef> {
        match path {
            [] => self.vars().find(|v| {
                let v = &self[*v];
                v.name(self) == name.as_ref() && (index.is_none() || (v.index == *index))
            }),
            scopes => {
                let scope = &self[self.lookup_scope(scopes)?];
                scope.vars(self).find(|v| {
                    let v = &self[*v];
                    v.name(self) == name.as_ref() && (index.is_none() || (v.index == *index))
                })
            }
        }
    }
}

impl Hierarchy {
    /// Retrieves the length of a signal identified by its id by looking up a
    /// variable that refers to the signal.
    pub fn get_signal_tpe(&self, signal_idx: SignalRef) -> Option<SignalEncoding> {
        self.signal_encodings.get(signal_idx.index()).copied()
    }

    pub fn get_derived_signal(&self, signal_idx: SignalRef) -> Option<&DerivedBitVecSignal> {
        self.signal_derivations.get(&signal_idx)
    }
}

// private implementation
impl Hierarchy {
    fn get_source_loc(&self, id: SourceLocId) -> (&str, u64) {
        let loc = &self.source_locs[id.index()];
        (&self[loc.path], loc.line)
    }

    fn get_enum_type(&self, id: EnumTypeId) -> (&str, Vec<(&str, &str)>) {
        let enum_tpe = &self.enums[id.index()];
        let name = &self[enum_tpe.name];
        let mapping = enum_tpe
            .mapping
            .iter()
            .map(|(a, b)| (&self[*a], &self[*b]))
            .collect::<Vec<_>>();
        (name, mapping)
    }

    fn get_item(&self, id: ScopeOrVarRef) -> ScopeOrVar<'_> {
        match id {
            ScopeOrVarRef::Scope(id) => ScopeOrVar::Scope(&self[id]),
            ScopeOrVarRef::Var(id) => ScopeOrVar::Var(&self[id]),
        }
    }
}

impl Index<VarRef> for Hierarchy {
    type Output = Var;

    fn index(&self, index: VarRef) -> &Self::Output {
        &self.vars[index.index()]
    }
}

impl IndexMut<VarRef> for Hierarchy {
    fn index_mut(&mut self, index: VarRef) -> &mut Self::Output {
        &mut self.vars[index.index()]
    }
}

impl Index<ScopeRef> for Hierarchy {
    type Output = Scope;

    fn index(&self, index: ScopeRef) -> &Self::Output {
        &self.scopes[index.index()]
    }
}

impl IndexMut<ScopeRef> for Hierarchy {
    fn index_mut(&mut self, index: ScopeRef) -> &mut Self::Output {
        &mut self.scopes[index.index()]
    }
}

impl Index<HierarchyStringId> for Hierarchy {
    type Output = str;

    fn index(&self, index: HierarchyStringId) -> &Self::Output {
        &self.strings[index.index()]
    }
}

impl Index<SignalRef> for Hierarchy {
    type Output = SignalEncoding;

    fn index(&self, index: SignalRef) -> &Self::Output {
        &self.signal_encodings[index.index()]
    }
}

struct ScopeStackEntry {
    scope_id: usize,
    last_child: Option<ScopeOrVarRef>,
    /// indicates that this scope is being flattened and all operations should be done on the parent instead
    flattened: bool,
}

pub struct HierarchyBuilder {
    vars: Vec<Var>,
    scopes: Vec<Scope>,
    scope_stack: Vec<ScopeStackEntry>,
    source_locs: Vec<SourceLoc>,
    enums: Vec<EnumType>,
    signal_encodings: Vec<SignalEncoding>,
    meta: HierarchyMetaData,
    /// derived signals where we know the SignalRef during building
    signal_derivations: FxHashMap<SignalRef, DerivedBitVecSignal>,
    /// derived signals where we do not have a SignalRef at creation time
    var_to_derived: FxHashMap<VarRef, DerivedBitVecSignal>,
    /// used to deduplicate strings
    strings: IndexSet<String, FxBuildHasher>,
    /// keeps track of the number of children to decide when to switch to a hash table based
    /// deduplication strategy
    scope_child_count: Vec<u8>,
    scope_dedup_tables: FxHashMap<ScopeRef, FxHashMap<HierarchyStringId, ScopeRef>>,
}

const EMPTY_STRING: HierarchyStringId = HierarchyStringId(NonZeroU32::new(1).unwrap());
const FAKE_TOP_SCOPE: ScopeRef = ScopeRef::from_index(0).unwrap();
/// Scopes that have a larger number of children will get a HashTable to
/// speed up searching for duplicates. Otherwise, we run into a O(n**2) problem.
const DUPLICATE_SCOPE_HASH_TABLE_THRESHOLD: u8 = 128;

impl HierarchyBuilder {
    pub fn new(file_type: FileFormat) -> Self {
        // we start with a fake entry in the scope stack to keep track of multiple items in the top scope
        let scope_stack = vec![ScopeStackEntry {
            scope_id: FAKE_TOP_SCOPE.index(),
            last_child: None,
            flattened: false,
        }];
        let fake_top_scope = Scope {
            name: EMPTY_STRING,
            component: None,
            tpe: ScopeType::Module,
            pack: None,
            declaration_source: None,
            instance_source: None,
            child: None,
            parent: None,
            next: None,
        };
        let mut strings: IndexSet<String, FxBuildHasher> = Default::default();
        let (zero_id, _) = strings.insert_full("".to_string());
        // empty string should always map to zero
        debug_assert_eq!(zero_id, 0);
        HierarchyBuilder {
            vars: Vec::default(),
            scopes: vec![fake_top_scope],
            scope_stack,
            strings,
            source_locs: Vec::default(),
            enums: Vec::default(),
            signal_encodings: Vec::default(),
            meta: HierarchyMetaData::new(file_type),
            signal_derivations: FxHashMap::default(),
            var_to_derived: FxHashMap::default(),
            scope_child_count: vec![0],
            scope_dedup_tables: Default::default(),
        }
    }
}

impl HierarchyBuilder {
    pub fn finish(mut self) -> Hierarchy {
        self.generate_signal_refs_for_derived();
        debug_assert!(self.var_to_derived.is_empty());
        self.vars.shrink_to_fit();
        self.scopes.shrink_to_fit();
        self.strings.shrink_to_fit();
        self.source_locs.shrink_to_fit();
        self.enums.shrink_to_fit();
        self.signal_encodings.shrink_to_fit();
        self.signal_derivations.shrink_to_fit();
        debug_assert!(
            self.signal_derivations
                .keys()
                .all(|r| r.is_derived_signal())
        );
        Hierarchy {
            vars: self.vars,
            scopes: self.scopes,
            strings: self.strings.into_iter().collect::<Vec<_>>(),
            source_locs: self.source_locs,
            enums: self.enums,
            meta: self.meta,
            signal_derivations: self.signal_derivations,
            signal_encodings: self.signal_encodings,
        }
    }

    /// Called from finish. Assigns signal references for all derived signals
    /// and patches the variables with the final reference.
    /// This can only be done at the end of the builder phase, as only then, do we know
    /// that no new signal references will be created.
    fn generate_signal_refs_for_derived(&mut self) {
        if self.var_to_derived.is_empty() {
            return;
        }
        // deduplicate signals
        let mut signal_to_ref = FxHashMap::default();
        let var_to_derived = std::mem::take(&mut self.var_to_derived);
        for (var, signal) in var_to_derived {
            let signal_enc = signal.output_encoding();
            let signal_ref = *signal_to_ref.entry(signal).or_insert_with(|| {
                let r = SignalRef::derived_from_index(self.signal_encodings.len()).unwrap();
                self.signal_encodings.push(signal_enc);
                r
            });
            self.vars[var.index()].signal_idx = signal_ref;
        }

        for (signal, signal_ref) in signal_to_ref {
            debug_assert!(!self.signal_derivations.contains_key(&signal_ref));
            self.signal_derivations.insert(signal_ref, signal);
        }
    }

    pub fn add_string(&mut self, value: std::borrow::Cow<str>) -> HierarchyStringId {
        if value.is_empty() {
            return EMPTY_STRING;
        }
        if let Some(index) = self.strings.get_index_of(value.as_ref()) {
            HierarchyStringId::from_index(index)
        } else {
            let (index, _) = self.strings.insert_full(value.into_owned());
            HierarchyStringId::from_index(index)
        }
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
    fn add_to_hierarchy_tree(&mut self, node_id: ScopeOrVarRef) -> Option<ScopeRef> {
        let entry_pos = find_parent_scope(&self.scope_stack);
        let entry = &mut self.scope_stack[entry_pos];
        let parent = entry.scope_id;
        match entry.last_child {
            Some(ScopeOrVarRef::Var(child)) => {
                // add pointer to new node from last child
                assert!(self.vars[child.index()].next.is_none());
                self.vars[child.index()].next = Some(node_id);
            }
            Some(ScopeOrVarRef::Scope(child)) => {
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
        // return the parent id, unless it is the fake top scope
        let parent_ref = ScopeRef::from_index(parent).unwrap();
        if parent_ref == FAKE_TOP_SCOPE {
            None
        } else {
            Some(parent_ref)
        }
    }

    /// Checks to see if a scope of the same name already exists.
    fn find_duplicate_scope(&self, name_id: HierarchyStringId) -> Option<ScopeRef> {
        let parent = &self.scope_stack[find_parent_scope(&self.scope_stack)];

        if self.scope_child_count[parent.scope_id] > DUPLICATE_SCOPE_HASH_TABLE_THRESHOLD {
            let parent_ref = ScopeRef::from_index(parent.scope_id).unwrap();
            debug_assert!(self.scope_dedup_tables.contains_key(&parent_ref));
            self.scope_dedup_tables[&parent_ref].get(&name_id).cloned()
        } else {
            // linear search
            let mut maybe_item = self.scopes[parent.scope_id].child;

            while let Some(item) = maybe_item {
                if let ScopeOrVarRef::Scope(other) = item {
                    // it is enough to compare the string id
                    // since strings get the same id iff they have the same value
                    if self.scopes[other.index()].name == name_id {
                        // duplicate found!
                        return Some(other);
                    }
                }
                maybe_item = self.get_next(item);
            }
            // no duplicate found
            None
        }
    }

    fn get_next(&self, item: ScopeOrVarRef) -> Option<ScopeOrVarRef> {
        match item {
            ScopeOrVarRef::Scope(scope_ref) => self.scopes[scope_ref.index()].next,
            ScopeOrVarRef::Var(var_ref) => self.vars[var_ref.index()].next,
        }
    }

    fn find_last_child(&self, scope: ScopeRef) -> Option<ScopeOrVarRef> {
        if let Some(mut child) = self.scopes[scope.index()].child {
            while let Some(next) = self.get_next(child) {
                child = next;
            }
            Some(child)
        } else {
            None
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_scope(
        &mut self,
        name: HierarchyStringId,
        component: Option<HierarchyStringId>,
        tpe: ScopeType,
        pack: Option<ScopePackInfo>,
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
            let scope_ref = ScopeRef::from_index(node_id).unwrap();
            let parent = self.add_to_hierarchy_tree(scope_ref.into());

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
                pack,
                declaration_source,
                instance_source,
            };
            self.scopes.push(node);
            self.scope_child_count.push(0);

            // increment the child count of the parent
            self.increment_child_count(parent, name, scope_ref.into());
        }
    }

    /// increments the child count and generates a hash table if we cross the threshold
    /// must be called after the child is inserted into the list and into the scope Vec
    fn increment_child_count(
        &mut self,
        parent: Option<ScopeRef>,
        child_name: HierarchyStringId,
        child_ref: ScopeOrVarRef,
    ) {
        let p = if let Some(p) = parent {
            p
        } else {
            FAKE_TOP_SCOPE
        };

        let child_count = self.scope_child_count[p.index()];

        if child_count < DUPLICATE_SCOPE_HASH_TABLE_THRESHOLD {
            self.scope_child_count[p.index()] += 1;
        } else if child_count == DUPLICATE_SCOPE_HASH_TABLE_THRESHOLD {
            // we are at the threshold
            self.scope_child_count[p.index()] += 1;
            debug_assert!(!self.scope_dedup_tables.contains_key(&p));
            let lookup = self.build_child_scope_map(p);
            self.scope_dedup_tables.insert(p, lookup);
        } else {
            debug_assert_eq!(child_count, DUPLICATE_SCOPE_HASH_TABLE_THRESHOLD + 1);
            debug_assert!(self.scope_dedup_tables.contains_key(&p));
            // insert new name
            if let ScopeOrVarRef::Scope(child_scope_ref) = child_ref {
                self.scope_dedup_tables
                    .get_mut(&p)
                    .unwrap()
                    .insert(child_name, child_scope_ref);
            }
        }
    }

    /// Creates a mapping of child scope names to scope references
    fn build_child_scope_map(&self, parent: ScopeRef) -> FxHashMap<HierarchyStringId, ScopeRef> {
        let mut maybe_item = self.scopes[parent.index()].child;
        let mut out = FxHashMap::default();
        while let Some(item) = maybe_item {
            if let ScopeOrVarRef::Scope(child_scope) = item {
                let scope = &self.scopes[child_scope.index()];
                out.insert(scope.name, child_scope);
            }
            maybe_item = self.get_next(item);
        }
        out
    }

    /// Helper function for adding scopes that were generated from a nested array name in VCD or FST.
    #[inline]
    pub fn add_array_scopes(&mut self, names: Vec<std::borrow::Cow<str>>) {
        for name in names {
            let name_id = self.add_string(name);
            self.add_scope(name_id, None, ScopeType::VhdlArray, None, None, None, false);
        }
    }

    /// Checks to see if the previous var has the same name and an adjacent index.
    /// In this case, the two variables are merged into one.
    /// This is important to deal with QuestaSim and ModelSim splitting bit-vectors into
    /// individual bits when generating VCDs.
    fn check_for_split_var(
        &mut self,
        new_name: HierarchyStringId,
        index: VarIndex,
        signal_encoding: SignalEncoding,
        signal_idx: SignalRef,
    ) -> bool {
        debug_assert!(
            !signal_idx.is_derived_signal(),
            "Only works for original signals."
        );
        // lookup previous item
        let entry_pos = find_parent_scope(&self.scope_stack);
        let entry = &mut self.scope_stack[entry_pos];
        // is it a variable?
        if let Some(ScopeOrVarRef::Var(prev_var_ref)) = entry.last_child {
            let prev_var = &mut self.vars[prev_var_ref.index()];
            // does the name match? does the variable have an index?
            if prev_var.name == new_name
                && let Some(prev_index) = prev_var.index
            {
                let new_is_msb = index.lsb() == prev_index.msb() + 1;
                let new_is_lsb = prev_index.lsb() == index.msb() + 1;
                if new_is_lsb || new_is_msb {
                    let prev_derived =
                        self.var_to_derived.entry(prev_var_ref).or_insert_with(|| {
                            DerivedBitVecSignal::new_identity(
                                prev_var.signal_ref(),
                                self.signal_encodings[prev_var.signal_ref().index()],
                            )
                        });
                    // modify existing variable (we assume that the other properties are the same
                    if new_is_msb {
                        prev_var.index = Some(VarIndex::new(index.msb(), prev_index.lsb()));
                        prev_derived.concat_left_full(signal_idx, signal_encoding);
                    } else {
                        prev_var.index = Some(VarIndex::new(prev_index.msb(), index.lsb()));
                        prev_derived.concat_right_full(signal_idx, signal_encoding);
                    };
                    debug_assert_eq!(prev_var.index.unwrap().width(), prev_derived.width());

                    // remember type of (potentially) new signal
                    debug_assert_eq!(
                        signal_encoding,
                        SignalEncoding::BitVector(NonZeroU32::new(index.width()).unwrap())
                    );
                    self.set_signal_encoding(signal_idx, signal_encoding, new_name);

                    // a merge happened!
                    return true;
                }
            }
        }
        false
    }

    fn set_signal_encoding(
        &mut self,
        signal: SignalRef,
        encoding: SignalEncoding,
        name: HierarchyStringId,
    ) {
        let ii = signal.index();
        if self.signal_encodings.len() <= ii {
            self.signal_encodings
                .resize(ii + 1, SignalEncoding::Unknown);
        }
        debug_assert!(
            self.signal_encodings[ii] == SignalEncoding::Unknown
                || self.signal_encodings[ii] == encoding,
            "Trying to redefine signal encoding: {:?} -> {:?} for {}",
            self.signal_encodings[ii],
            encoding,
            self.strings[name.index()],
        );
        self.signal_encodings[ii] = encoding;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_var(
        &mut self,
        name: HierarchyStringId,
        tpe: VarType,
        signal_encoding: SignalEncoding,
        direction: VarDirection,
        index: Option<VarIndex>,
        signal_idx: SignalRef,
        enum_type: Option<EnumTypeId>,
        vhdl_type_name: Option<HierarchyStringId>,
    ) {
        if let Some(ii) = index
            && self.check_for_split_var(name, ii, signal_encoding, signal_idx)
        {
            // we merged with an existing variable, no need to add a new one
            return;
        }

        let node_id = self.vars.len();
        let var_id = VarRef::from_index(node_id).unwrap();
        let parent = self.add_to_hierarchy_tree(var_id.into());

        // update signal encoding
        self.set_signal_encoding(signal_idx, signal_encoding, name);

        // now we can build the node data structure and store it
        let node = Var {
            parent,
            name,
            var_tpe: tpe,
            index,
            direction,
            signal_idx,
            enum_type,
            next: None,
            vhdl_type_name,
        };
        self.vars.push(node);
        // increment the child count of the parent
        self.increment_child_count(parent, name, var_id.into());
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
        debug_assert!(
            !sliced_signal.is_derived_signal(),
            "we can only slice a signal that actually exists, not a derived signal!"
        );
        debug_assert!(
            signal_ref.is_derived_signal(),
            "the signal that is derived from a slice needs to be marked as such"
        );
        debug_assert!(msb >= lsb);
        debug_assert!(!self.signal_derivations.contains_key(&signal_ref));
        let sliced_signal_enc = self.signal_encodings[sliced_signal.index()];
        self.signal_derivations.insert(
            signal_ref,
            DerivedBitVecSignal::new_slice(sliced_signal, sliced_signal_enc, msb, lsb),
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
        assert_eq!(std::mem::size_of::<ScopeOrVarRef>(), 8);

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
                + std::mem::size_of::<Option<VarIndex>>()   // index
                + std::mem::size_of::<SignalRef>()          // signal_idx
                + std::mem::size_of::<Option<EnumTypeId>>() // enum type
                + std::mem::size_of::<HierarchyStringId>()  // VHDL type name
                + std::mem::size_of::<Option<ScopeRef>>()   // parent
                + std::mem::size_of::<ScopeOrVarRef>() // next
        );
        // currently this all comes out to 40 bytes (~= 5x 64-bit pointers)
        assert_eq!(std::mem::size_of::<Var>(), 40);

        // Scope
        assert_eq!(
            std::mem::size_of::<Scope>(),
            std::mem::size_of::<HierarchyStringId>() // name
                + std::mem::size_of::<HierarchyStringId>() // component name
                + 1 // tpe
                + 1 // packed info
                + 4 // source info
                + 4 // source info
                + std::mem::size_of::<ScopeOrVarRef>() // child
                + std::mem::size_of::<ScopeRef>() // parent
                + std::mem::size_of::<ScopeOrVarRef>() // next
                + 2 // padding
        );
        // currently this all comes out to 40 bytes (= 5x 64-bit pointers)
        assert_eq!(std::mem::size_of::<Scope>(), 40);

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
