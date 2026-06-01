// Copyright 2023-2024 The Regents of the University of California
// Copyright 2024-2026 Cornell University
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@cornell.edu>

use crate::FileFormat;
use crate::fst::{Attribute, parse_var_attributes};
use crate::signal::DerivedBitVecSignal;
use crate::vcd::{ScopeNames, parse_name};
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
    // some parameters are also events
    EventParameter,
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

impl SignalRef {
    #[inline]
    pub fn from_index(index: usize) -> Option<Self> {
        NonZeroU32::new(index as u32 + 1).map(Self)
    }

    #[inline]
    pub fn index(&self) -> usize {
        (self.0.get() - 1) as usize
    }

    pub const MAX: Self = Self(NonZeroU32::MAX);
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
    BitVector(u32),
}

/// Internal representation of everything we need to know about a signal in order to decode it.
/// A instead of a enum in order to have more fine-grained control over the size in memory.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
struct SignalInfo {
    kind: SignalKind,
    tpe: SignalEncodingType,
    bits: u32,
}

impl Default for SignalInfo {
    #[inline]
    fn default() -> Self {
        // only `kind` matters
        Self {
            kind: SignalKind::None,
            tpe: SignalEncodingType::BitVector,
            bits: 0,
        }
    }
}

impl SignalInfo {
    #[inline]
    fn derived(enc: SignalEncoding) -> Self {
        let (tpe, bits) = Self::encoding_to_tpe_and_bits(enc);
        Self {
            kind: SignalKind::Derived,
            tpe,
            bits,
        }
    }

    #[inline]
    fn encoding_to_tpe_and_bits(enc: SignalEncoding) -> (SignalEncodingType, u32) {
        match enc {
            SignalEncoding::String => (SignalEncodingType::String, 0),
            SignalEncoding::Real => (SignalEncodingType::Real, 64),
            SignalEncoding::BitVector(len) => (SignalEncodingType::BitVector, len),
        }
    }

    #[inline]
    fn is_none(&self) -> bool {
        self.kind == SignalKind::None
    }

    #[inline]
    fn update_encoding_ground(&mut self, enc: SignalEncoding) {
        self.update_encoding_internal(enc, SignalKind::Ground)
    }

    #[inline]
    fn update_encoding_internal(&mut self, enc: SignalEncoding, kind: SignalKind) {
        debug_assert!(kind != SignalKind::None);
        let (tpe, bits) = Self::encoding_to_tpe_and_bits(enc);
        debug_assert!(
            self.is_none() || (self.tpe == tpe && self.bits == bits),
            "Trying to redefine signal encoding: {:?} -> {:?}",
            enc,
            SignalEncoding::from(*self),
        );
        self.tpe = tpe;
        self.bits = bits;
        self.kind = kind;
    }

    #[inline]
    fn ref_from_index(&self, index: usize) -> Option<SignalRef> {
        match self.kind {
            SignalKind::None => None,
            SignalKind::Ground => SignalRef::from_index(index),
            SignalKind::Derived => SignalRef::from_index(index),
        }
    }

    #[inline]
    fn is_derived_signal(&self) -> bool {
        self.kind == SignalKind::Derived
    }
}

impl From<SignalInfo> for SignalEncoding {
    fn from(info: SignalInfo) -> Self {
        match info.tpe {
            SignalEncodingType::String => SignalEncoding::String,
            SignalEncodingType::Real => SignalEncoding::Real,
            SignalEncodingType::BitVector => SignalEncoding::BitVector(info.bits),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
enum SignalKind {
    /// the signal does not exist
    None,
    /// the signal is part of the VCD/FST/GHW/... file that we parsed
    Ground,
    /// the signal is derived from ground signals
    Derived,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
enum SignalEncodingType {
    /// encoded as variable length strings
    String,
    /// encoded as 64-bit floating point values
    Real,
    /// encoded as a fixed width bit-vector
    BitVector,
}

impl SignalEncoding {
    pub fn length(&self) -> Option<u32> {
        match &self {
            SignalEncoding::String | SignalEncoding::Real => None,
            SignalEncoding::BitVector(len) => Some(*len),
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
    next: Option<ItemRef>,
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
        h.signals[self.signal_idx.index()].into()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub enum ItemRef {
    Scope(ScopeRef),
    Var(VarRef),
}

impl ItemRef {
    pub fn deref<'a>(&self, h: &'a Hierarchy) -> Item<'a> {
        h.get_item(*self)
    }

    pub fn name<'a>(&self, h: &'a Hierarchy) -> &'a str {
        h.get_item(*self).name(h)
    }
}

impl From<ScopeRef> for ItemRef {
    fn from(value: ScopeRef) -> Self {
        Self::Scope(value)
    }
}

impl From<VarRef> for ItemRef {
    fn from(value: VarRef) -> Self {
        Self::Var(value)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Item<'a> {
    Scope(&'a Scope),
    Var(&'a Var),
}

impl<'a> Item<'a> {
    #[inline]
    pub fn name<'b>(&self, h: &'b Hierarchy) -> &'b str {
        match self {
            Item::Scope(s) => s.name(h),
            Item::Var(v) => v.name(h),
        }
    }

    #[inline]
    fn child(&self) -> Option<ItemRef> {
        match self {
            Item::Scope(s) => s.child,
            Item::Var(_) => None,
        }
    }

    #[inline]
    fn next(&self) -> Option<ItemRef> {
        match self {
            Item::Scope(s) => s.next,
            Item::Var(v) => v.next,
        }
    }

    #[inline]
    fn parent(&self) -> Option<ScopeRef> {
        match self {
            Item::Scope(s) => s.parent,
            Item::Var(v) => v.parent,
        }
    }
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
    child: Option<ItemRef>,
    parent: Option<ScopeRef>,
    next: Option<ItemRef>,
}

impl Scope {
    /// Local name of the scope.
    #[inline]
    pub fn name<'a>(&self, hierarchy: &'a Hierarchy) -> &'a str {
        &hierarchy[self.name]
    }

    /// Local name of the component, e.g., the name of the module that was instantiated.
    #[inline]
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

    #[inline]
    pub fn scope_type(&self) -> ScopeType {
        self.tpe
    }

    #[inline]
    pub fn pack_info(&self) -> Option<ScopePackInfo> {
        self.pack
    }

    #[inline]
    pub fn is_array(&self) -> bool {
        matches!(self.tpe, ScopeType::VhdlArray | ScopeType::SvArray)
    }

    #[inline]
    pub fn is_packed_array(&self) -> bool {
        self.is_array() && self.is_packed()
    }

    #[inline]
    pub fn is_unpacked_array(&self) -> bool {
        self.is_array() && self.is_unpacked()
    }

    #[inline]
    pub fn is_packed(&self) -> bool {
        matches!(
            self.pack,
            Some(ScopePackInfo::Packed) | Some(ScopePackInfo::TaggedPacked)
        )
    }

    #[inline]
    pub fn is_unpacked(&self) -> bool {
        matches!(self.pack, Some(ScopePackInfo::Unpacked))
    }

    #[inline]
    pub fn is_record(&self) -> bool {
        matches!(self.tpe, ScopeType::Struct | ScopeType::VhdlRecord)
    }

    #[inline]
    pub fn is_packed_record(&self) -> bool {
        self.is_record() && self.is_packed()
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
    ) -> impl Iterator<Item =ItemRef> + 'a {
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
    item: Option<ItemRef>,
    is_first: bool,
}

impl<'a> HierarchyItemIdIterator<'a> {
    fn new(hierarchy: &'a Hierarchy, item: Option<ItemRef>) -> Self {
        Self {
            hierarchy,
            item,
            is_first: true,
        }
    }

    fn get_next(&self, item: ItemRef) -> Option<ItemRef> {
        match self.hierarchy.get_item(item) {
            Item::Scope(scope) => scope.next,
            Item::Var(var) => var.next,
        }
    }
}

impl Iterator for HierarchyItemIdIterator<'_> {
    type Item = ItemRef;

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

#[inline]
fn to_var_ref_iterator(iter: impl Iterator<Item =ItemRef>) -> impl Iterator<Item = VarRef> {
    iter.flat_map(|i| match i {
        ItemRef::Scope(_) => None,
        ItemRef::Var(v) => Some(v),
    })
}

fn to_scope_ref_iterator(
    iter: impl Iterator<Item =ItemRef>,
) -> impl Iterator<Item = ScopeRef> {
    iter.flat_map(|i| match i {
        ItemRef::Scope(s) => Some(s),
        ItemRef::Var(_) => None,
    })
}


struct HierarchyItemIdRecursiveIterator<'a> {
    hierarchy: &'a Hierarchy,
    parent: Option<ScopeRef>,
    current: Option<ItemRef>,
    done: bool,
}

impl<'a> HierarchyItemIdRecursiveIterator<'a> {
    fn new(hierarchy: &'a Hierarchy, parent: Option<ScopeRef>) -> Self {
        Self {
            hierarchy,
            parent,
            current: None,
            done: false,
        }
    }
}

impl Iterator for HierarchyItemIdRecursiveIterator<'_> {
    type Item = ItemRef;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        if let Some(prev) = self.current {
            // find next item
            let prev = &self.hierarchy.get_item(prev);
            if let Some(next) = prev.next() {
                self.current = Some(find_leaf(self.hierarchy, next));
            } else {
                self.current = prev.parent().map(|p| p.into());
            }
        } else {
            // first item
            if let Some(pp) = self.parent {
                self.current = Some(find_leaf(self.hierarchy, pp.into()));
            } else {
                self.current = self.hierarchy.first_item().map(|i| find_leaf(self.hierarchy, i));
            }
            // if the child does not exist, we are done
            self.done = self.current.is_none();
        }

        if self.current == self.parent.map(|p| p.into()) {
            self.done = true;
            self.current = None;
        }

        self.current
    }
}

fn find_leaf(h: &Hierarchy, mut item: ItemRef) -> ItemRef {
    while let Some(child) = h.get_item(item).child() {
        item = child;
    }
    item
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

#[derive(Clone)]
#[cfg_attr(feature = "serde1", derive(serde::Serialize, serde::Deserialize))]
pub struct Hierarchy {
    vars: Vec<Var>,
    scopes: Vec<Scope>,
    strings: Vec<String>,
    source_locs: Vec<SourceLoc>,
    enums: Vec<EnumType>,
    signals: Vec<SignalInfo>,
    meta: HierarchyMetaData,
    signal_derivations: FxHashMap<SignalRef, DerivedBitVecSignal>,
}

#[derive(Clone)]
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
    /// Recursively iterates over all scopes and variables in a depth first manner
    pub fn all_items(&self) -> impl Iterator<Item =ItemRef> + '_ {
        HierarchyItemIdRecursiveIterator::new(self, None)
    }

    /// Recursively iterates over all variables (at all levels).
    pub fn all_vars(&self) -> impl Iterator<Item = VarRef> + '_ {
        to_var_ref_iterator(self.all_items())
    }

    /// Recursively iterates over all scopes (at all levels).
    pub fn all_scopes(&self) -> impl Iterator<Item = ScopeRef> + '_ {
        to_scope_ref_iterator(self.all_items())
    }

    /// Retrieves the first item inside the implicit fake top scope
    fn first_item(&self) -> Option<ItemRef> {
        debug_assert!(self.scopes[FAKE_TOP_SCOPE.index()].next.is_none());
        self.scopes[FAKE_TOP_SCOPE.index()].child
    }

    /// Returns an iterator over references to all top-level scopes and variables.
    pub fn items(&self) -> impl Iterator<Item =ItemRef> + '_ {
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

    /// Encoding of all ground signals, i.e., all signals that actually appear in the underlying file.
    pub fn ground_signal_encodings(&self) -> impl Iterator<Item = Option<SignalEncoding>> {
        self.signals.iter().map(|info| {
            if info.is_derived_signal() {
                None
            } else {
                Some((*info).into())
            }
        })
    }

    /// Iterate over all valid signal references.
    pub fn signals(&self) -> impl Iterator<Item = SignalRef> {
        self.signals
            .iter()
            .enumerate()
            .flat_map(|(ii, info)| info.ref_from_index(ii))
    }

    pub fn is_derived_signal(&self, signal: SignalRef) -> bool {
        self.signals[signal.index()].is_derived_signal()
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

/// Item access.
impl Hierarchy {
    pub fn lookup_item_by_name(&self, name: &str) -> Option<ItemRef> {
        lookup_item_by_name(self, self.items(), name)
    }

    pub fn lookup_item_in_scope_by_name(
        &self,
        root: ScopeRef,
        name: &str,
    ) -> Option<ItemRef> {
        lookup_item_by_name(self, self[root].items(self), name)
    }
}

/// Common implementation of name lookup
fn lookup_item_by_name(
    h: &Hierarchy,
    items: impl Iterator<Item =ItemRef>,
    suffix: &str,
) -> Option<ItemRef> {
    find_longest_match(h, items, suffix).and_then(|(remaining_suffix, item)| {
        if remaining_suffix.is_empty() {
            Some(item)
        } else if let ItemRef::Scope(scope) = item {
            lookup_item_by_name(h, h[scope].items(h), remaining_suffix)
        } else {
            // we matched a variable, but there is a suffix
            // TODO: see if we match an index or something like this
            Some(item)
        }
    })
}

fn find_longest_match<'a>(
    h: &Hierarchy,
    items: impl Iterator<Item =ItemRef>,
    suffix: &'a str,
) -> Option<(&'a str, ItemRef)> {
    let mut match_item = None;
    let mut match_suffix = suffix;
    let mut match_len = 0;
    for item in items {
        let name = item.name(h);
        let new_match_len = name.len();
        if new_match_len > match_len
            && let Some(new_suffix) = suffix.strip_prefix(item.name(h))
        {
            if new_suffix.is_empty() {
                return Some((new_suffix, item));
            } else if let Some(new_suffix) = new_suffix.strip_prefix(SCOPE_SEPARATOR) {
                match_len = new_match_len;
                match_item = Some(item);
                match_suffix = new_suffix;
            }
            // if there is a match that breaks somewhere inside a name, that would not be valid
        }
    }
    match_item.map(|i| (match_suffix, i))
}

impl Hierarchy {
    /// Retrieves the length of a signal identified by its id by looking up a
    /// variable that refers to the signal.
    pub fn get_signal_tpe(&self, signal_idx: SignalRef) -> Option<SignalEncoding> {
        self.signals.get(signal_idx.index()).map(|s| (*s).into())
    }

    pub fn get_derived_signal(&self, signal_idx: SignalRef) -> Option<&DerivedBitVecSignal> {
        self.signal_derivations.get(&signal_idx)
    }

    pub fn all_derived_signals(&self) -> impl Iterator<Item = (SignalRef, &DerivedBitVecSignal)> {
        self.signal_derivations.iter().map(|(k, v)| (*k, v))
    }

    pub fn has_derived_signals(&self) -> bool {
        !self.signal_derivations.is_empty()
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

    fn get_item(&self, id: ItemRef) -> Item<'_> {
        match id {
            ItemRef::Scope(id) => Item::Scope(&self[id]),
            ItemRef::Var(id) => Item::Var(&self[id]),
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

struct ScopeStackEntry {
    scope_id: usize,
    last_child: Option<ItemRef>,
    /// indicates that this scope is being flattened and all operations should be done on the parent instead
    flattened: bool,
}

pub struct HierarchyBuilder {
    vars: Vec<Var>,
    scopes: Vec<Scope>,
    scope_stack: Vec<ScopeStackEntry>,
    source_locs: Vec<SourceLoc>,
    enums: Vec<EnumType>,
    signals: Vec<SignalInfo>,
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
            signals: Vec::default(),
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
        self.signals.shrink_to_fit();
        self.signal_derivations.shrink_to_fit();
        #[cfg(debug_assertions)]
        {
            for (ii, info) in self.signals.iter().enumerate() {
                if !info.is_none() {
                    let signal = info.ref_from_index(ii).unwrap();
                    if info.is_derived_signal() {
                        debug_assert!(self.signal_derivations.contains_key(&signal));
                    } else {
                        debug_assert!(!self.signal_derivations.contains_key(&signal));
                    }
                }
            }
        }

        Hierarchy {
            vars: self.vars,
            scopes: self.scopes,
            strings: self.strings.into_iter().collect::<Vec<_>>(),
            source_locs: self.source_locs,
            enums: self.enums,
            meta: self.meta,
            signal_derivations: self.signal_derivations,
            signals: self.signals,
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
                let r = SignalRef::from_index(self.signals.len()).unwrap();
                self.signals.push(SignalInfo::derived(signal_enc));
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
    fn add_to_hierarchy_tree(&mut self, node_id: ItemRef) -> Option<ScopeRef> {
        let entry_pos = find_parent_scope(&self.scope_stack);
        let entry = &mut self.scope_stack[entry_pos];
        let parent = entry.scope_id;
        match entry.last_child {
            Some(ItemRef::Var(child)) => {
                // add pointer to new node from last child
                assert!(self.vars[child.index()].next.is_none());
                self.vars[child.index()].next = Some(node_id);
            }
            Some(ItemRef::Scope(child)) => {
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

    fn get_parent_and_prev_child(&self) -> (Option<ScopeRef>, Option<ItemRef>) {
        let entry_pos = find_parent_scope(&self.scope_stack);
        let entry = &self.scope_stack[entry_pos];
        let parent_ref = ScopeRef::from_index(entry.scope_id).unwrap();
        if parent_ref == FAKE_TOP_SCOPE {
            (None, entry.last_child)
        } else {
            (Some(parent_ref), entry.last_child)
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
                if let ItemRef::Scope(other) = item {
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

    fn get_next(&self, item: ItemRef) -> Option<ItemRef> {
        match item {
            ItemRef::Scope(scope_ref) => self.scopes[scope_ref.index()].next,
            ItemRef::Var(var_ref) => self.vars[var_ref.index()].next,
        }
    }

    fn find_last_child(&self, scope: ScopeRef) -> Option<ItemRef> {
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
        child_ref: ItemRef,
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
            if let ItemRef::Scope(child_scope_ref) = child_ref {
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
            if let ItemRef::Scope(child_scope) = item {
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
        // only bit vector signals with length greater than zero can be possibly merged
        if let SignalEncoding::BitVector(width) = signal_encoding
            && width > 0
            && index.width() == width
        {
            // lookup previous item
            if let (Some(parent), Some(ItemRef::Var(prev_var_ref))) =
                self.get_parent_and_prev_child()
            {
                // for unpacked arrays, we specifically do _not_ want to merge entries into a single
                // (packed) bit-vector
                if self.scopes[parent.index()].is_unpacked_array() {
                    return false;
                }

                let prev_var = &mut self.vars[prev_var_ref.index()];
                // does the name match? does the variable have an index?
                if prev_var.name == new_name
                    && let Some(prev_index) = prev_var.index
                {
                    // is the previous variable already derived?
                    if let Some(prev_derived) = self.var_to_derived.get_mut(&prev_var_ref) {
                        return Self::try_concat(
                            signal_idx,
                            width,
                            index,
                            prev_index,
                            prev_var,
                            prev_derived,
                        );
                    } else {
                        // check to make sure that we can actually concat this signal
                        let prev_enc = self.signals[prev_var.signal_ref().index()].into();
                        // only bit vector signals with length greater than zero can be possibly merged
                        if let SignalEncoding::BitVector(prev_width) = prev_enc
                            && prev_width > 0
                            && prev_index.width() == prev_width
                        {
                            let prev_derived =
                                self.var_to_derived.entry(prev_var_ref).or_insert_with(|| {
                                    DerivedBitVecSignal::new_identity(
                                        prev_var.signal_ref(),
                                        prev_enc,
                                    )
                                });
                            return Self::try_concat(
                                signal_idx,
                                width,
                                index,
                                prev_index,
                                prev_var,
                                prev_derived,
                            );
                        }
                    }
                }
            }
        }
        false
    }

    fn try_concat(
        signal_idx: SignalRef,
        signal_width: u32,
        index: VarIndex,
        prev_index: VarIndex,
        prev_var: &mut Var,
        prev_derived: &mut DerivedBitVecSignal,
    ) -> bool {
        let new_is_msb = index.lsb() == prev_index.msb() + 1;
        let new_is_lsb = prev_index.lsb() == index.msb() + 1;
        if new_is_msb | new_is_lsb {
            // modify existing variable (we assume that the other properties are the same
            if new_is_msb {
                prev_var.index = Some(VarIndex::new(index.msb(), prev_index.lsb()));
                prev_derived.concat_left_full(signal_idx, signal_width);
            } else {
                prev_var.index = Some(VarIndex::new(prev_index.msb(), index.lsb()));
                prev_derived.concat_right_full(signal_idx, signal_width);
            };
            debug_assert_eq!(prev_var.index.unwrap().width(), prev_derived.width());
            true
        } else {
            false
        }
    }

    // Verilator has the bad habit of expanding the full name for the final scalar in an array.
    // So instead of producing v_arru.[2] it will produce v_arru.v_arru[2]
    fn handle_verilator_array_element(
        &mut self,
        name_id: HierarchyStringId,
        index: Option<VarIndex>,
        array_scopes: &mut ScopeNames,
    ) -> (HierarchyStringId, Option<VarIndex>) {
        if let (Some(parent_id), _) = self.get_parent_and_prev_child() {
            let parent = &self.scopes[parent_id.index()];
            if parent.is_array() {
                let name = &self.strings[name_id.index()];
                let parent_name = &self.strings[parent.name.index()];
                let n1 = if let Some(suffix) = name.strip_prefix(parent_name) {
                    suffix
                } else {
                    name
                };

                // remove array scopes if they already exist
                let mut pp = if n1 == name {
                    Some(parent_id)
                } else {
                    parent.parent
                };
                while let Some(parent_id) = pp {
                    let parent = &self.scopes[parent_id.index()];
                    let parent_name = &self.strings[parent.name.index()];
                    if let Some(name) = array_scopes.last()
                        && name.as_ref() == parent_name
                    {
                        array_scopes.pop();
                        pp = parent.parent;
                    } else {
                        break;
                    }
                }

                // for unpacked array elements, the "index" is most likely the actual array index instead of a bit index
                if parent.is_unpacked_array()
                    && let Some(index) = index
                    && index.width() == 1
                {
                    let new_name = format!("{n1}[{}]", index.lsb());
                    let new_name_id = self.add_string(new_name.into());
                    return (new_name_id, None);
                } else if n1 != name {
                    let n1 = n1.to_string();
                    let new_name_id = self.add_string(n1.into());
                    return (new_name_id, index);
                }
            }
        }
        (name_id, index)
    }

    /// Adds a variable with a "raw" name directly from the VCD/FST
    /// This name will be further processed in order to extract index and possible array scopes.
    #[allow(clippy::too_many_arguments)]
    pub fn add_var_raw_name(
        &mut self,
        name: &[u8],
        length: u32,
        raw_tpe: VarType,
        attributes: &mut Vec<Attribute>,
        signal_encoding: SignalEncoding,
        direction: VarDirection,
        signal_idx: SignalRef,
    ) -> crate::vcd::Result<()> {
        let (var_name, index, mut scopes) = parse_name(name, length)?;
        let (type_name, var_type, enum_type) =
            parse_var_attributes(attributes, raw_tpe, &var_name)?;
        let vhdl_type_name = type_name.map(|s| self.add_string(s.into()));
        let name = self.add_string(var_name);

        // verilator specific variable name fixes
        let (name, index) = self.handle_verilator_array_element(name, index, &mut scopes);

        let num_scopes = scopes.len();
        self.add_array_scopes(scopes);

        self.add_var(
            name,
            var_type,
            signal_encoding,
            direction,
            index,
            signal_idx,
            enum_type,
            vhdl_type_name,
        );
        self.pop_scopes(num_scopes);
        Ok(())
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
        self.set_encoding_for_ground_signal(signal_idx, signal_encoding);

        if let Some(ii) = index
            && self.check_for_split_var(name, ii, signal_encoding, signal_idx)
        {
            // we merged with an existing variable, no need to add a new one
            return;
        }

        let node_id = self.vars.len();
        let var_id = VarRef::from_index(node_id).unwrap();
        let parent = self.add_to_hierarchy_tree(var_id.into());

        // ensure that 0-bit bit-vectors are always typed as event
        let tpe = match (signal_encoding, tpe) {
            (SignalEncoding::BitVector(0), VarType::Parameter) => VarType::EventParameter,
            (SignalEncoding::BitVector(0), _) => VarType::Event,
            _ => tpe,
        };

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
        debug_assert!(msb >= lsb);
        debug_assert!(!self.signal_derivations.contains_key(&signal_ref));
        let info = self.signals[sliced_signal.index()];
        if let SignalEncoding::BitVector(width) = info.into() {
            debug_assert!(width > 0 && msb < width);
            self.signal_derivations.insert(
                signal_ref,
                DerivedBitVecSignal::new_slice(sliced_signal, msb, lsb),
            );
            if let Some(info) = self.signals.get_mut(signal_ref.index()) {
                if let SignalEncoding::BitVector(width) = (*info).into() {
                    debug_assert_eq!(width, msb - lsb + 1);
                    info.kind = SignalKind::Derived;
                } else {
                    unreachable!("The result of a slice must be a bit-vector!");
                }
            }
        } else {
            unreachable!("Can only slice non-zero-width bit-vectors.")
        }
    }

    fn set_encoding_for_ground_signal(&mut self, signal: SignalRef, encoding: SignalEncoding) {
        let ii = signal.index();
        if self.signals.len() <= ii {
            self.signals.resize(ii + 1, SignalInfo::default());
        }
        self.signals[ii].update_encoding_ground(encoding);
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
        assert_eq!(std::mem::size_of::<ItemRef>(), 8);

        // 4 byte length + tag + padding
        assert_eq!(std::mem::size_of::<SignalEncoding>(), 8);
        // signal info size should be dominated by signal encoding size
        assert_eq!(
            std::mem::size_of::<SignalEncoding>(),
            std::mem::size_of::<SignalInfo>()
        );

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
                + std::mem::size_of::<ItemRef>() // next
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
                + std::mem::size_of::<ItemRef>() // child
                + std::mem::size_of::<ScopeRef>() // parent
                + std::mem::size_of::<ItemRef>() // next
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

    #[test]
    fn test_depth_first_iterator() {
        let mut b = HierarchyBuilder::new(FileFormat::Ghw);
        let s1 = b.add_string("s1".into());
        let s2 = b.add_string("s2".into());
        let v1 = b.add_string("v1".into());
        let v2 = b.add_string("v2".into());
        let v3 = b.add_string("v3".into());
        b.add_var(v1, VarType::Bit, SignalEncoding::BitVector(1), VarDirection::InOut, None, SignalRef::from_index(1).unwrap(), None, None);
        b.add_scope(s1, None, ScopeType::Struct, None, None, None, false);
        b.add_var(v1, VarType::Bit, SignalEncoding::BitVector(1), VarDirection::InOut, None, SignalRef::from_index(1).unwrap(), None, None);
        b.add_scope(s2, None, ScopeType::Struct, None, None, None, false);
        b.add_var(v2, VarType::Bit, SignalEncoding::BitVector(1), VarDirection::InOut, None, SignalRef::from_index(1).unwrap(), None, None);
        b.pop_scope();
        b.add_var(v3, VarType::Bit, SignalEncoding::BitVector(1), VarDirection::InOut, None, SignalRef::from_index(1).unwrap(), None, None);
        b.pop_scope();
        let h = b.finish();

        let names: Vec<_> = h.all_items().map(|i| h.get_item(i).name(&h).to_string()).collect();
        assert_eq!(names, ["v1", "v1", "v2", "s2", "v3", "s1"]);
    }
}
