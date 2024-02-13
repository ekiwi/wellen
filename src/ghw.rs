// Copyright 2024 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use crate::WellenError;

type Result<T> = std::result::Result<T, WellenError>;

fn load(filename: &str) -> Result<()> {
    let f = std::fs::File::open(filename)?;
    let mut input = std::io::BufReader::new(f);

    Ok(())
}

/// https://github.com/ghdl/ghdl/blob/7400f2449298d8d1ce988c059454ff5a03088915/src/grt/grt-rtis.ads#L33
#[repr(u8)]
#[derive(Debug, PartialEq, Copy, Clone)]
enum GhdlRtik {
    Top = 0,
    Library = 1,
    Package = 2,
    PackageBody = 3,
    Entity = 4,
    Architecture = 5,
    Process = 6,
    Block = 7,
    IfGenerate = 8,
    CaseGenerate = 9,
    ForGenerate = 10,
    GenerateBody = 11,
    Instance = 12,
    Constant = 13,
    Iterator = 14,
    Variable = 15,
    Signal = 16,
    File = 17,
    Port = 18,
    Generic = 19,
    Alias = 20,
    Guard = 21,
    Component = 22,
    Attribute = 23,
    TypeB1 = 24,
    TypeE8 = 25,
    TypeE32 = 26,
    TypeI32 = 27,
    TypeI64 = 28,
    TypeF64 = 29,
    TypeP32 = 30,
    TypeP64 = 31,
    TypeAccess = 32,
    TypeArray = 33,
    TypeRecord = 34,
    TypeUnboundedRecord = 35,
    TypeFile = 36,
    SubtypeScalar = 37,
    SubtypeArray = 38,
    SubtypeUnboundedArray = 39,
    SubtypeRecord = 40,
    SubtypeUnboundedRecord = 41,
    SubtypeAccess = 42,
    TypeProtected = 43,
    Element = 44,
    Unit64 = 45,
    Unitptr = 46,
    AttributeTransaction = 47,
    AttributeQuiet = 48,
    AttributeStable = 49,
    PslAssert = 50,
    PslAssume = 51,
    PslCover = 52,
    PslEndpoint = 53,
    Error = 54,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_load() {
        load("inputs/ghdl/tb_recv.ghw").unwrap();
    }
}
