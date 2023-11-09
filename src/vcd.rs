// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use crate::hierarchy::*;
use crate::Waveform;
use rayon::prelude::*;
use std::fmt::{Debug, Formatter};
use std::io::{BufRead, Seek};

#[derive(Debug)]
pub struct WaveformError {}
pub type Result<T> = std::result::Result<T, WaveformError>;

pub fn read(filename: &str) -> Result<Waveform> {
    read_file_internal(filename, true)
}

pub fn read_single_thread(filename: &str) -> Result<Waveform> {
    read_file_internal(filename, false)
}

pub fn read_from_bytes(bytes: &[u8]) -> Result<Waveform> {
    read_bytes_internal(bytes, true)
}

pub fn read_from_bytes_single_thread(bytes: &[u8]) -> Result<Waveform> {
    read_bytes_internal(bytes, false)
}

fn read_file_internal(filename: &str, multi_threaded: bool) -> Result<Waveform> {
    // load file into memory (lazily)
    let input_file = std::fs::File::open(filename).expect("failed to open input file!");
    let mmap = unsafe { memmap2::Mmap::map(&input_file).expect("failed to memory map file") };
    let (header_len, hierarchy) = read_hierarchy(&mut std::io::Cursor::new(&mmap[..]));
    let wave_mem = read_values(&mmap[header_len..], multi_threaded, &hierarchy);
    Ok(Waveform::new(hierarchy, wave_mem))
}

fn read_bytes_internal(bytes: &[u8], multi_threaded: bool) -> Result<Waveform> {
    let (header_len, hierarchy) = read_hierarchy(&mut std::io::Cursor::new(&bytes));
    let wave_mem = read_values(&bytes[header_len..], multi_threaded, &hierarchy);
    Ok(Waveform::new(hierarchy, wave_mem))
}

// fn print_stats() {
//     println!(
//         "The full VCD data takes up  {} bytes in memory.",
//         ByteSize::b(wave_mem.size_in_memory() as u64)
//     );
//     println!(
//         "The size of the VCD file is {} bytes on disk.",
//         ByteSize::b((input_size - header_len) as u64)
//     );
//     wave_mem.print_statistics();
// }

fn read_hierarchy(input: &mut (impl BufRead + Seek)) -> (usize, Hierarchy) {
    let start = input.stream_position().unwrap();
    let mut h = HierarchyBuilder::default();

    let foo = |cmd: HeaderCmd| match cmd {
        HeaderCmd::Scope(tpe, name) => {
            h.add_scope(
                std::str::from_utf8(name).unwrap().to_string(),
                convert_scope_tpe(tpe),
            );
        }
        HeaderCmd::UpScope => h.pop_scope(),
        HeaderCmd::ScalarVar(tpe, size, id, name) => h.add_var(
            std::str::from_utf8(name).unwrap().to_string(),
            convert_var_tpe(tpe),
            VarDirection::Todo,
            u32::from_str_radix(std::str::from_utf8(size).unwrap(), 10).unwrap(),
            SignalRef::from_index(id_to_int(id).unwrap() as usize).unwrap(),
        ),
        HeaderCmd::VectorVar(tpe, size, id, name, _) => {
            let length = match u32::from_str_radix(std::str::from_utf8(size).unwrap(), 10) {
                Ok(len) => len,
                Err(_) => {
                    panic!(
                        "Failed to parse length: {} for {}",
                        String::from_utf8_lossy(size),
                        String::from_utf8_lossy(name)
                    );
                }
            };
            h.add_var(
                std::str::from_utf8(name).unwrap().to_string(),
                convert_var_tpe(tpe),
                VarDirection::Todo,
                length,
                SignalRef::from_index(id_to_int(id).unwrap() as usize).unwrap(),
            );
        }
        HeaderCmd::Date(value) => {
            h.set_date(String::from_utf8_lossy(value).to_string());
        }
        HeaderCmd::Version(value) => {
            h.set_version(String::from_utf8_lossy(value).to_string());
        }
        HeaderCmd::Comment(value) => {
            h.add_comment(String::from_utf8_lossy(value).to_string());
        }
        HeaderCmd::Timescale(factor, unit) => {
            let factor_int = u32::from_str_radix(std::str::from_utf8(factor).unwrap(), 10).unwrap();
            let value = Timescale::new(factor_int, convert_timescale_unit(unit));
            h.set_timescale(value);
        }
    };

    read_header(input, foo).unwrap();
    let end = input.stream_position().unwrap();
    let hierarchy = h.finish();
    ((end - start) as usize, hierarchy)
}

fn convert_timescale_unit(name: &[u8]) -> TimescaleUnit {
    match name {
        b"fs" => TimescaleUnit::FemtoSeconds,
        b"ps" => TimescaleUnit::PicoSeconds,
        b"ns" => TimescaleUnit::NanoSeconds,
        b"us" => TimescaleUnit::MicroSeconds,
        b"ms" => TimescaleUnit::MilliSeconds,
        b"s" => TimescaleUnit::Seconds,
        _ => TimescaleUnit::Unknown,
    }
}

fn convert_scope_tpe(tpe: &[u8]) -> ScopeType {
    match tpe {
        b"module" => ScopeType::Module,
        _ => ScopeType::Todo,
    }
}

fn convert_var_tpe(tpe: &[u8]) -> VarType {
    match tpe {
        b"wire" => VarType::Wire,
        b"reg" => VarType::Reg,
        b"string" => VarType::String,
        _ => panic!("TODO: convert {}", String::from_utf8_lossy(tpe)),
    }
}

const ID_CHAR_MIN: u8 = b'!';
const ID_CHAR_MAX: u8 = b'~';
const NUM_ID_CHARS: u64 = (ID_CHAR_MAX - ID_CHAR_MIN + 1) as u64;

/// Copied from https://github.com/kevinmehall/rust-vcd, licensed under MIT
#[inline]
fn id_to_int(id: &[u8]) -> Option<u64> {
    if id.is_empty() {
        return None;
    }
    let mut result = 0u64;
    for &i in id.iter().rev() {
        if !(ID_CHAR_MIN..=ID_CHAR_MAX).contains(&i) {
            return None;
        }
        let c = ((i - ID_CHAR_MIN) as u64) + 1;
        result = match result
            .checked_mul(NUM_ID_CHARS)
            .and_then(|x| x.checked_add(c))
        {
            None => return None,
            Some(value) => value,
        };
    }
    Some(result - 1)
}

/// very hacky read header implementation, will fail on a lot of valid headers
fn read_header(
    input: &mut impl BufRead,
    mut callback: impl FnMut(HeaderCmd),
) -> std::io::Result<()> {
    let mut buf: Vec<u8> = Vec::with_capacity(128);
    loop {
        buf.clear();
        let (cmd, body) = read_command(input, &mut buf)?;
        let parsed = match cmd {
            VcdCmd::Scope => {
                let tokens = find_tokens(body);
                HeaderCmd::Scope(tokens[0], tokens[1])
            }
            VcdCmd::Var => {
                let tokens = find_tokens(body);
                match tokens.len() {
                    4 => HeaderCmd::ScalarVar(tokens[0], tokens[1], tokens[2], tokens[3]),
                    5 => {
                        HeaderCmd::VectorVar(tokens[0], tokens[1], tokens[2], tokens[3], tokens[4])
                    }
                    _ => panic!(
                        "Unexpected var declaration: {}",
                        std::str::from_utf8(&buf).unwrap()
                    ),
                }
            }
            VcdCmd::UpScope => HeaderCmd::UpScope,
            VcdCmd::Date => HeaderCmd::Date(body),
            VcdCmd::Comment => HeaderCmd::Comment(body),
            VcdCmd::Version => HeaderCmd::Version(body),
            VcdCmd::Timescale => {
                let tokens = find_tokens(body);
                let (factor, unit) = match tokens.len() {
                    1 => {
                        // find the first non-numeric character
                        let token = tokens[0];
                        match token.iter().position(|c| *c < b'0' || *c > b'9') {
                            None => (token, &[] as &[u8]),
                            Some(pos) => (&token[..pos], &token[pos..]),
                        }
                    }
                    2 => (tokens[0], tokens[1]),
                    _ => panic!(
                        "Unexpected number of tokens for timescale: {}",
                        iter_bytes_to_list_str(tokens.iter())
                    ),
                };
                HeaderCmd::Timescale(factor, unit)
            }
            VcdCmd::EndDefinitions => {
                // header is done
                return Ok(());
            }
        };
        (callback)(parsed);
    }
}

const VCD_DATE: &[u8] = b"date";
const VCD_TIMESCALE: &[u8] = b"timescale";
const VCD_VAR: &[u8] = b"var";
const VCD_SCOPE: &[u8] = b"scope";
const VCD_UP_SCOPE: &[u8] = b"upscope";
const VCD_COMMENT: &[u8] = b"comment";
const VCD_VERSION: &[u8] = b"version";
const VCD_END_DEFINITIONS: &[u8] = b"enddefinitions";
const VCD_COMMANDS: [&[u8]; 8] = [
    VCD_DATE,
    VCD_TIMESCALE,
    VCD_VAR,
    VCD_SCOPE,
    VCD_UP_SCOPE,
    VCD_COMMENT,
    VCD_VERSION,
    VCD_END_DEFINITIONS,
];

/// Used to show all commands when printing an error message.
fn get_vcd_command_str() -> String {
    iter_bytes_to_list_str(VCD_COMMANDS.iter())
}

fn iter_bytes_to_list_str<'a, I>(bytes: I) -> String
where
    I: Iterator<Item = &'a &'a [u8]>,
{
    bytes
        .map(|c| String::from_utf8_lossy(c))
        .collect::<Vec<_>>()
        .join(", ")
}

#[derive(Debug, PartialEq)]
enum VcdCmd {
    Date,
    Timescale,
    Var,
    Scope,
    UpScope,
    Comment,
    Version,
    EndDefinitions,
}

impl VcdCmd {
    fn from_bytes(name: &[u8]) -> Option<Self> {
        match name {
            VCD_VAR => Some(VcdCmd::Var),
            VCD_SCOPE => Some(VcdCmd::Scope),
            VCD_UP_SCOPE => Some(VcdCmd::UpScope),
            VCD_DATE => Some(VcdCmd::Date),
            VCD_TIMESCALE => Some(VcdCmd::Timescale),
            VCD_COMMENT => Some(VcdCmd::Comment),
            VCD_VERSION => Some(VcdCmd::Version),
            VCD_END_DEFINITIONS => Some(VcdCmd::EndDefinitions),
            _ => None,
        }
    }

    fn from_bytes_or_panic(name: &[u8]) -> Self {
        match Self::from_bytes(name) {
            None => {
                panic!(
                    "Unexpected VCD command {}. Supported commands are: {:?}",
                    String::from_utf8_lossy(name),
                    get_vcd_command_str()
                );
            }
            Some(cmd) => cmd,
        }
    }
}

/// Reads in a command until the `$end`. Uses buf to store the read data.
/// Returns the name and the body of the command.
fn read_command<'a>(
    input: &mut impl BufRead,
    buf: &'a mut Vec<u8>,
) -> std::io::Result<(VcdCmd, &'a [u8])> {
    // start out with an empty buffer
    assert!(buf.is_empty());

    // skip over any preceding whitespace
    let start_char = skip_whitespace(input)?;
    assert_eq!(
        start_char,
        b'$',
        "Expected `$` but got `{}`",
        String::from_utf8_lossy(&[start_char])
    );

    // read the rest of the command into the buffer
    read_token(input, buf)?;

    // check to see if this is a valid command
    let cmd = VcdCmd::from_bytes_or_panic(&buf);
    buf.clear();

    // read until we find the end token
    read_until_end_token(input, buf)?;

    // return the name and body of the command
    Ok((cmd, &buf[..]))
}

#[inline]
fn find_tokens(line: &[u8]) -> Vec<&[u8]> {
    line.split(|c| matches!(*c, b' '))
        .filter(|e| !e.is_empty())
        .collect()
}

#[inline]
fn read_until_end_token(input: &mut impl BufRead, buf: &mut Vec<u8>) -> std::io::Result<()> {
    // count how many characters of the $end token we have recognized
    let mut end_index = 0;
    // we skip any whitespace at the beginning, but not between tokens
    let mut skipping_preceding_whitespace = true;
    loop {
        let byte = read_byte(input)?;
        if skipping_preceding_whitespace {
            match byte {
                b' ' | b'\n' | b'\r' | b'\t' => {
                    continue;
                }
                _ => {
                    skipping_preceding_whitespace = false;
                }
            }
        }
        // we always append and then later drop the `$end` bytes.
        buf.push(byte);
        end_index = match (end_index, byte) {
            (0, b'$') => 1,
            (1, b'e') => 2,
            (2, b'n') => 3,
            (3, b'd') => {
                // we are done!
                buf.truncate(buf.len() - 4); // drop $end
                right_strip(buf);
                return Ok(());
            }
            _ => 0, // reset
        };
    }
}

#[inline]
fn read_token(input: &mut impl BufRead, buf: &mut Vec<u8>) -> std::io::Result<()> {
    loop {
        let byte = read_byte(input)?;
        match byte {
            b' ' | b'\n' | b'\r' | b'\t' => {
                return Ok(());
            }
            other => {
                buf.push(other);
            }
        }
    }
}

/// Advances the input until the first non-whitespace character which is then returned.
#[inline]
fn skip_whitespace(input: &mut impl BufRead) -> std::io::Result<u8> {
    loop {
        let byte = read_byte(input)?;
        match byte {
            b' ' | b'\n' | b'\r' | b'\t' => {}
            other => return Ok(other),
        }
    }
}

#[inline]
fn read_byte(input: &mut impl BufRead) -> std::io::Result<u8> {
    let mut buf = [0u8; 1];
    input.read_exact(&mut buf)?;
    Ok(buf[0])
}

#[inline]
fn right_strip(buf: &mut Vec<u8>) {
    while !buf.is_empty() {
        match buf.last().unwrap() {
            b' ' | b'\n' | b'\r' | b'\t' => buf.pop(),
            _ => break,
        };
    }
}

enum HeaderCmd<'a> {
    Date(&'a [u8]),
    Version(&'a [u8]),
    Comment(&'a [u8]),
    Timescale(&'a [u8], &'a [u8]), // factor, unit
    Scope(&'a [u8], &'a [u8]),     // tpe, name
    UpScope,
    ScalarVar(&'a [u8], &'a [u8], &'a [u8], &'a [u8]), // tpe, size, id, name
    VectorVar(&'a [u8], &'a [u8], &'a [u8], &'a [u8], &'a [u8]), // tpe, size, id, name, vector def
}

/// The minimum number of bytes we want to read per thread.
const MIN_CHUNK_SIZE: usize = 8 * 1024;

#[inline]
pub(crate) fn usize_div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

#[inline]
pub(crate) fn u32_div_ceil(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

/// Returns starting byte and read length for every thread. Note that read-length is just an
/// approximation and the thread might have to read beyond or might also run out of data before
/// reaching read length.
#[inline]
fn determine_thread_chunks(body_len: usize) -> Vec<(usize, usize)> {
    let max_threads = rayon::current_num_threads();
    let number_of_threads_for_min_chunk_size = usize_div_ceil(body_len, MIN_CHUNK_SIZE);
    let num_threads = std::cmp::min(max_threads, number_of_threads_for_min_chunk_size);
    let chunk_size = usize_div_ceil(body_len, num_threads);
    // TODO: for large file it might make sense to have more chunks than threads
    (0..num_threads)
        .map(|ii| (ii * chunk_size, chunk_size))
        .collect()
}

/// Reads the body of a VCD with multiple threads
fn read_values(
    input: &[u8],
    multi_threaded: bool,
    hierarchy: &Hierarchy,
) -> Box<crate::wavemem::Reader> {
    if multi_threaded {
        let chunks = determine_thread_chunks(input.len());
        let encoders: Vec<crate::wavemem::Encoder> = chunks
            .par_iter()
            .map(|(start, len)| {
                let is_first = *start == 0;
                read_single_stream_of_values(&input[*start..], *len - 1, is_first, hierarchy)
            })
            .collect();

        // combine encoders
        let mut encoder_iter = encoders.into_iter();
        let mut encoder = encoder_iter.next().unwrap();
        for other in encoder_iter {
            encoder.append(other);
        }
        Box::new(encoder.finish())
    } else {
        let encoder = read_single_stream_of_values(input, input.len() - 1, true, hierarchy);
        Box::new(encoder.finish())
    }
}

fn read_single_stream_of_values<'a>(
    input: &[u8],
    stop_pos: usize,
    is_first: bool,
    hierarchy: &Hierarchy,
) -> crate::wavemem::Encoder {
    let mut encoder = crate::wavemem::Encoder::new(hierarchy);

    let input2 = if is_first {
        input
    } else {
        advance_to_first_newline(input)
    };
    let mut reader = BodyReader::new(input2);
    // We only start recording once we have encountered out first time step
    let mut found_first_time_step = false;
    // In the first thread, we might encounter a dump values which dumps all initial values
    // without specifying a timestamp
    if is_first {
        encoder.time_change(0);
        found_first_time_step = true;
    }
    loop {
        if let Some((pos, cmd)) = reader.next() {
            if pos > stop_pos {
                if let BodyCmd::Time(_) = cmd {
                    break; // stop before the next time value when we go beyond the stop position
                }
            }
            match cmd {
                BodyCmd::Time(value) => {
                    found_first_time_step = true;
                    let int_value =
                        u64::from_str_radix(std::str::from_utf8(value).unwrap(), 10).unwrap();
                    encoder.time_change(int_value);
                }
                BodyCmd::Value(value, id) => {
                    if found_first_time_step {
                        encoder.vcd_value_change(id_to_int(id).unwrap(), value);
                    }
                }
            };
        } else {
            break; // done, no more values to read
        }
    }

    encoder
}

#[inline]
fn advance_to_first_newline(input: &[u8]) -> &[u8] {
    for (pos, byte) in input.iter().enumerate() {
        match *byte {
            b'\n' => {
                return &input[pos..];
            }
            _ => {}
        }
    }
    &[] // no whitespaces found
}

struct BodyReader<'a> {
    input: &'a [u8],
    // state
    pos: usize,
    // statistics
    lines_read: usize,
}

impl<'a> BodyReader<'a> {
    fn new(input: &'a [u8]) -> Self {
        BodyReader {
            input,
            pos: 0,
            lines_read: 0,
        }
    }

    #[inline]
    fn try_finish_token(
        &mut self,
        pos: usize,
        token_start: &mut Option<usize>,
        prev_token: &mut Option<&'a [u8]>,
    ) -> Option<BodyCmd<'a>> {
        match *token_start {
            None => None,
            Some(start) => {
                let token = &self.input[start..pos];
                if token.is_empty() {
                    return None;
                }
                let ret = match *prev_token {
                    None => {
                        if token.len() == 1 {
                            // too short
                            return None;
                        }
                        // 1-token commands are binary changes or time commands
                        match token[0] {
                            b'#' => Some(BodyCmd::Time(&token[1..])),
                            b'0' | b'1' | b'z' | b'Z' | b'x' | b'X' => {
                                Some(BodyCmd::Value(&token[0..1], &token[1..]))
                            }
                            _ => {
                                if token != b"$dumpvars" && token != b"$end" {
                                    // ignore dumpvars and end command
                                    *prev_token = Some(token);
                                }
                                None
                            }
                        }
                    }
                    Some(first) => {
                        let cmd = match first[0] {
                            b'b' | b'B' | b'r' | b'R' | b's' | b'S' => {
                                BodyCmd::Value(&first[0..], token)
                            }
                            _ => {
                                panic!(
                                    "Unexpected tokens: `{}` and `{}` ({} lines after header)",
                                    String::from_utf8_lossy(first),
                                    String::from_utf8_lossy(token),
                                    self.lines_read
                                );
                            }
                        };
                        *prev_token = None;
                        Some(cmd)
                    }
                };
                *token_start = None;
                ret
            }
        }
    }
}

impl<'a> Iterator for BodyReader<'a> {
    type Item = (usize, BodyCmd<'a>);

    #[inline]
    fn next(&mut self) -> Option<(usize, BodyCmd<'a>)> {
        if self.pos >= self.input.len() {
            return None; // done!
        }
        let mut token_start: Option<usize> = None;
        let mut prev_token: Option<&'a [u8]> = None;
        let mut pending_lines = 0;
        for (offset, b) in self.input[self.pos..].iter().enumerate() {
            let pos = self.pos + offset;
            match b {
                b' ' | b'\n' | b'\r' | b'\t' => {
                    if token_start.is_none() {
                        if *b == b'\n' {
                            self.lines_read += 1;
                        }
                    } else {
                        match self.try_finish_token(pos, &mut token_start, &mut prev_token) {
                            None => {
                                if *b == b'\n' {
                                    pending_lines += 1;
                                }
                            }
                            Some(cmd) => {
                                // save state
                                self.pos = pos;
                                self.lines_read += pending_lines;
                                if *b == b'\n' {
                                    self.lines_read += 1;
                                }
                                return Some((pos, cmd));
                            }
                        }
                    }
                }
                _ => match token_start {
                    None => {
                        token_start = Some(pos);
                    }
                    Some(_) => {}
                },
            }
        }
        // update final position
        self.pos = self.input.len();
        // check to see if there is a final token at the end
        match self.try_finish_token(self.pos, &mut token_start, &mut prev_token) {
            None => {}
            Some(cmd) => {
                return Some((self.pos, cmd));
            }
        }
        // now we are done
        None
    }
}

enum BodyCmd<'a> {
    Time(&'a [u8]),
    Value(&'a [u8], &'a [u8]),
}

impl<'a> Debug for BodyCmd<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BodyCmd::Time(value) => {
                write!(f, "Time({})", String::from_utf8_lossy(value))
            }
            BodyCmd::Value(value, id) => {
                write!(
                    f,
                    "Value({}, {})",
                    String::from_utf8_lossy(id),
                    String::from_utf8_lossy(value)
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_body_to_vec(input: &[u8]) -> Vec<String> {
        let mut out = Vec::new();
        let reader = BodyReader::new(input);
        for (_, cmd) in reader {
            let desc = match cmd {
                BodyCmd::Time(value) => {
                    format!("Time({})", std::str::from_utf8(value).unwrap())
                }
                BodyCmd::Value(value, id) => {
                    format!(
                        "{} = {}",
                        std::str::from_utf8(id).unwrap(),
                        std::str::from_utf8(value).unwrap()
                    )
                }
            };
            out.push(desc);
        }
        out
    }

    #[test]
    fn test_read_body() {
        let input = r#"
1I,!
1J,!
1#2!
#2678437829
b00 D2!
b0000 d2!
b11 e2!
b00000 f2!
b10100 g2!
b00000 h2!
b00000 i2!
x(i"
x'i"
x&i"
x%i"
0j2!"#;
        let expected = vec![
            "I,! = 1",
            "J,! = 1",
            "#2! = 1",
            "Time(2678437829)",
            "D2! = b00",
            "d2! = b0000",
            "e2! = b11",
            "f2! = b00000",
            "g2! = b10100",
            "h2! = b00000",
            "i2! = b00000",
            "(i\" = x",
            "'i\" = x",
            "&i\" = x",
            "%i\" = x",
            "j2! = 0",
        ];
        let res = read_body_to_vec(&mut input.as_bytes());
        assert_eq!(res, expected);
    }

    #[test]
    fn test_read_command() {
        let mut buf = Vec::with_capacity(128);
        let input_0 = b"$upscope $end";
        let (cmd_0, body_0) = read_command(&mut input_0.as_slice(), &mut buf).unwrap();
        assert_eq!(cmd_0, VcdCmd::UpScope);
        assert!(body_0.is_empty());

        // test with more whitespace
        buf.clear();
        let input_1 = b" \t $upscope \n $end  \n ";
        let (cmd_1, body_1) = read_command(&mut input_1.as_slice(), &mut buf).unwrap();
        assert_eq!(cmd_1, VcdCmd::UpScope);
        assert!(body_1.is_empty());
    }
}
