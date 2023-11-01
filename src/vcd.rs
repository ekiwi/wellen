// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use crate::hierarchy::*;
use crate::signals::SignalSource;
use rayon::prelude::*;
use std::fmt::{Debug, Formatter};
use std::io::{BufRead, Seek};

pub fn read(filename: &str) -> (Hierarchy, Box<dyn SignalSource>) {
    read_internal(filename, true)
}

pub fn read_single_thread(filename: &str) -> (Hierarchy, Box<dyn SignalSource>) {
    read_internal(filename, false)
}

fn read_internal(filename: &str, multi_threaded: bool) -> (Hierarchy, Box<dyn SignalSource>) {
    // load file into memory (lazily)
    let input_file = std::fs::File::open(filename).expect("failed to open input file!");
    let mmap = unsafe { memmap2::Mmap::map(&input_file).expect("failed to memory map file") };
    let (header_len, hierarchy) = read_hierarchy(&mut std::io::Cursor::new(&mmap[..]));
    let wave_mem = read_values(&mmap[header_len..], multi_threaded, &hierarchy);
    (hierarchy, wave_mem)
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
            id_to_int(id).unwrap() as u32,
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
                id_to_int(id).unwrap() as u32,
            );
        }
    };

    read_header(input, foo).unwrap();
    let end = input.stream_position().unwrap();
    let hierarchy = h.finish();
    ((end - start) as usize, hierarchy)
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
        _ => VarType::Todo,
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
        let read = input.read_until(b'\n', &mut buf)?;
        if read == 0 {
            return Ok(());
        }
        truncate(&mut buf);
        if buf.is_empty() {
            continue;
        }

        // decode
        if buf.starts_with(b"$scope") {
            let parts: Vec<&[u8]> = line_to_tokens(&buf);
            assert_eq!(parts.last().unwrap(), b"$end");
            (callback)(HeaderCmd::Scope(parts[1], parts[2]));
        } else if buf.starts_with(b"$var") {
            let parts: Vec<&[u8]> = line_to_tokens(&buf);
            assert_eq!(parts.last().unwrap(), b"$end");
            match parts.len() - 2 {
                4 => (callback)(HeaderCmd::ScalarVar(parts[1], parts[2], parts[3], parts[4])),
                5 => {
                    (callback)(HeaderCmd::VectorVar(
                        parts[1], parts[2], parts[3], parts[4], parts[4],
                    ));
                }
                _ => panic!(
                    "Unexpected var declaration: {}",
                    std::str::from_utf8(&buf).unwrap()
                ),
            }
        } else if buf.starts_with(b"$upscope") {
            let parts: Vec<&[u8]> = line_to_tokens(&buf);
            assert_eq!(parts.last().unwrap(), b"$end");
            assert_eq!(parts.len(), 2);
            (callback)(HeaderCmd::UpScope);
        } else if buf.starts_with(b"$enddefinitions") {
            let parts: Vec<&[u8]> = line_to_tokens(&buf);
            assert_eq!(parts.last().unwrap(), b"$end");
            // header is done
            return Ok(());
        } else if buf.starts_with(b"$date")
            || buf.starts_with(b"$version")
            || buf.starts_with(b"$comment")
            || buf.starts_with(b"$timescale")
        {
            // ignored commands, just find the $end
            while !contains_end(&buf) {
                buf.clear();
                let read = input.read_until(b'\n', &mut buf)?;
                if read == 0 {
                    return Ok(());
                }
                truncate(&mut buf);
            }
        } else {
            panic!("Unexpected line: {}", std::str::from_utf8(&buf).unwrap());
        }
    }
}

#[inline]
fn line_to_tokens(line: &[u8]) -> Vec<&[u8]> {
    line.split(|c| *c == b' ')
        .filter(|e| !e.is_empty())
        .collect()
}

#[inline]
fn truncate(buf: &mut Vec<u8>) {
    while !buf.is_empty() {
        match buf.first().unwrap() {
            b' ' | b'\n' | b'\r' | b'\t' => buf.remove(0),
            _ => break,
        };
    }

    while !buf.is_empty() {
        match buf.last().unwrap() {
            b' ' | b'\n' | b'\r' | b'\t' => buf.pop(),
            _ => break,
        };
    }
}

#[inline]
fn contains_end(line: &[u8]) -> bool {
    let str_view = std::str::from_utf8(line).unwrap();
    str_view.contains("$end")
}

enum HeaderCmd<'a> {
    Scope(&'a [u8], &'a [u8]), // tpe, name
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
}
