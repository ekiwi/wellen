// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use crate::hierarchy::*;
use crate::values::*;
use rayon::prelude::*;
use std::fmt::{Debug, Formatter};
use std::io::{BufRead, Read, Seek, SeekFrom};

pub fn read(filename: &str) -> (Hierarchy, Values) {
    let (file_len, (hierarchy_len, hierarchy)) = {
        let input_file = std::fs::File::open(filename).expect("failed to open input file!");
        let file_len = input_file.metadata().unwrap().len();
        let mut input = std::io::BufReader::new(input_file);
        (file_len, read_hierarchy(&mut input))
    };
    let values = read_values_multi_threaded(filename, file_len, hierarchy_len, &hierarchy);
    (hierarchy, values)
}

fn read_hierarchy(input: &mut (impl BufRead + Seek)) -> (u64, Hierarchy) {
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
            id_to_int(id),
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
                id_to_int(id),
            );
        }
    };

    read_header(input, foo).unwrap();
    let end = input.stream_position().unwrap();
    h.print_statistics();
    let hierarchy = h.finish();
    (end - start, hierarchy)
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

/// Each printable character is a digit in base (126 - 32) = 94.
/// The most significant digit is on the right!
#[inline]
fn id_to_int(id: &[u8]) -> SignalHandle {
    assert!(!id.is_empty());
    let mut value: u64 = 0;
    for bb in id.iter().rev() {
        let char_val = (*bb - 33) as u64;
        value = (value * 94) + char_val;
    }
    value as u32
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
const MIN_CHUNK_SIZE: u64 = 8 * 1024;

#[inline]
fn int_div_ceil(a: u64, b: u64) -> u64 {
    (a + b - 1) / b
}

/// Returns starting byte and read length for every thread. Note that read-length is just an
/// approximation and the thread might have to read beyond or might also run out of data before
/// reaching read length.
#[inline]
fn determine_thread_chunks(body_len: u64) -> Vec<(u64, u64)> {
    let max_threads = rayon::current_num_threads() as u64;
    let number_of_threads_for_min_chunk_size = int_div_ceil(body_len, MIN_CHUNK_SIZE);
    let num_threads = std::cmp::min(max_threads, number_of_threads_for_min_chunk_size);
    let chunk_size = int_div_ceil(body_len, num_threads);
    (0..num_threads)
        .map(|ii| (ii * chunk_size, chunk_size))
        .collect()
}

/// Reads the body of a VCD with multiple threads
fn read_values_multi_threaded(
    filename: &str,
    total_len: u64,
    header_len: u64,
    hierarchy: &Hierarchy,
) -> Values {
    assert!(total_len >= header_len);
    let chunks = determine_thread_chunks(total_len - header_len);
    println!("{chunks:?}");
    let blocks: Vec<Values> = chunks
        .par_iter()
        .map(|(start, len)| {
            let mut input = std::fs::File::open(filename).expect("failed to open input file!");
            input
                .seek(SeekFrom::Start((header_len + start) as u64))
                .unwrap();
            read_values(&mut input, hierarchy)
        })
        .collect();

    println!("connect blocks");
    blocks.into_iter().next().unwrap()
}

fn read_values<R: Read>(input: &mut R, hierarchy: &Hierarchy) -> Values {
    let mut v = ValueBuilder::default();
    for var in hierarchy.iter_vars() {
        v.add_signal(var.handle(), var.length())
    }

    let foo = |pos: usize, cmd: BodyCmd| match cmd {
        BodyCmd::Time(value) => {
            let int_value = u64::from_str_radix(std::str::from_utf8(value).unwrap(), 10).unwrap();
            v.time_change(int_value);
        }
        BodyCmd::Value(value, id) => {
            v.value_change(id_to_int(id), value);
        }
    };

    read_body(input, foo, None).unwrap();

    v.print_statistics();
    v.finish()
}

fn read_body(
    input: &mut impl Read,
    mut callback: impl FnMut(usize, BodyCmd),
    read_len: Option<usize>,
) -> std::io::Result<()> {
    let mut total_read_len = 0;
    let mut buf = vec![0u8; 8 * 1024];
    let mut remaining_bytes = 0usize;
    let mut lines_read = 0usize;
    loop {
        // fill buffer
        let buf_read_len = input.read(&mut &mut buf[remaining_bytes..])?;
        let buf_len = buf_read_len + remaining_bytes;
        if buf_len == 0 {
            return Ok(());
        }

        // search for tokens
        let mut token_start: Option<usize> = None;
        let mut prev_token: Option<&[u8]> = None;
        let mut bytes_consumed = 0usize;
        let mut pending_lines = 0usize;
        for (pos, b) in buf.iter().take(buf_len).enumerate() {
            match b {
                b' ' | b'\n' | b'\r' | b'\t' => {
                    if token_start.is_none() {
                        // if we aren't tracking anything, we can just consume the whitespace
                        bytes_consumed = pos + 1;
                        if *b == b'\n' {
                            lines_read += 1;
                        }
                    } else {
                        match try_finish_token(
                            &buf,
                            pos,
                            lines_read,
                            &mut token_start,
                            &mut prev_token,
                        ) {
                            None => {
                                if *b == b'\n' {
                                    pending_lines += 1;
                                }
                            }
                            Some(cmd) => {
                                (callback)(pos, cmd);
                                bytes_consumed = pos + 1;
                                lines_read += pending_lines;
                                pending_lines = 0;
                                if *b == b'\n' {
                                    lines_read += 1;
                                }
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
            total_read_len += 1;
            match read_len {
                Some(value) if total_read_len >= value => {
                    return Ok(());
                }
                _ => {}
            }
        }

        // if we did not consume any bytes, we might be at the end of the stream which ends in
        // a token
        if bytes_consumed == 0 && buf_read_len == 0 {
            match try_finish_token(&buf, buf_len, lines_read, &mut token_start, &mut prev_token) {
                None => {}
                Some(cmd) => {
                    (callback)(buf_len, cmd);
                    bytes_consumed = buf_len;
                }
            }
            return Ok(()); // in case we did not make progress
        }

        // move remaining bytes to the front
        remaining_bytes = buf_len - bytes_consumed;
        if remaining_bytes > 0 && bytes_consumed > 0 {
            // copy remaining bytes to the left
            let overlaps = bytes_consumed < remaining_bytes;
            if overlaps {
                for ii in 0..remaining_bytes {
                    buf[ii] = buf[ii + bytes_consumed];
                }
            } else {
                let (dest, src) = buf.split_at_mut(bytes_consumed);
                dest[0..remaining_bytes].copy_from_slice(&src[0..remaining_bytes]);
            }
        }
    }
}

#[inline]
fn try_finish_token<'a>(
    buf: &'a [u8],
    pos: usize,
    lines_read: usize,
    token_start: &mut Option<usize>,
    prev_token: &mut Option<&'a [u8]>,
) -> Option<BodyCmd<'a>> {
    match *token_start {
        None => None,
        Some(start) => {
            let token = &buf[start..pos];
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
                            if token != b"$dumpvars" {
                                // ignore dumpvars command
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
                                "Unexpected tokens: `{}` and `{}` ({lines_read} lines after header)",
                                String::from_utf8_lossy(first),
                                String::from_utf8_lossy(token)
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

    fn read_body_to_vec(input: &mut impl BufRead) -> Vec<String> {
        let mut out = Vec::new();
        let foo = |cmd: BodyCmd| {
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
        };

        read_body(input, foo, None).unwrap();
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
