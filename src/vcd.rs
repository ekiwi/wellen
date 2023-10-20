// Copyright 2023 The Regents of the University of California
// released under BSD 3-Clause License
// author: Kevin Laeufer <laeufer@berkeley.edu>

use crate::hierarchy::*;
use crate::values::*;
use std::io::BufRead;
use vcd::Command;

pub fn read(filename: &str) -> (Hierarchy, Values) {
    let input = std::fs::File::open(filename).expect("failed to open input file!");
    let mut parser = vcd::Parser::new(std::io::BufReader::new(input));
    let hierarchy = read_hierarchy(&mut parser);
    let values = read_values(&mut parser, &hierarchy);
    (hierarchy, values)
}

fn read_hierarchy<R: BufRead>(parser: &mut vcd::Parser<R>) -> Hierarchy {
    let header = parser.parse_header().unwrap();
    let mut h = HierarchyBuilder::default();
    for item in header.items {
        add_item(item, &mut h);
    }
    h.print_statistics();
    h.finish()
}

fn convert_scope_tpe(tpe: vcd::ScopeType) -> ScopeType {
    match tpe {
        vcd::ScopeType::Module => ScopeType::Module,
        _ => ScopeType::Todo,
    }
}

fn convert_var_tpe(tpe: vcd::VarType) -> VarType {
    match tpe {
        vcd::VarType::Wire => VarType::Wire,
        _ => VarType::Todo,
    }
}

fn add_item(item: vcd::ScopeItem, h: &mut HierarchyBuilder) {
    match item {
        vcd::ScopeItem::Scope(scope) => {
            h.add_scope(scope.identifier, convert_scope_tpe(scope.scope_type));
            for item in scope.items {
                add_item(item, h);
            }
            h.pop_scope();
        }
        vcd::ScopeItem::Var(var) => {
            h.add_var(
                var.reference,
                convert_var_tpe(var.var_type),
                VarDirection::Todo,
                var.size,
                0, // problem: we cannot get to the actual ID because it is private!
            );
        }
        vcd::ScopeItem::Comment(_) => {} // ignore comments
        _ => {}
    }
}

fn read_values<R: BufRead>(parser: &mut vcd::Parser<R>, hierarchy: &Hierarchy) -> Values {
    let mut v = ValueBuilder::default();
    for var in hierarchy.iter_vars() {
        v.add_signal(var.handle(), var.length())
    }

    for command_result in parser {
        let command = command_result.unwrap();
        use vcd::Command::*;
        match command {
            ChangeScalar(i, v) => {
                todo!()
            }

            _ => {}
        }
    }
    v.print_statistics();
    v.finish()
}

fn read_body(
    input: &mut impl BufRead,
    mut callback: impl FnMut(BodyCmd),
    read_len: Option<usize>,
) -> std::io::Result<()> {
    let mut total_read_len = 0;
    loop {
        // fill buffer
        let buf = input.fill_buf()?;
        let buf_len = buf.len();
        if buf.is_empty() {
            // if we read to the end of the file, see if there is still a token that can be finished

            return Ok(());
        }

        // search for tokens
        let mut token_start: Option<usize> = None;
        let mut prev_token: Option<&[u8]> = None;
        let mut bytes_consumed = 0usize;
        for (pos, b) in buf.iter().enumerate() {
            match b {
                b' ' | b'\n' | b'\r' | b'\t' => {
                    if token_start.is_none() {
                        // if we aren't tracking anything, we can just consume the whitespace
                        bytes_consumed = pos + 1;
                    } else {
                        match try_finish_token(buf, pos, &mut token_start, &mut prev_token) {
                            None => {}
                            Some(cmd) => {
                                (callback)(cmd);
                                bytes_consumed = pos + 1;
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
        if bytes_consumed == 0 {
            match try_finish_token(buf, buf.len(), &mut token_start, &mut prev_token) {
                None => {}
                Some(cmd) => {
                    (callback)(cmd);
                    bytes_consumed = buf.len();
                }
            }
        }

        // notify the input of how many byte we consumed
        input.consume(bytes_consumed);
    }
}

#[inline]
fn try_finish_token<'a>(
    buf: &'a [u8],
    pos: usize,
    token_start: &mut Option<usize>,
    prev_token: &mut Option<&'a [u8]>,
) -> Option<BodyCmd<'a>> {
    match *token_start {
        None => None,
        Some(start) => {
            let token = &buf[start..pos];
            let ret = match *prev_token {
                None => {
                    // 1-token commands are binary changes or time commands
                    match token[0] {
                        b'#' => Some(BodyCmd::Time(&token[1..])),
                        b'0' | b'1' | b'z' | b'Z' | b'x' | b'X' => {
                            Some(BodyCmd::Value(&token[0..1], &token[1..]))
                        }
                        _ => {
                            *prev_token = Some(token);
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
                            panic!("Unexpected tokens: {first:?} {token:?}");
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
            "j2! = 0",
        ];
        let res = read_body_to_vec(&mut input.as_bytes());
        assert_eq!(res, expected);
    }
}
