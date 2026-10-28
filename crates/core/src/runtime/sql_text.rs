// SPDX-License-Identifier: Apache-2.0

//! SQL text helpers: identifier quoting, code tokens, and the "may this SQL reference /
//! call ..." checks used by the engine guard and the host fallback.

use super::*;

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// A code token of a (lower-cased) SQL text, see [`sql_code_tokens`].
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, PartialEq)]
pub(super) enum SqlToken<'a> {
    /// An unquoted identifier or keyword (identifier characters of [`SqlNames`]).
    Word(&'a str),
    /// The content of a `"..."` identifier (`""` unescaped).
    Quoted(String),
    /// The content of a `'...'` string literal (`''` unescaped).
    Str(String),
    /// Any other non-space character.
    Punct(char),
}

/// Split `sql` into code tokens, skipping whitespace, `--` and `/* */` comments, and keeping
/// string literals and quoted identifiers as single tokens (W4-D-002). Unterminated quotes
/// and comments run to the end of the text.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn sql_code_tokens(sql: &str) -> Vec<SqlToken<'_>> {
    let mut tokens = Vec::new();
    let mut chars = sql.char_indices().peekable();
    while let Some((start, c)) = chars.next() {
        match c {
            c if c.is_whitespace() => {}
            '-' if chars.peek().is_some_and(|&(_, n)| n == '-') => {
                for (_, n) in chars.by_ref() {
                    if n == '\n' {
                        break;
                    }
                }
            }
            '/' if chars.peek().is_some_and(|&(_, n)| n == '*') => {
                chars.next();
                let mut star = false;
                for (_, n) in chars.by_ref() {
                    if star && n == '/' {
                        break;
                    }
                    star = n == '*';
                }
            }
            '\'' | '"' => {
                let mut text = String::new();
                while let Some((_, n)) = chars.next() {
                    if n == c {
                        if chars.peek().is_some_and(|&(_, m)| m == c) {
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    text.push(n);
                }
                tokens.push(if c == '\'' {
                    SqlToken::Str(text)
                } else {
                    SqlToken::Quoted(text)
                });
            }
            c if SqlNames::is_word_char(c) => {
                let mut end = start + c.len_utf8();
                while let Some(&(i, n)) = chars.peek() {
                    if !SqlNames::is_word_char(n) {
                        break;
                    }
                    end = i + n.len_utf8();
                    chars.next();
                }
                tokens.push(SqlToken::Word(&sql[start..end]));
            }
            other => tokens.push(SqlToken::Punct(other)),
        }
    }
    tokens
}

/// Whether `sql` has a fractional number literal (`0.1`, `.5`, `12.25`; not `1e3` or `1.5e3`),
/// which DuckDB types as an exact DECIMAL and craton-bolt 0.9 as DOUBLE (W8-B-002). Works on
/// code tokens, so string literals, quoted identifiers and comments do not count.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn has_fractional_literal(sql: &str) -> bool {
    let tokens = sql_code_tokens(sql);
    let digits = |w: &str| !w.is_empty() && w.bytes().all(|b| b.is_ascii_digit());
    tokens.windows(2).enumerate().any(|(i, pair)| {
        let (SqlToken::Punct('.'), SqlToken::Word(fraction)) = (&pair[0], &pair[1]) else {
            return false;
        };
        // `<digits>.<digits>` or `.<digits>` (not `t.c` or `"t".c`).
        digits(fraction)
            && match i.checked_sub(1).map(|p| &tokens[p]) {
                Some(SqlToken::Word(whole)) => digits(whole),
                Some(SqlToken::Quoted(_)) => false,
                _ => true,
            }
    })
}

/// The identifier-like words of a SQL text, for "might this SQL reference table `name`" checks
/// (W2-E-007).
///
/// The text is split into maximal runs of identifier characters (ASCII letters, digits, `_`,
/// `$`, and every non-ASCII character), ASCII-lowercased. Quoted identifiers and string
/// literals are split the same way, so `"My T"`, `'t'` (e.g. `query_table('t')`) and comments
/// still count: false positives only cost work, and there are no false negatives for literal
/// references. Unlike the former substring test, `t` no longer matches `select`, and
/// `__bolt_x_1` no longer matches `__bolt_x_10`.
#[cfg(not(target_arch = "wasm32"))]
pub(super) struct SqlNames {
    pub(super) words: HashSet<String>,
    lower: String,
}

#[cfg(not(target_arch = "wasm32"))]
impl SqlNames {
    pub(super) fn is_word_char(c: char) -> bool {
        c.is_ascii_alphanumeric() || c == '_' || c == '$' || !c.is_ascii()
    }

    pub(super) fn new(sql: &str) -> Self {
        let lower = sql.to_ascii_lowercase();
        let words = lower
            .split(|c: char| !Self::is_word_char(c))
            .filter(|w| !w.is_empty())
            .map(str::to_string)
            .collect();
        Self { words, lower }
    }

    /// Whether `name` (a catalog key) may be referenced by the SQL text.
    pub(super) fn may_reference(&self, name: &str) -> bool {
        let name = name.to_ascii_lowercase();
        if name.chars().all(Self::is_word_char) {
            return self.words.contains(&name);
        }
        // A name with separators (`My T`, `a/b`, `x.y`) can only be written quoted:
        // substring test on the whole text, also in the escaped spelling (`a"b` is written
        // `"a""b"`, W3-E-002).
        self.lower.contains(&name) || self.lower.contains(&name.replace('"', "\"\""))
    }

    /// Whether the query calls into the extension (W3-E-004): a `bolt_*(...)` function call
    /// (not a catalog table name), a `bolt.` qualified name, or a `'bolt:x'` / `"bolt:x"`
    /// table reference after `FROM` / `JOIN`.
    ///
    /// Works on code tokens, so string literals, quoted identifiers and comments that merely
    /// mention `bolt_x`, `bolt.com` or `bolt: done`, and columns or aliases named `bolt_*`, do
    /// not count (W4-D-002). The check only gives an early, clear error: a call it misses is
    /// still refused immediately by [`BoltRuntime::request`] on the worker thread.
    pub(super) fn may_call_bolt(&self, is_table: impl Fn(&str) -> bool) -> bool {
        let tokens = sql_code_tokens(&self.lower);
        tokens.iter().enumerate().any(|(i, token)| {
            let next = tokens.get(i + 1);
            let prev = i.checked_sub(1).and_then(|p| tokens.get(p));
            match token {
                SqlToken::Word(w) if w.starts_with("bolt_") => {
                    matches!(next, Some(SqlToken::Punct('('))) && !is_table(w)
                }
                SqlToken::Word("bolt") => matches!(next, Some(SqlToken::Punct('.'))),
                SqlToken::Quoted(q) if q == "bolt" => matches!(next, Some(SqlToken::Punct('.'))),
                SqlToken::Quoted(text) | SqlToken::Str(text) => {
                    text.trim_start().starts_with("bolt:")
                        && matches!(prev, Some(SqlToken::Word("from" | "join")))
                }
                _ => false,
            }
        })
    }

    /// Number of `AND` / `OR` keywords: an upper bound on the depth of the boolean predicate
    /// trees of the query (W2-B-001).
    pub(super) fn predicate_terms(&self) -> usize {
        self.lower
            .split(|c: char| !Self::is_word_char(c))
            .filter(|w| *w == "or" || *w == "and")
            .count()
    }
}
