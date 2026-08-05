//! ANSI styles for the CLI's own messages, matching clap's palette.
//!
//! Print through `anstream::println!` / `anstream::eprintln!` so styles
//! are stripped automatically when the stream is not a terminal (or when
//! `NO_COLOR` is set), and render a style as `{STYLE}text{STYLE:#}` —
//! the alternate form emits the reset sequence.

/// Bold red, for the `error:` prefix on stderr.
pub const ERROR: anstyle::Style = anstyle::AnsiColor::Red.on_default().bold();

/// Green, for success confirmations such as saved-file notes.
pub const SUCCESS: anstyle::Style = anstyle::AnsiColor::Green.on_default();
