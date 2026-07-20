//! Convert contract documents between JSON and XML.
//!
//! Both formats deserialize into the same data model, so conversion is a
//! transcode with no schema knowledge involved.
//!
//! ```bash
//! cargo run --release --example convert_format -- in.json out.xml
//! cargo run --release --example convert_format -- in.xml  out.json
//! ```

use std::process::ExitCode;

use rustyqlib::core::serialization::{parse_value, render_value, Format};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: convert_format <input.(json|xml)> <output.(json|xml)>");
        return ExitCode::FAILURE;
    }
    let (input, output) = (&args[1], &args[2]);

    let text = match std::fs::read_to_string(input) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read {input}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let in_format = Format::from_path(input).unwrap_or_else(|| Format::detect(&text));
    let out_format = match Format::from_path(output) {
        Some(f) => f,
        None => {
            eprintln!("cannot infer output format from {output}: use a .json or .xml extension");
            return ExitCode::FAILURE;
        }
    };

    let value = match parse_value(&text, in_format) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("cannot parse {input} as {in_format:?}: {e}");
            return ExitCode::FAILURE;
        }
    };

    // "contracts" is the document element for contract files; it is only
    // used when writing XML
    let rendered = render_value(&value, out_format, "contracts");
    if let Err(e) = std::fs::write(output, rendered) {
        eprintln!("cannot write {output}: {e}");
        return ExitCode::FAILURE;
    }
    println!("{input} ({in_format:?}) -> {output} ({out_format:?})");
    ExitCode::SUCCESS
}
