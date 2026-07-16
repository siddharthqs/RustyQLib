use rustyqlib::utils::build_cli;

fn main() {
    let matches = build_cli::build_cli().get_matches();

    // Match and dispatch subcommand
    match matches.subcommand() {
        Some(("build", build_matches)) => build_cli::handle_build(build_matches),
        Some(("file", file_matches)) => build_cli::handle_file(file_matches),
        Some(("dir", dir_matches)) => build_cli::handle_dir(dir_matches),
        Some(("interactive", _)) => build_cli::handle_interactive(),
        _ => {
            // No mode specified or unknown mode
            println!("No valid subcommand specified. Use --help to see available options.");
        }
    }
}
