//! Thin shim over the library plotter (`rustyqlib::utils::plot3d`),
//! keeping the examples' original panic-on-error, print-on-save
//! behavior. The figure generation itself lives in the library so the
//! CLI `build` command and the examples render identical artifacts.

// each example uses a different subset of these re-exports
#[allow(unused_imports)]
pub use rustyqlib::utils::plot3d::{greek_surface, linspace, GreekSurface, Labels, LineSeries};

/// Render the surface to a self-contained interactive HTML file at
/// `path` (creating parent directories).
pub fn save_surface_html(surface: &GreekSurface, path: &str, labels: &Labels) {
    rustyqlib::utils::plot3d::save_surface_html(surface, path, labels)
        .unwrap_or_else(|e| panic!("cannot write {path}: {e}"));
    println!("  saved {path}");
}

/// Multi-series 2-D chart; set `log_x` / `log_y` for log axes.
#[allow(clippy::too_many_arguments)]
pub fn save_lines_html(
    series: &[LineSeries],
    path: &str,
    title: &str,
    x_label: &str,
    y_label: &str,
    log_x: bool,
    log_y: bool,
) {
    rustyqlib::utils::plot3d::save_lines_html(series, path, title, x_label, y_label, log_x, log_y)
        .unwrap_or_else(|e| panic!("cannot write {path}: {e}"));
    println!("  saved {path}");
}
