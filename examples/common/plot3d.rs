//! Interactive 3D Greek-surface plots as self-contained Plotly HTML.
//!
//! Each surface (a Greek over moneyness and maturity) is written as a single
//! HTML file with all data embedded inline as JSON and an interactive Plotly
//! `surface` trace — open it in a browser to rotate, zoom and hover, which
//! makes the shape and smoothness of a Greek easy to inspect.
//!
//! The Plotly figure spec is generated directly with `serde_json` (a core
//! dependency) rather than through the `plotly` crate, whose current release
//! pulls a broken transitive dependency. The Plotly JavaScript library is
//! loaded from its CDN (the standard for Plotly HTML exports); to view fully
//! offline, replace the one `<script src=...>` line with a local copy.

use std::fs;
use std::path::Path;

use serde_json::json;

/// A `z[i][j]` surface sampled at `xs[i]` (moneyness) and `ys[j]` (maturity).
pub struct GreekSurface {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub z: Vec<Vec<f64>>,
}

/// `n` points evenly spaced over `[a, b]` (inclusive).
pub fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![a];
    }
    (0..n).map(|i| a + (b - a) * i as f64 / (n - 1) as f64).collect()
}

/// Sample `f(x, y)` on the `xs` x `ys` grid (x = moneyness, y = maturity).
pub fn greek_surface(xs: &[f64], ys: &[f64], f: impl Fn(f64, f64) -> f64) -> GreekSurface {
    let z = xs.iter().map(|&x| ys.iter().map(|&y| f(x, y)).collect()).collect();
    GreekSurface { xs: xs.to_vec(), ys: ys.to_vec(), z }
}

/// Axis and title labels.
pub struct Labels<'a> {
    pub title: &'a str,
    pub x: &'a str,
    pub y: &'a str,
    pub z: &'a str,
}

const PLOTLY_CDN: &str = "https://cdn.plot.ly/plotly-2.35.2.min.js";

/// Render the surface to a self-contained interactive HTML file at `path`
/// (creating parent directories).
pub fn save_surface_html(surface: &GreekSurface, path: &str, labels: &Labels) {
    // plotly wants z indexed [row = y][col = x]; our grid is [x][y]
    let nx = surface.xs.len();
    let ny = surface.ys.len();
    let z: Vec<Vec<f64>> =
        (0..ny).map(|j| (0..nx).map(|i| surface.z[i][j]).collect()).collect();

    let data = json!([{
        "type": "surface",
        "x": surface.xs,
        "y": surface.ys,
        "z": z,
        "colorscale": "Viridis",
        "colorbar": { "title": { "text": labels.z } },
        // project the surface onto the z-floor as filled contours: makes the
        // shape (ridges, sign changes) legible from any viewing angle
        "contours": { "z": {
            "show": true,
            "usecolormap": true,
            "highlightcolor": "#ffffff",
            "project": { "z": true }
        }},
        "hovertemplate":
            format!("{}: %{{x:.3f}}<br>{}: %{{y:.3f}}<br>{}: %{{z:.5f}}<extra></extra>",
                    labels.x, labels.y, labels.z),
    }]);

    let layout = json!({
        "title": { "text": labels.title },
        "autosize": true,
        "margin": { "l": 0, "r": 0, "t": 50, "b": 0 },
        "scene": {
            "xaxis": { "title": { "text": labels.x } },
            "yaxis": { "title": { "text": labels.y } },
            "zaxis": { "title": { "text": labels.z } },
            "camera": { "eye": { "x": 1.7, "y": -1.7, "z": 0.9 } }
        }
    });

    let html = format!(
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\"/>\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>\n\
         <title>{title}</title>\n\
         <script src=\"{cdn}\" charset=\"utf-8\"></script>\n\
         <style>html,body{{height:100%;margin:0}}#plot{{width:100vw;height:100vh}}</style>\n\
         </head>\n<body>\n<div id=\"plot\"></div>\n<script>\n\
         Plotly.newPlot('plot', {data}, {layout}, {{responsive:true}});\n\
         </script>\n</body>\n</html>\n",
        title = labels.title,
        cdn = PLOTLY_CDN,
        data = serde_json::to_string(&data).unwrap(),
        layout = serde_json::to_string(&layout).unwrap(),
    );

    if let Some(parent) = Path::new(path).parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(path, html).unwrap_or_else(|e| panic!("cannot write {path}: {e}"));
    println!("  saved {path}");
}
