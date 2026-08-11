//! Interactive Plotly HTML artifacts: 3-D surfaces (Greeks, implied
//! vol) and 2-D line charts.
//!
//! Each figure is one self-contained HTML file with the data embedded
//! inline as JSON and an interactive Plotly trace — open it in a browser
//! to rotate, zoom and hover. The figure spec is generated directly with
//! `serde_json` (a core dependency) rather than through the `plotly`
//! crate, so this module costs no extra dependencies. The Plotly
//! JavaScript library is loaded from its CDN (the standard for Plotly
//! HTML exports); to view fully offline, replace the one
//! `<script src=...>` line with a local copy.
//!
//! The `*_html` functions are pure (data in, HTML string out); the
//! `save_*` wrappers write the file, creating parent directories.

use std::fs;
use std::path::Path;

use serde_json::json;

use crate::core::vols::{SmileCoordinate, VolInput, VolSurface};

const PLOTLY_CDN: &str = "https://cdn.plot.ly/plotly-2.35.2.min.js";

/// A `z[i][j]` surface sampled at `xs[i]` and `ys[j]`.
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
    (0..n)
        .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
        .collect()
}

/// Sample `f(x, y)` on the `xs` x `ys` grid.
pub fn greek_surface(xs: &[f64], ys: &[f64], f: impl Fn(f64, f64) -> f64) -> GreekSurface {
    let z = xs
        .iter()
        .map(|&x| ys.iter().map(|&y| f(x, y)).collect())
        .collect();
    GreekSurface {
        xs: xs.to_vec(),
        ys: ys.to_vec(),
        z,
    }
}

/// Axis and title labels.
pub struct Labels<'a> {
    pub title: &'a str,
    pub x: &'a str,
    pub y: &'a str,
    pub z: &'a str,
}

/// The standard document shell around a Plotly figure.
fn html_page(title: &str, data: &serde_json::Value, layout: &serde_json::Value) -> String {
    format!(
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\"/>\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>\n\
         <title>{title}</title>\n\
         <script src=\"{PLOTLY_CDN}\" charset=\"utf-8\"></script>\n\
         <style>html,body{{height:100%;margin:0}}#plot{{width:100vw;height:100vh}}</style>\n\
         </head>\n<body>\n<div id=\"plot\"></div>\n<script>\n\
         Plotly.newPlot('plot', {data}, {layout}, {{responsive:true}});\n\
         </script>\n</body>\n</html>\n",
        data = serde_json::to_string(data).unwrap_or_default(),
        layout = serde_json::to_string(layout).unwrap_or_default(),
    )
}

fn scene_layout(labels: &Labels) -> serde_json::Value {
    json!({
        "title": { "text": labels.title },
        "autosize": true,
        "margin": { "l": 0, "r": 0, "t": 50, "b": 0 },
        "scene": {
            "xaxis": { "title": { "text": labels.x } },
            "yaxis": { "title": { "text": labels.y } },
            "zaxis": { "title": { "text": labels.z } },
            "camera": { "eye": { "x": 1.7, "y": -1.7, "z": 0.9 } }
        }
    })
}

fn surface_trace(surface: &GreekSurface, labels: &Labels) -> serde_json::Value {
    // plotly wants z indexed [row = y][col = x]; our grid is [x][y]
    let nx = surface.xs.len();
    let ny = surface.ys.len();
    let z: Vec<Vec<f64>> = (0..ny)
        .map(|j| (0..nx).map(|i| surface.z[i][j]).collect())
        .collect();
    json!({
        "type": "surface",
        "x": surface.xs,
        "y": surface.ys,
        "z": z,
        "colorscale": "Viridis",
        "colorbar": { "title": { "text": labels.z } },
        // project the surface onto the z-floor as filled contours: makes
        // the shape (ridges, sign changes) legible from any viewing angle
        "contours": { "z": {
            "show": true,
            "usecolormap": true,
            "highlightcolor": "#ffffff",
            "project": { "z": true }
        }},
        "hovertemplate":
            format!("{}: %{{x:.3f}}<br>{}: %{{y:.3f}}<br>{}: %{{z:.5f}}<extra></extra>",
                    labels.x, labels.y, labels.z),
    })
}

/// Render one sampled surface as a self-contained HTML figure.
pub fn surface_html(surface: &GreekSurface, labels: &Labels) -> String {
    html_page(
        labels.title,
        &json!([surface_trace(surface, labels)]),
        &scene_layout(labels),
    )
}

/// [`surface_html`] written to `path` (creating parent directories).
pub fn save_surface_html(
    surface: &GreekSurface,
    path: &str,
    labels: &Labels,
) -> std::io::Result<()> {
    write_html(path, &surface_html(surface, labels))
}

/// An implied vol surface as an interactive figure: the interpolated
/// surface sampled over the pillar range, with one marker per actual
/// quote pillar — raw points versus interpolation is the honest view of
/// a fitted surface.
///
/// The x-axis is the surface's own smile coordinate (absolute strike for
/// chain-built surfaces; forward moneyness or log-moneyness for surfaces
/// quoted that way). Time runs over the pillar span, padded a little
/// past the ends to show the flat extrapolation. A flat surface renders
/// as its constant plane.
pub fn vol_surface_html(surface: &VolSurface, title: &str) -> String {
    let (expiries, smiles, coordinate) = match surface.to_input() {
        VolInput::StrikeSmiles {
            expiries,
            smiles,
            coordinate,
            ..
        } => (expiries, smiles, coordinate),
        // flat (or any grid-only) surface: a constant plane over a
        // nominal window
        _ => {
            let vol = surface.vol(100.0, 100.0, 1.0);
            (
                vec![
                    crate::core::curves::Tenor::YearFraction(0.25),
                    crate::core::curves::Tenor::YearFraction(2.0),
                ],
                vec![
                    vec![(50.0, vol), (150.0, vol)],
                    vec![(50.0, vol), (150.0, vol)],
                ],
                SmileCoordinate::Strike,
            )
        }
    };
    let times: Vec<f64> = expiries
        .iter()
        .map(|tenor| match tenor {
            crate::core::curves::Tenor::YearFraction(t) => *t,
            // to_input never emits dates, but stay total
            crate::core::curves::Tenor::Date(_) => 0.0,
        })
        .collect();

    // querying (strike = coordinate mapped back, forward = 1) hits the
    // stored smile coordinate exactly for all three coordinate kinds
    let to_strike = |x: f64| match coordinate {
        SmileCoordinate::Strike | SmileCoordinate::Moneyness => x,
        SmileCoordinate::LogMoneyness => x.exp(),
    };
    let x_label = match coordinate {
        SmileCoordinate::Strike => "strike",
        SmileCoordinate::Moneyness => "moneyness K/F",
        SmileCoordinate::LogMoneyness => "log-moneyness ln(K/F)",
    };

    let xs_all: Vec<f64> = smiles.iter().flatten().map(|&(x, _)| x).collect();
    let (x_min, x_max) = xs_all
        .iter()
        .fold((f64::MAX, f64::MIN), |(lo, hi), &x| (lo.min(x), hi.max(x)));
    let (t_min, t_max) = times
        .iter()
        .fold((f64::MAX, f64::MIN), |(lo, hi), &t| (lo.min(t), hi.max(t)));
    let xs = linspace(x_min, x_max, 60);
    // pad past the pillars so the flat time extrapolation is visible
    let ys = linspace(0.85 * t_min, 1.1 * t_max, 40);
    let sampled = greek_surface(&xs, &ys, |x, t| surface.vol(to_strike(x), 1.0, t));

    let labels = Labels {
        title,
        x: x_label,
        y: "expiry (years)",
        z: "implied vol",
    };
    let mut marker_x = Vec::new();
    let mut marker_y = Vec::new();
    let mut marker_z = Vec::new();
    for (t, smile) in times.iter().zip(&smiles) {
        for &(x, vol) in smile {
            marker_x.push(x);
            marker_y.push(*t);
            marker_z.push(vol);
        }
    }
    let markers = json!({
        "type": "scatter3d",
        "mode": "markers",
        "name": "quote pillars",
        "x": marker_x,
        "y": marker_y,
        "z": marker_z,
        "marker": { "size": 3, "color": "#111111", "symbol": "circle" },
        "hovertemplate":
            format!("{x_label}: %{{x:.3f}}<br>expiry: %{{y:.3f}}<br>vol: %{{z:.5f}}<extra>quote</extra>"),
    });
    html_page(
        title,
        &json!([surface_trace(&sampled, &labels), markers]),
        &scene_layout(&labels),
    )
}

/// [`vol_surface_html`] written to `path` (creating parent directories).
pub fn save_vol_surface_html(
    surface: &VolSurface,
    path: &str,
    title: &str,
) -> std::io::Result<()> {
    write_html(path, &vol_surface_html(surface, title))
}

// ── 2-D line charts (convergence diagrams etc.) ─────────────────────────

pub struct LineSeries {
    pub name: String,
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
}

/// Multi-series 2-D chart; set `log_x` / `log_y` for log axes (the shape
/// convergence plots want: error vs steps on log-log axes shows the
/// convergence order as the slope).
pub fn lines_html(
    series: &[LineSeries],
    title: &str,
    x_label: &str,
    y_label: &str,
    log_x: bool,
    log_y: bool,
) -> String {
    let data: Vec<serde_json::Value> = series
        .iter()
        .map(|s| {
            json!({
                "type": "scatter",
                "mode": "lines+markers",
                "name": s.name,
                "x": s.xs,
                "y": s.ys,
            })
        })
        .collect();
    let layout = json!({
        "title": { "text": title },
        "xaxis": { "title": { "text": x_label }, "type": if log_x { "log" } else { "linear" } },
        "yaxis": { "title": { "text": y_label }, "type": if log_y { "log" } else { "linear" } },
        "legend": { "orientation": "h", "y": -0.2 },
        "margin": { "t": 60 },
    });
    html_page(title, &serde_json::Value::Array(data), &layout)
}

/// [`lines_html`] written to `path` (creating parent directories).
#[allow(clippy::too_many_arguments)]
pub fn save_lines_html(
    series: &[LineSeries],
    path: &str,
    title: &str,
    x_label: &str,
    y_label: &str,
    log_x: bool,
    log_y: bool,
) -> std::io::Result<()> {
    write_html(path, &lines_html(series, title, x_label, y_label, log_x, log_y))
}

fn write_html(path: &str, html: &str) -> std::io::Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, html)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::Tenor;
    use crate::core::daycount::DayCountConvention;
    use chrono::NaiveDate;

    fn asof() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 8, 5).unwrap()
    }

    #[test]
    fn linspace_and_sampling_shape() {
        let xs = linspace(0.0, 1.0, 5);
        assert_eq!(xs, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(linspace(3.0, 9.0, 1), vec![3.0]);
        let s = greek_surface(&[1.0, 2.0], &[10.0, 20.0, 30.0], |x, y| x * y);
        assert_eq!(s.z, vec![vec![10.0, 20.0, 30.0], vec![20.0, 40.0, 60.0]]);
    }

    #[test]
    fn vol_surface_figure_embeds_surface_and_quote_markers() {
        let surface = VolSurface::from_strike_smiles(
            &[Tenor::YearFraction(0.5), Tenor::YearFraction(1.0)],
            &[
                vec![(90.0, 0.30), (100.0, 0.27), (110.0, 0.25)],
                vec![(95.0, 0.31), (100.0, 0.28)],
            ],
            asof(),
            DayCountConvention::Act365,
        )
        .unwrap();
        let html = vol_surface_html(&surface, "ACME implied vol");
        assert!(html.contains("Plotly.newPlot"));
        assert!(html.contains("\"surface\""));
        assert!(html.contains("scatter3d"), "quote pillar markers present");
        assert!(html.contains("ACME implied vol"));
        assert!(html.contains("implied vol"));
        // all five pillar vols appear verbatim in the marker trace
        for vol in ["0.3", "0.27", "0.25", "0.31", "0.28"] {
            assert!(html.contains(vol), "missing pillar vol {vol}");
        }
    }

    #[test]
    fn flat_surfaces_render_as_a_plane() {
        let flat = VolSurface::flat(0.2, asof(), DayCountConvention::Act365).unwrap();
        let html = vol_surface_html(&flat, "flat");
        assert!(html.contains("Plotly.newPlot"));
        assert!(html.contains("0.2"));
    }

    #[test]
    fn line_chart_renders_all_series() {
        let series = [
            LineSeries {
                name: "a".into(),
                xs: vec![1.0, 2.0],
                ys: vec![3.0, 4.0],
            },
            LineSeries {
                name: "b".into(),
                xs: vec![1.0, 2.0],
                ys: vec![5.0, 6.0],
            },
        ];
        let html = lines_html(&series, "t", "x", "y", true, false);
        assert!(html.contains("\"a\"") && html.contains("\"b\""));
        assert!(html.contains("\"log\""));
    }
}
