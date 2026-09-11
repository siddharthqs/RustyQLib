//! The hazard-rate term structure every credit-sensitive instrument
//! prices off: risky bonds ([`bonds::credit`](crate::bonds::credit)),
//! convertibles under jump to default ([`hybrid`](crate::hybrid)), and
//! credit default swaps ([`cds`](super::cds)).

use crate::core::errors::RustyQLibError;

/// A term structure of default intensity: piecewise-constant hazard
/// rates on pillar times (year fractions from the pricing settlement,
/// on the discount curve's day count). The hazard beyond the last
/// pillar extends flat, mirroring the discount curve's zero-rate
/// extrapolation.
#[derive(Debug, Clone)]
pub struct CreditCurve {
    /// Pillar end times, strictly increasing and positive.
    times: Vec<f64>,
    /// Hazard on `(t_{i-1}, t_i]` (and beyond the last pillar).
    hazards: Vec<f64>,
}

impl CreditCurve {
    /// A single constant hazard for all horizons (one pillar, extended
    /// flat).
    pub fn flat(hazard_rate: f64) -> Result<Self, RustyQLibError> {
        Self::new(&[(1.0, hazard_rate)])
    }

    /// A piecewise-constant curve from `(pillar_time, hazard)` pairs.
    pub fn new(pillars: &[(f64, f64)]) -> Result<Self, RustyQLibError> {
        if pillars.is_empty() {
            return Err(RustyQLibError::invalid_input("credit curve", "no pillars"));
        }
        for &(time, hazard) in pillars {
            if !(time > 0.0 && time.is_finite()) {
                return Err(RustyQLibError::invalid_input(
                    "credit curve",
                    format!("pillar times must be positive and finite, got {time}"),
                ));
            }
            if !(hazard >= 0.0 && hazard.is_finite()) {
                return Err(RustyQLibError::invalid_input(
                    "credit curve",
                    format!("hazards must be non-negative, got {hazard} at {time}"),
                ));
            }
        }
        if pillars.windows(2).any(|w| w[1].0 <= w[0].0) {
            return Err(RustyQLibError::invalid_input(
                "credit curve",
                "pillar times must be strictly increasing",
            ));
        }
        Ok(CreditCurve {
            times: pillars.iter().map(|&(t, _)| t).collect(),
            hazards: pillars.iter().map(|&(_, h)| h).collect(),
        })
    }

    /// The pillars as `(time, hazard)` pairs.
    pub fn pillars(&self) -> Vec<(f64, f64)> {
        self.times
            .iter()
            .zip(&self.hazards)
            .map(|(&t, &h)| (t, h))
            .collect()
    }

    /// The instantaneous hazard at `t`: the pillar's on `(t_{i-1}, t_i]`,
    /// the last pillar's beyond.
    pub fn hazard_at(&self, t: f64) -> f64 {
        for (&segment_end, &hazard) in self.times.iter().zip(&self.hazards) {
            if t <= segment_end {
                return hazard;
            }
        }
        self.hazards[self.hazards.len() - 1]
    }

    /// Survival probability to `t`: `exp(-integral of the hazard)`.
    pub fn survival(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        let mut integral = 0.0;
        let mut segment_start = 0.0;
        for (&segment_end, &hazard) in self.times.iter().zip(&self.hazards) {
            if t <= segment_end {
                integral += hazard * (t - segment_start);
                return (-integral).exp();
            }
            integral += hazard * (segment_end - segment_start);
            segment_start = segment_end;
        }
        // flat extrapolation of the last hazard
        integral += self.hazards[self.hazards.len() - 1] * (t - segment_start);
        (-integral).exp()
    }

    /// Probability of default in `(t1, t2]`.
    pub fn default_probability(&self, t1: f64, t2: f64) -> f64 {
        self.survival(t1) - self.survival(t2)
    }
}
