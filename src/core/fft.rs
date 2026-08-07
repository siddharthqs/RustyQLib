//! Fast Fourier transform and FFT-accelerated convolution.
//!
//! Self-contained, like the rest of the numerical core: an in-place
//! iterative radix-2 Cooley-Tukey transform for power-of-two lengths,
//! extended to **any** length through Bluestein's chirp-z algorithm
//! (which re-expresses an arbitrary-length DFT as a power-of-two
//! convolution), so callers never need to care about the size.
//!
//! - [`fft`] / [`ifft`] — forward and inverse discrete Fourier
//!   transforms of a complex buffer, any length, `O(n log n)`;
//! - [`RealConvolver`] — a precomputed plan for repeated linear
//!   convolution of real signals against one fixed kernel: the kernel's
//!   spectrum is transformed once, each apply costs one forward and one
//!   inverse FFT. This is the workhorse of the rough Bergomi hybrid
//!   scheme, where every Monte Carlo path convolves its Brownian
//!   increments with the same fractional-kernel weights.
//!
//! Conventions: `fft` computes `X_k = sum_j x_j e^{-2 pi i j k / n}`
//! (no scaling); `ifft` includes the `1/n` factor, so `ifft(fft(x))`
//! is the identity.

/// Complex number in rectangular form, the transform's element type.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };

    pub fn new(re: f64, im: f64) -> Complex {
        Complex { re, im }
    }

    /// `e^{i theta}` on the unit circle.
    pub fn cis(theta: f64) -> Complex {
        Complex {
            re: theta.cos(),
            im: theta.sin(),
        }
    }

    pub fn conj(self) -> Complex {
        Complex {
            re: self.re,
            im: -self.im,
        }
    }

    pub fn scale(self, s: f64) -> Complex {
        Complex {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Complex;
    fn add(self, o: Complex) -> Complex {
        Complex::new(self.re + o.re, self.im + o.im)
    }
}

impl std::ops::Sub for Complex {
    type Output = Complex;
    fn sub(self, o: Complex) -> Complex {
        Complex::new(self.re - o.re, self.im - o.im)
    }
}

impl std::ops::Mul for Complex {
    type Output = Complex;
    fn mul(self, o: Complex) -> Complex {
        Complex::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

/// Forward DFT, in place, any length (radix-2 for powers of two,
/// Bluestein otherwise). No scaling.
pub fn fft(x: &mut [Complex]) {
    let n = x.len();
    if n <= 1 {
        return;
    }
    if n.is_power_of_two() {
        radix2(x);
    } else {
        bluestein(x);
    }
}

/// Inverse DFT, in place, any length, including the `1/n` scaling:
/// `ifft(fft(x)) == x`.
pub fn ifft(x: &mut [Complex]) {
    let n = x.len();
    if n == 0 {
        return;
    }
    // conjugation reduces the inverse to the forward transform
    for v in x.iter_mut() {
        *v = v.conj();
    }
    fft(x);
    let inv_n = 1.0 / n as f64;
    for v in x.iter_mut() {
        *v = v.conj().scale(inv_n);
    }
}

/// Iterative radix-2 decimation-in-time with a precomputed twiddle
/// table. `x.len()` must be a power of two.
fn radix2(x: &mut [Complex]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    // bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            x.swap(i, j);
        }
    }
    // twiddles for the full size; a stage of half-length `half` strides
    // the table by n / (2 * half)
    let twiddle: Vec<Complex> = (0..n / 2)
        .map(|k| Complex::cis(-2.0 * std::f64::consts::PI * k as f64 / n as f64))
        .collect();
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let stride = n / len;
        for base in (0..n).step_by(len) {
            for k in 0..half {
                let t = twiddle[k * stride] * x[base + k + half];
                let u = x[base + k];
                x[base + k] = u + t;
                x[base + k + half] = u - t;
            }
        }
        len <<= 1;
    }
}

/// Bluestein's chirp-z algorithm: an arbitrary-length DFT as one
/// power-of-two circular convolution. The chirp exponents are reduced
/// `mod 2n` in integer arithmetic so the phase stays accurate at any
/// index.
fn bluestein(x: &mut [Complex]) {
    let n = x.len();
    let m = (2 * n - 1).next_power_of_two();
    let chirp: Vec<Complex> = (0..n)
        .map(|k| {
            let sq = (k as u64 * k as u64) % (2 * n as u64);
            Complex::cis(-std::f64::consts::PI * sq as f64 / n as f64)
        })
        .collect();
    let mut a = vec![Complex::ZERO; m];
    for k in 0..n {
        a[k] = x[k] * chirp[k];
    }
    let mut b = vec![Complex::ZERO; m];
    b[0] = chirp[0].conj();
    for k in 1..n {
        let c = chirp[k].conj();
        b[k] = c;
        b[m - k] = c;
    }
    radix2(&mut a);
    radix2(&mut b);
    for (av, bv) in a.iter_mut().zip(&b) {
        *av = *av * *bv;
    }
    // inverse power-of-two transform of the product
    for v in a.iter_mut() {
        *v = v.conj();
    }
    radix2(&mut a);
    let inv_m = 1.0 / m as f64;
    for (k, out) in x.iter_mut().enumerate() {
        *out = a[k].conj().scale(inv_m) * chirp[k];
    }
}

/// Precomputed plan for repeated linear convolution of real signals
/// against one fixed real kernel.
///
/// The kernel's spectrum is computed once at the padded power-of-two
/// size; every [`convolve_into`](Self::convolve_into) then costs one
/// forward and one inverse FFT — `O(m log m)` against the naive
/// `O(kernel * signal)` — with a caller-supplied scratch buffer so hot
/// loops (one convolution per Monte Carlo path) allocate nothing after
/// warm-up.
pub struct RealConvolver {
    m: usize,
    max_signal_len: usize,
    kernel_fft: Vec<Complex>,
}

impl RealConvolver {
    /// Plan for convolving signals of length up to `max_signal_len`
    /// against `kernel`. The padded size covers the full linear
    /// convolution, so no wrap-around ever aliases into the output.
    pub fn new(kernel: &[f64], max_signal_len: usize) -> RealConvolver {
        let m = (kernel.len() + max_signal_len).max(2).next_power_of_two();
        let mut kernel_fft = vec![Complex::ZERO; m];
        for (slot, &g) in kernel_fft.iter_mut().zip(kernel) {
            slot.re = g;
        }
        fft(&mut kernel_fft);
        RealConvolver {
            m,
            max_signal_len,
            kernel_fft,
        }
    }

    /// Linear convolution `out[i] = sum_j kernel[j] * signal[i - j]`,
    /// writing the first `out.len()` lags (at most the padded size).
    /// `scratch` is resized as needed and reusable across calls.
    pub fn convolve_into(&self, signal: &[f64], out: &mut [f64], scratch: &mut Vec<Complex>) {
        debug_assert!(signal.len() <= self.max_signal_len);
        debug_assert!(out.len() <= self.m);
        scratch.clear();
        scratch.resize(self.m, Complex::ZERO);
        for (slot, &s) in scratch.iter_mut().zip(signal) {
            slot.re = s;
        }
        fft(scratch);
        for (s, k) in scratch.iter_mut().zip(&self.kernel_fft) {
            *s = *s * *k;
        }
        ifft(scratch);
        for (o, s) in out.iter_mut().zip(scratch.iter()) {
            *o = s.re;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct `O(n^2)` DFT, the correctness reference.
    fn naive_dft(x: &[Complex]) -> Vec<Complex> {
        let n = x.len();
        (0..n)
            .map(|k| {
                let mut s = Complex::ZERO;
                for (j, &xj) in x.iter().enumerate() {
                    let theta = -2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;
                    s = s + xj * Complex::cis(theta);
                }
                s
            })
            .collect()
    }

    /// Deterministic pseudo-random complex test vector.
    fn test_vector(n: usize, seed: u64) -> Vec<Complex> {
        use crate::core::montecarlo::splitmix64;
        (0..n)
            .map(|i| {
                let a = splitmix64(seed ^ (2 * i) as u64) as f64 / u64::MAX as f64 - 0.5;
                let b = splitmix64(seed ^ (2 * i + 1) as u64) as f64 / u64::MAX as f64 - 0.5;
                Complex::new(a, b)
            })
            .collect()
    }

    fn assert_close(got: &[Complex], want: &[Complex], tol: f64) {
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g.re - w.re).abs() < tol && (g.im - w.im).abs() < tol,
                "index {i}: {g:?} vs {w:?}"
            );
        }
    }

    #[test]
    fn power_of_two_sizes_match_the_naive_dft() {
        for n in [1usize, 2, 4, 8, 16, 64, 256] {
            let x = test_vector(n, 7);
            let want = naive_dft(&x);
            let mut got = x.clone();
            fft(&mut got);
            assert_close(&got, &want, 1e-10 * n as f64);
        }
    }

    #[test]
    fn bluestein_sizes_match_the_naive_dft() {
        for n in [3usize, 5, 12, 17, 100, 121] {
            let x = test_vector(n, 11);
            let want = naive_dft(&x);
            let mut got = x.clone();
            fft(&mut got);
            assert_close(&got, &want, 1e-9 * n as f64);
        }
    }

    #[test]
    fn inverse_round_trips_at_any_length() {
        for n in [1usize, 2, 8, 12, 17, 64, 100] {
            let x = test_vector(n, 13);
            let mut y = x.clone();
            fft(&mut y);
            ifft(&mut y);
            assert_close(&y, &x, 1e-11 * (n as f64).max(1.0));
        }
    }

    #[test]
    fn impulse_transforms_to_a_flat_spectrum() {
        let mut x = vec![Complex::ZERO; 16];
        x[0] = Complex::new(1.0, 0.0);
        fft(&mut x);
        for v in &x {
            assert!((v.re - 1.0).abs() < 1e-12 && v.im.abs() < 1e-12);
        }
    }

    #[test]
    fn parseval_energy_is_preserved() {
        let x = test_vector(64, 17);
        let time: f64 = x.iter().map(|v| v.re * v.re + v.im * v.im).sum();
        let mut y = x.clone();
        fft(&mut y);
        let freq: f64 = y.iter().map(|v| v.re * v.re + v.im * v.im).sum();
        assert!((freq / 64.0 - time).abs() < 1e-10, "{freq} vs {time}");
    }

    #[test]
    fn convolver_matches_the_naive_convolution() {
        use crate::core::montecarlo::splitmix64;
        let uniform = |seed: u64, i: usize| -> f64 {
            splitmix64(seed ^ i as u64) as f64 / u64::MAX as f64 - 0.5
        };
        let kernel: Vec<f64> = (0..7).map(|i| uniform(3, i)).collect();
        let signal: Vec<f64> = (0..13).map(|i| uniform(5, i)).collect();
        let plan = RealConvolver::new(&kernel, signal.len());
        let mut out = vec![0.0; signal.len()];
        let mut scratch = Vec::new();
        plan.convolve_into(&signal, &mut out, &mut scratch);
        for (i, &o) in out.iter().enumerate() {
            let want: f64 = kernel
                .iter()
                .enumerate()
                .filter(|(j, _)| *j <= i && i - *j < signal.len())
                .map(|(j, &g)| g * signal[i - j])
                .sum();
            assert!((o - want).abs() < 1e-12, "lag {i}: {o} vs {want}");
        }
        // the plan and scratch are reusable: a second signal gets the
        // same treatment without re-planning
        let signal2: Vec<f64> = (0..13).map(|i| uniform(9, i)).collect();
        plan.convolve_into(&signal2, &mut out, &mut scratch);
        let want0 = kernel[0] * signal2[0];
        assert!((out[0] - want0).abs() < 1e-12);
    }

    #[test]
    fn convolver_kernel_with_leading_zero_shifts_the_output() {
        // a pure-delay kernel [0, 1] reproduces the signal one lag late —
        // the exact shape the rough Bergomi lag kernel has (lag 0 is
        // handled exactly outside the convolution)
        let plan = RealConvolver::new(&[0.0, 1.0], 4);
        let mut out = vec![0.0; 5];
        let mut scratch = Vec::new();
        plan.convolve_into(&[1.0, 2.0, 3.0, 4.0], &mut out, &mut scratch);
        for (o, w) in out.iter().zip(&[0.0, 1.0, 2.0, 3.0, 4.0]) {
            assert!((o - w).abs() < 1e-12);
        }
    }
}
