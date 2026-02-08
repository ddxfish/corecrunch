use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;
use rand::prelude::*;
use sha2::{Sha256, Digest};
use std::f64::consts::PI;

use crate::bench_low::BenchResult;

// ── Pure-Rust Cooley-Tukey FFT ──────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Complex64 {
    re: f64,
    im: f64,
}

impl Complex64 {
    fn new(re: f64, im: f64) -> Self { Self { re, im } }

    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    fn add(self, other: Self) -> Self {
        Self { re: self.re + other.re, im: self.im + other.im }
    }

    fn sub(self, other: Self) -> Self {
        Self { re: self.re - other.re, im: self.im - other.im }
    }
}

fn bit_reverse(mut x: usize, log_n: u32) -> usize {
    let mut result = 0;
    for _ in 0..log_n {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

fn fft_in_place(buf: &mut [Complex64]) {
    let n = buf.len();
    let log_n = (n as f64).log2() as u32;

    // Bit-reversal permutation
    for i in 0..n {
        let j = bit_reverse(i, log_n);
        if i < j {
            buf.swap(i, j);
        }
    }

    // Cooley-Tukey butterfly
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * PI / len as f64;
        let wn = Complex64::new(angle.cos(), angle.sin());

        for start in (0..n).step_by(len) {
            let mut w = Complex64::new(1.0, 0.0);
            for j in 0..half {
                let u = buf[start + j];
                let v = buf[start + j + half].mul(w);
                buf[start + j] = u.add(v);
                buf[start + j + half] = u.sub(v);
                w = w.mul(wn);
            }
        }
        len *= 2;
    }
}

pub fn run_fft_benchmark(processes: usize, intensity: u32) -> BenchResult {
    let base_size = 8192;
    let size = base_size * (2_u32.pow(intensity)) as usize;

    let mut rng = StdRng::seed_from_u64(42);
    let mut buffer: Vec<Complex64> = (0..size)
        .map(|_| Complex64::new(rng.gen(), rng.gen()))
        .collect();

    let chunk_size = size / processes;
    let start = Instant::now();

    for _ in 0..5 {
        buffer.par_chunks_mut(chunk_size)
            .for_each(|chunk| {
                // Ensure chunk size is power of 2 for FFT
                let n = chunk.len().next_power_of_two().min(chunk.len());
                if n > 1 && (n & (n - 1)) == 0 {
                    fft_in_place(&mut chunk[..n]);
                }
            });
    }
    black_box(&buffer);

    let duration = start.elapsed();
    let operations = size as u64 * (size as f64).log2() as u64 * 5;
    let gflops = (operations as f64) / duration.as_secs_f64() / 1e9;

    BenchResult::new("FFT", duration, "GFLOPS", gflops)
}

// ── SHA-256 (software, via sha2 crate) ──────────────────────────────────────

pub fn run_sha256_benchmark(processes: usize, intensity: u32) -> BenchResult {
    let base_size = 16 * 1024 * 1024; // 16MB
    let size = base_size * (2_u32.pow(intensity)) as usize;

    let data: Vec<u8> = (0..size).map(|i| i as u8).collect();
    let chunk_size = size / processes;

    let start = Instant::now();

    for _ in 0..3 {
        data.par_chunks(chunk_size)
            .for_each(|chunk| {
                let mut hasher = Sha256::new();
                hasher.update(chunk);
                let hash = hasher.finalize();
                black_box(hash);
            });
    }

    let duration = start.elapsed();
    let gbps = (size as f64 * 3.0) / duration.as_secs_f64() / 1e9;

    BenchResult::new("SHA-256", duration, "GB/s", gbps)
}

// ── Matrix Multiplication ───────────────────────────────────────────────────

pub fn run_matrix_benchmark(_processes: usize, intensity: u32) -> BenchResult {
    let base_size = 1024;
    let size = (base_size as f64 * 1.2f64.powi(intensity as i32)) as usize;

    let mut rng = StdRng::seed_from_u64(42);
    let a: Vec<f64> = (0..size * size).map(|_| rng.gen()).collect();
    let b: Vec<f64> = (0..size * size).map(|_| rng.gen()).collect();
    let mut c = vec![0.0f64; size * size];

    let block_size = 32;
    let start = Instant::now();

    let blocks: Vec<_> = (0..size)
        .step_by(block_size)
        .flat_map(|i| (0..size).step_by(block_size).map(move |j| (i, j)))
        .collect();

    // Collect block results then merge (avoids Mutex overhead from v1)
    let block_results: Vec<_> = blocks.par_iter().map(|&(i, j)| {
        let i_end = (i + block_size).min(size);
        let j_end = (j + block_size).min(size);
        let rows = i_end - i;
        let cols = j_end - j;
        let mut block = vec![0.0f64; rows * cols];

        for k in (0..size).step_by(block_size) {
            let k_end = (k + block_size).min(size);
            for ii in 0..rows {
                for kk in k..k_end {
                    let a_val = a[(i + ii) * size + kk];
                    for jj in 0..cols {
                        block[ii * cols + jj] += a_val * b[kk * size + (j + jj)];
                    }
                }
            }
        }

        (i, j, i_end, j_end, block)
    }).collect();

    for (i, j, i_end, j_end, block) in block_results {
        let cols = j_end - j;
        for ii in i..i_end {
            for jj in j..j_end {
                c[ii * size + jj] += block[(ii - i) * cols + (jj - j)];
            }
        }
    }

    black_box(&c);
    let duration = start.elapsed();
    let operations = 2 * size as u64 * size as u64 * size as u64;
    let gflops = (operations as f64) / duration.as_secs_f64() / 1e9;

    BenchResult::new("Matrix Multiplication", duration, "GFLOPS", gflops)
}

// ── Floating Point Ops ──────────────────────────────────────────────────────

pub fn run_fp_benchmark(processes: usize, intensity: u32) -> BenchResult {
    let base_iterations = 10_000_000;
    let iterations = base_iterations * (2_u32.pow(intensity)) as usize;
    let chunk_size = iterations / processes;

    let start = Instant::now();

    (0..iterations).into_par_iter()
        .chunks(chunk_size)
        .for_each(|chunk| {
            let mut rng = StdRng::seed_from_u64(42);
            for _ in chunk {
                let x: f64 = rng.gen();
                let y = x.sin().cos().tan().exp().ln();
                let z = y.sqrt().cbrt().exp2().log2().log10();
                black_box(z.hypot(x));
            }
        });

    let duration = start.elapsed();
    let operations = iterations as u64 * 12;
    let gflops = (operations as f64) / duration.as_secs_f64() / 1e9;

    BenchResult::new("Floating Point Ops", duration, "GFLOPS", gflops)
}

// ── Monte Carlo Pi ──────────────────────────────────────────────────────────

pub fn run_monte_carlo_pi(processes: usize, intensity: u32) -> BenchResult {
    let base_points = 10_000_000;
    let total_points = base_points * (2_u32.pow(intensity)) as usize;
    let points_per_process = total_points / processes;

    let start = Instant::now();

    let inside_count: usize = (0..processes).into_par_iter()
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(42 + i as u64);
            let mut local_inside = 0;
            for _ in 0..points_per_process {
                let x: f64 = rng.gen();
                let y: f64 = rng.gen();
                if x * x + y * y <= 1.0 {
                    local_inside += 1;
                }
            }
            local_inside
        })
        .sum();

    let pi_estimate = 4.0 * (inside_count as f64) / (total_points as f64);
    let duration = start.elapsed();
    let error = ((pi_estimate - PI).abs() / PI * 100.0).abs();

    BenchResult::new("Monte Carlo Pi", duration, "Error %", error)
}

// ── Prime Sieve ─────────────────────────────────────────────────────────────

pub fn run_prime_sieve(processes: usize, intensity: u32) -> BenchResult {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let base_n = 10_000_000;
    let max_n = (base_n as f64 * 1.5f64.powi(intensity as i32)) as usize;

    let start = Instant::now();

    let sqrt_n = (max_n as f64).sqrt() as usize;
    let mut small_primes = vec![true; sqrt_n + 1];
    small_primes[0] = false;
    if sqrt_n > 0 { small_primes[1] = false; }

    for i in 2..=(sqrt_n as f64).sqrt() as usize {
        if small_primes[i] {
            for j in (i * i..=sqrt_n).step_by(i) {
                small_primes[j] = false;
            }
        }
    }

    let small_prime_list: Vec<usize> = (2..=sqrt_n)
        .filter(|&i| small_primes[i])
        .collect();

    let chunk_size = (max_n - sqrt_n) / processes + 1;
    let prime_count = AtomicUsize::new(small_prime_list.len());

    let ranges: Vec<_> = (sqrt_n + 1..=max_n)
        .step_by(chunk_size)
        .map(|s| s..=(s + chunk_size - 1).min(max_n))
        .collect();

    ranges.par_iter().for_each(|range| {
        let range_start = *range.start();
        let range_end = *range.end();
        let mut segment = vec![true; range_end - range_start + 1];

        for &prime in &small_prime_list {
            let first_multiple = ((range_start + prime - 1) / prime) * prime;
            if first_multiple <= range_end {
                let first_index = first_multiple.saturating_sub(range_start);
                for i in (first_index..segment.len()).step_by(prime) {
                    segment[i] = false;
                }
            }
        }

        let count = segment.iter().filter(|&&x| x).count();
        prime_count.fetch_add(count, Ordering::Relaxed);
    });

    let total_primes = prime_count.load(Ordering::Relaxed);
    let duration = start.elapsed();

    BenchResult::new("Prime Sieve", duration, "Primes", total_primes as f64)
}

// ── N-Body Simulation ───────────────────────────────────────────────────────

pub fn run_nbody_sim(processes: usize, intensity: u32) -> BenchResult {
    let base_bodies = 2000;
    let num_bodies = (base_bodies as f64 * 1.3f64.powi(intensity as i32)) as usize;
    let num_steps = 25;

    #[derive(Clone)]
    struct Body {
        x: f64, y: f64, z: f64,
        vx: f64, vy: f64, vz: f64,
        mass: f64,
    }

    let mut rng = StdRng::seed_from_u64(42);
    let mut bodies: Vec<Body> = (0..num_bodies)
        .map(|_| Body {
            x: rng.gen_range(-100.0..100.0),
            y: rng.gen_range(-100.0..100.0),
            z: rng.gen_range(-100.0..100.0),
            vx: rng.gen_range(-1.0..1.0),
            vy: rng.gen_range(-1.0..1.0),
            vz: rng.gen_range(-1.0..1.0),
            mass: rng.gen_range(0.1..100.0),
        })
        .collect();

    let start = Instant::now();

    for _ in 0..num_steps {
        let snapshot = bodies.clone();
        let chunk_size = (num_bodies + processes - 1) / processes;

        bodies.par_chunks_mut(chunk_size)
            .for_each(|chunk| {
                for body in chunk.iter_mut() {
                    let mut fx = 0.0;
                    let mut fy = 0.0;
                    let mut fz = 0.0;

                    for other in &snapshot {
                        let dx = other.x - body.x;
                        let dy = other.y - body.y;
                        let dz = other.z - body.z;
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                        if dist > 0.0001 {
                            let f = (other.mass * body.mass) / (dist * dist);
                            fx += f * dx / dist;
                            fy += f * dy / dist;
                            fz += f * dz / dist;
                        }
                    }

                    body.vx += fx;
                    body.vy += fy;
                    body.vz += fz;
                    body.x += body.vx;
                    body.y += body.vy;
                    body.z += body.vz;
                }
            });
    }

    black_box(&bodies);
    let duration = start.elapsed();
    let operations = num_bodies as u64 * num_bodies as u64 * num_steps as u64;
    let gflops = (operations as f64 * 20.0) / duration.as_secs_f64() / 1e9;

    BenchResult::new("N-Body Simulation", duration, "GFLOPS", gflops)
}

// ── Runner ──────────────────────────────────────────────────────────────────

pub fn run_all_legacy(processes: usize, intensity: u32) -> Vec<BenchResult> {
    vec![
        run_fft_benchmark(processes, intensity),
        run_sha256_benchmark(processes, intensity),
        run_matrix_benchmark(processes, intensity),
        run_fp_benchmark(processes, intensity),
        run_monte_carlo_pi(processes, intensity),
        run_prime_sieve(processes, intensity),
        run_nbody_sim(processes, intensity),
    ]
}
