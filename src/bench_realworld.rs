use rayon::prelude::*;
use std::time::Instant;
use rand::prelude::*;
use std::f64::consts::PI;

use crate::bench_low::BenchResult;

pub fn run_monte_carlo_pi(processes: usize, intensity: u32) -> BenchResult {
    // Increased base points significantly
    let base_points = 10_000_000;
    let total_points = base_points * (2_u32.pow(intensity)) as usize;
    let points_per_process = total_points / processes;
    
    let start = Instant::now();
    
    let inside_count: usize = (0..processes).into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
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
    
    BenchResult::new(
        "Monte Carlo Pi",
        duration,
        "Error %",
        error,
    )
}

pub fn run_prime_sieve(processes: usize, intensity: u32) -> BenchResult {
    use rayon::prelude::*;
    use std::time::Instant;
    use std::sync::atomic::{AtomicUsize, Ordering};
    
    // Increased base size significantly
    let base_n = 10_000_000;
    let max_n = (base_n as f64 * 1.5f64.powi(intensity as i32)) as usize;
    
    let start = Instant::now();
    
    // Calculate small primes up to sqrt(max_n) sequentially
    let sqrt_n = (max_n as f64).sqrt() as usize;
    let mut small_primes = vec![true; sqrt_n + 1];
    small_primes[0] = false;
    small_primes[1] = false;
    
    for i in 2..=(sqrt_n as f64).sqrt() as usize {
        if small_primes[i] {
            for j in (i * i..=sqrt_n).step_by(i) {
                small_primes[j] = false;
            }
        }
    }
    
    // Collect actual small primes
    let small_prime_list: Vec<usize> = (2..=sqrt_n)
        .filter(|&i| small_primes[i])
        .collect();
    
    // Split the remaining range into chunks and process in parallel
    let chunk_size = (max_n - sqrt_n) / processes + 1;
    let prime_count = AtomicUsize::new(small_prime_list.len());
    
    // Create ranges that can be processed in parallel
    let ranges: Vec<_> = (sqrt_n + 1..=max_n)
        .step_by(chunk_size)
        .map(|start| start..=(start + chunk_size - 1).min(max_n))
        .collect();
    
    ranges.par_iter().for_each(|range| {
        let start = *range.start(); // Dereference here to get the value
        let end = *range.end();     // Dereference here too
        let mut segment = vec![true; end - start + 1];
        
        // Mark composites in this segment using small primes
        for &prime in &small_prime_list {
            let first_multiple = ((start + prime - 1) / prime) * prime;
            if first_multiple <= end {
                let first_index = first_multiple.saturating_sub(start);
                for i in (first_index..segment.len()).step_by(prime) {
                    segment[i] = false;
                }
            }
        }
        
        // Count primes in this segment
        let count = segment.iter().filter(|&&x| x).count();
        prime_count.fetch_add(count, Ordering::Relaxed);
    });
    
    let total_primes = prime_count.load(Ordering::Relaxed);
    let duration = start.elapsed();
    
    BenchResult::new(
        "Prime Sieve",
        duration,
        "Primes",
        total_primes as f64,
    )
}

pub fn run_nbody_sim(processes: usize, intensity: u32) -> BenchResult {
    // Increased base bodies and steps
    let base_bodies = 2000;
    let num_bodies = (base_bodies as f64 * 1.3f64.powi(intensity as i32)) as usize;
    let num_steps = 25; // Increased from 10
    
    #[derive(Clone)]
    struct Body {
        x: f64,
        y: f64,
        z: f64,
        vx: f64,
        vy: f64,
        vz: f64,
        mass: f64,
    }
    
    let mut bodies: Vec<Body> = (0..num_bodies)
        .map(|_| {
            let mut rng = rand::thread_rng();
            Body {
                x: rng.gen_range(-100.0..100.0),
                y: rng.gen_range(-100.0..100.0),
                z: rng.gen_range(-100.0..100.0),
                vx: rng.gen_range(-1.0..1.0),
                vy: rng.gen_range(-1.0..1.0),
                vz: rng.gen_range(-1.0..1.0),
                mass: rng.gen_range(0.1..100.0),
            }
        })
        .collect();
    
    let start = Instant::now();
    
    for _ in 0..num_steps {
        let bodies_snapshot = bodies.clone();
        let chunk_size = (num_bodies + processes - 1) / processes;
        
        bodies.par_chunks_mut(chunk_size)
            .for_each(|chunk| {
                for body in chunk {
                    let mut fx = 0.0;
                    let mut fy = 0.0;
                    let mut fz = 0.0;
                    
                    for other in &bodies_snapshot {
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
    
    let duration = start.elapsed();
    let operations = num_bodies as u64 * num_bodies as u64 * num_steps as u64;
    let gflops = (operations as f64 * 20.0) / duration.as_secs_f64() / 1e9;
    
    BenchResult::new(
        "N-Body Simulation",
        duration,
        "GFLOPS",
        gflops,
    )
}

pub fn run_all_real_world(processes: usize, intensity: u32) -> Vec<BenchResult> {
    vec![
        run_monte_carlo_pi(processes, intensity),
        run_prime_sieve(processes, intensity),
        run_nbody_sim(processes, intensity),
    ]
}