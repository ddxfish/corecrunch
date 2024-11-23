use rayon::prelude::*;
use std::time::{Duration, Instant};
use rustfft::{FftPlanner, num_complex::Complex};
use sha2::{Sha256, Digest};
use rand::prelude::*;

pub struct BenchResult {
    pub name: String,
    pub duration: Duration,
    pub operations: u64,
    pub metric_name: String,
    pub metric_value: f64,
}

impl BenchResult {
    pub fn new(name: &str, duration: Duration, operations: u64, metric_name: &str, metric_value: f64) -> Self {
        Self {
            name: name.to_string(),
            duration,
            operations,
            metric_name: metric_name.to_string(),
            metric_value,
        }
    }
}

pub fn run_fft_benchmark(processes: usize, intensity: u32) -> BenchResult {
    // Increased base size significantly
    let base_size = 8192;
    let size = base_size * (2_u32.pow(intensity)) as usize;
    
    let mut planner = FftPlanner::<f64>::new();
    let _fft = planner.plan_fft_forward(size);
    
    let mut rng = StdRng::seed_from_u64(42);
    let mut buffer: Vec<Complex<f64>> = (0..size)
        .map(|_| Complex::new(rng.gen(), rng.gen()))
        .collect();
    
    let chunk_size = size / processes;
    let start = Instant::now();
    
    // Increased number of iterations
    for _ in 0..5 {
        buffer.par_chunks_mut(chunk_size)
            .for_each(|chunk| {
                let mut local_planner = FftPlanner::<f64>::new();
                let local_fft = local_planner.plan_fft_forward(chunk.len());
                local_fft.process(chunk);
            });
    }
    
    let duration = start.elapsed();
    let operations = size as u64 * (size as f64).log2() as u64 * 5; // Adjusted for iterations
    let gflops = (operations as f64) / duration.as_secs_f64() / 1e9;
    
    BenchResult::new("FFT", duration, operations, "GFLOPS", gflops)
}

pub fn run_sha256_benchmark(processes: usize, intensity: u32) -> BenchResult {
    // Increased base size to 16MB
    let base_size = 16 * 1024 * 1024;
    let size = base_size * (2_u32.pow(intensity)) as usize;
    
    let data: Vec<u8> = (0..size).map(|i| i as u8).collect();
    let chunk_size = size / processes;
    
    let start = Instant::now();
    
    // Increased number of iterations
    for _ in 0..3 {
        data.par_chunks(chunk_size)
            .for_each(|chunk| {
                let mut hasher = Sha256::new();
                hasher.update(chunk);
                let _hash = hasher.finalize();
            });
    }
    
    let duration = start.elapsed();
    let operations = size as u64 * 3; // Adjusted for iterations
    let gbps = (size as f64 * 3.0) / duration.as_secs_f64() / 1e9;
    
    BenchResult::new("SHA-256", duration, operations, "GB/s", gbps)
}

pub fn run_matrix_benchmark(_processes: usize, intensity: u32) -> BenchResult {
    use rayon::prelude::*;
    use std::time::Instant;
    use std::sync::Mutex;
    
    // Increased base size
    let base_size = 1024;
    let size = (base_size as f64 * (1.2f64.powi(intensity as i32))) as usize;
    
    let a: Vec<f64> = (0..size * size).map(|_| rand::random()).collect();
    let b: Vec<f64> = (0..size * size).map(|_| rand::random()).collect();
    let c = Mutex::new(vec![0.0; size * size]);
    
    // Block size chosen to fit L1 cache
    let block_size = 32;
    let start = Instant::now();
    
    // Split the work into blocks that can be processed independently
    let blocks: Vec<_> = (0..size)
        .step_by(block_size)
        .flat_map(|i| {
            (0..size).step_by(block_size).map(move |j| (i, j))
        })
        .collect();
    
    blocks.par_iter().for_each(|&(i, j)| {
        let i_end = (i + block_size).min(size);
        let j_end = (j + block_size).min(size);
        
        // Process one block
        let mut block_result = vec![0.0; (i_end - i) * (j_end - j)];
        
        // Compute the block
        for k in (0..size).step_by(block_size) {
            let k_end = (k + block_size).min(size);
            
            for ii in i..i_end {
                for kk in k..k_end {
                    let a_val = a[ii * size + kk];
                    for jj in j..j_end {
                        block_result[(ii - i) * (j_end - j) + (jj - j)] += 
                            a_val * b[kk * size + jj];
                    }
                }
            }
        }
        
        // Update the result matrix
        let mut c = c.lock().unwrap();
        for ii in i..i_end {
            for jj in j..j_end {
                let idx = ii * size + jj;
                c[idx] += block_result[(ii - i) * (j_end - j) + (jj - j)];
            }
        }
    });
    
    let duration = start.elapsed();
    let operations = 2 * size as u64 * size as u64 * size as u64;
    let gflops = (operations as f64) / duration.as_secs_f64() / 1e9;
    
    BenchResult::new("Matrix Multiplication", duration, operations, "GFLOPS", gflops)
}

pub fn run_fp_benchmark(processes: usize, intensity: u32) -> BenchResult {
    // Increased base iterations significantly
    let base_iterations = 10_000_000;
    let iterations = base_iterations * (2_u32.pow(intensity)) as usize;
    let chunk_size = iterations / processes;
    
    let start = Instant::now();
    
    // Added more operations per iteration
    (0..iterations).into_par_iter()
        .chunks(chunk_size)
        .for_each(|chunk| {
            for _ in chunk {
                let x = rand::random::<f64>();
                let y = x.sin().cos().tan().exp().ln();
                let z = y.sqrt().cbrt().exp2().log2().log10();
                let _ = z.hypot(x);
            }
        });
    
    let duration = start.elapsed();
    let operations = iterations as u64 * 12; // Adjusted for added operations
    let gflops = (operations as f64) / duration.as_secs_f64() / 1e9;
    
    BenchResult::new("Floating Point Ops", duration, operations, "GFLOPS", gflops)
}

pub fn run_all_low_level(processes: usize, intensity: u32) -> Vec<BenchResult> {
    vec![
        run_fft_benchmark(processes, intensity),
        run_sha256_benchmark(processes, intensity),
        run_matrix_benchmark(processes, intensity),
        run_fp_benchmark(processes, intensity),
    ]
}