use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;
use rand::prelude::*;

use crate::bench_low::BenchResult;

pub fn run_seq_read(processes: usize, intensity: u32) -> BenchResult {
    let capped = intensity.min(4) as usize;
    let buf_size = 256 * 1024 * 1024 * (1 << capped); // 256MB base
    let buf: Vec<u8> = vec![0xAB; buf_size];
    let chunk_size = buf_size / processes;

    let start = Instant::now();
    let sum: u64 = buf.par_chunks(chunk_size)
        .map(|chunk| {
            let mut s: u64 = 0;
            for &b in chunk {
                s = s.wrapping_add(b as u64);
            }
            s
        })
        .sum();
    black_box(sum);
    let duration = start.elapsed();

    let gbps = buf_size as f64 / duration.as_secs_f64() / 1e9;
    BenchResult::new("Seq Read Bandwidth", duration, "GB/s", gbps)
}

pub fn run_seq_write(processes: usize, intensity: u32) -> BenchResult {
    let capped = intensity.min(4) as usize;
    let buf_size = 256 * 1024 * 1024 * (1 << capped);
    let mut buf: Vec<u8> = vec![0u8; buf_size];
    let chunk_size = buf_size / processes;

    let start = Instant::now();
    buf.par_chunks_mut(chunk_size)
        .for_each(|chunk| {
            for (i, b) in chunk.iter_mut().enumerate() {
                *b = (i & 0xFF) as u8;
            }
        });
    black_box(&buf);
    let duration = start.elapsed();

    let gbps = buf_size as f64 / duration.as_secs_f64() / 1e9;
    BenchResult::new("Seq Write Bandwidth", duration, "GB/s", gbps)
}

pub fn run_memory_latency(_processes: usize, intensity: u32) -> BenchResult {
    // Pointer-chase through shuffled array — inherently single-threaded
    let capped = intensity.min(4) as usize;
    let size = 16 * 1024 * 1024 * (1 << capped); // 16M entries base

    // Build a shuffled linked list using Fisher-Yates
    let mut indices: Vec<usize> = (0..size).collect();
    let mut rng = StdRng::seed_from_u64(42);
    for i in (1..size).rev() {
        let j = rng.gen_range(0..=i);
        indices.swap(i, j);
    }

    // Convert permutation to next-pointer array
    let mut chain = vec![0usize; size];
    for i in 0..size - 1 {
        chain[indices[i]] = indices[i + 1];
    }
    chain[indices[size - 1]] = indices[0]; // close the loop

    let chase_steps = size * 2;
    let start = Instant::now();
    let mut idx = 0usize;
    for _ in 0..chase_steps {
        idx = chain[idx];
    }
    black_box(idx);
    let duration = start.elapsed();

    let ns_per_access = duration.as_nanos() as f64 / chase_steps as f64;
    BenchResult::new("Memory Latency", duration, "ns", ns_per_access)
}

pub fn run_all_memory(processes: usize, intensity: u32) -> Vec<BenchResult> {
    vec![
        run_seq_read(processes, intensity),
        run_seq_write(processes, intensity),
        run_memory_latency(processes, intensity),
    ]
}
