use rayon::prelude::*;
use std::hint::black_box;
use std::time::Instant;
use rand::prelude::*;
use flate2::Compression;
use flate2::write::{GzEncoder, GzDecoder};
use std::io::Write;

use crate::bench_low::BenchResult;

// ── LLM Inference Simulation ────────────────────────────────────────────────

pub fn run_llm_inference(processes: usize, intensity: u32) -> BenchResult {
    let d_model = 512;
    let num_layers = 4 + intensity as usize;
    let seq_len = 128 * intensity as usize;

    // Input activations: seq_len x d_model
    let mut rng = StdRng::seed_from_u64(42);
    let mut activations: Vec<f32> = (0..seq_len * d_model)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    // Weight matrices: one per layer, d_model x d_model
    let weights: Vec<Vec<f32>> = (0..num_layers)
        .map(|_| (0..d_model * d_model).map(|_| rng.gen_range(-0.1..0.1)).collect())
        .collect();

    let chunk_size = (seq_len + processes - 1) / processes;

    let start = Instant::now();

    for layer in 0..num_layers {
        let input = activations.clone();
        let w = &weights[layer];

        // Matrix multiply: output[row][col] = sum(input[row][k] * w[k][col])
        // Parallelize over output rows
        activations
            .par_chunks_mut(d_model * chunk_size.max(1))
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let row_start = chunk_idx * chunk_size;
                let rows_in_chunk = out_chunk.len() / d_model;
                for r in 0..rows_in_chunk {
                    let row = row_start + r;
                    for col in 0..d_model {
                        let mut sum = 0.0f32;
                        for k in 0..d_model {
                            sum += input[row * d_model + k] * w[k * d_model + col];
                        }
                        // ReLU
                        out_chunk[r * d_model + col] = if sum > 0.0 { sum } else { 0.0 };
                    }
                }
            });
    }
    black_box(&activations);
    let duration = start.elapsed();

    // 2 * seq_len * d_model * d_model per layer (matmul FLOPs)
    let flops = 2.0 * seq_len as f64 * d_model as f64 * d_model as f64 * num_layers as f64;
    let gflops = flops / duration.as_secs_f64() / 1e9;

    BenchResult::new("LLM Inference Sim", duration, "GFLOPS", gflops)
}

// ── Gzip Compression ────────────────────────────────────────────────────────

pub fn run_gzip_benchmark(processes: usize, intensity: u32) -> BenchResult {
    let base_size = 8 * 1024 * 1024; // 8MB
    let total_size = base_size * intensity as usize;

    // Semi-realistic data with patterns (like logs/text)
    let mut rng = StdRng::seed_from_u64(42);
    let patterns: Vec<&[u8]> = vec![
        b"INFO: Request processed in ",
        b"DEBUG: Cache hit for key=",
        b"WARN: Connection timeout after ",
        b"ERROR: Failed to parse response: ",
        b"INFO: User logged in from ",
    ];
    let mut data = Vec::with_capacity(total_size);
    while data.len() < total_size {
        let pattern = patterns[rng.gen_range(0..patterns.len())];
        data.extend_from_slice(pattern);
        // Add some random digits
        for _ in 0..rng.gen_range(4..20) {
            data.push(b'0' + rng.gen_range(0..10));
        }
        data.push(b'\n');
    }
    data.truncate(total_size);

    let chunk_size = total_size / processes.max(1);

    let start = Instant::now();

    // Each thread compresses then decompresses its chunk
    data.par_chunks(chunk_size)
        .for_each(|chunk| {
            // Compress
            let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
            encoder.write_all(chunk).unwrap();
            let compressed = encoder.finish().unwrap();

            // Decompress
            let mut decoder = GzDecoder::new(Vec::new());
            decoder.write_all(&compressed).unwrap();
            let decompressed = decoder.finish().unwrap();
            black_box(decompressed.len());
        });

    let duration = start.elapsed();

    // Total bytes processed = compress + decompress = 2x
    let mbps = (total_size as f64 * 2.0) / duration.as_secs_f64() / 1e6;

    BenchResult::new("Gzip Compress", duration, "MB/s", mbps)
}

// ── Image Processing — Gaussian Blur ────────────────────────────────────────

pub fn run_image_blur(processes: usize, intensity: u32) -> BenchResult {
    let width = 1920 * intensity as usize;
    let height = 1080;
    let channels = 3; // RGB

    // Synthetic image data
    let mut rng = StdRng::seed_from_u64(42);
    let input: Vec<u8> = (0..width * height * channels)
        .map(|_| rng.gen::<u8>())
        .collect();
    let mut output = vec![0u8; width * height * channels];

    // 5x5 Gaussian kernel (normalized to sum ~1.0 via integer weights / 256)
    let kernel: [[u16; 5]; 5] = [
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1],
    ];
    let kernel_sum: u16 = kernel.iter().flat_map(|r| r.iter()).sum(); // 273

    let row_stride = width * channels;
    let rows_per_chunk = (height + processes - 1) / processes;

    let start = Instant::now();

    output
        .par_chunks_mut(rows_per_chunk * row_stride)
        .enumerate()
        .for_each(|(chunk_idx, out_chunk)| {
            let y_start = chunk_idx * rows_per_chunk;
            let rows = out_chunk.len() / row_stride;

            for local_y in 0..rows {
                let y = y_start + local_y;
                for x in 0..width {
                    for c in 0..channels {
                        let mut acc: u32 = 0;
                        for ky in 0..5i32 {
                            for kx in 0..5i32 {
                                let sy = (y as i32 + ky - 2).clamp(0, height as i32 - 1) as usize;
                                let sx = (x as i32 + kx - 2).clamp(0, width as i32 - 1) as usize;
                                acc += input[sy * row_stride + sx * channels + c] as u32
                                    * kernel[ky as usize][kx as usize] as u32;
                            }
                        }
                        out_chunk[local_y * row_stride + x * channels + c] =
                            (acc / kernel_sum as u32).min(255) as u8;
                    }
                }
            }
        });

    black_box(&output);
    let duration = start.elapsed();

    let total_pixels = width as f64 * height as f64;
    let mpixs = total_pixels / duration.as_secs_f64() / 1e6;

    BenchResult::new("Image Blur (5x5)", duration, "Mpix/s", mpixs)
}

// ── Large Dataset Sort ──────────────────────────────────────────────────────

pub fn run_dataset_sort(processes: usize, intensity: u32) -> BenchResult {
    let base_records = 2_000_000;
    let num_records = base_records * intensity as usize;

    struct Record {
        _id: u64,
        value: f64,
        category: u32,
        _name: [u8; 32],
    }

    let mut rng = StdRng::seed_from_u64(42);
    let mut records: Vec<Record> = (0..num_records)
        .map(|i| {
            let mut name = [0u8; 32];
            for b in &mut name {
                *b = rng.gen_range(b'a'..=b'z');
            }
            Record {
                _id: i as u64,
                value: rng.gen(),
                category: rng.gen_range(0..100),
                _name: name,
            }
        })
        .collect();

    // Build a rayon pool to control parallelism for par_sort
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(processes)
        .build()
        .unwrap();

    let start = Instant::now();
    pool.install(|| {
        records.par_sort_unstable_by(|a, b| {
            a.category
                .cmp(&b.category)
                .then(b.value.partial_cmp(&a.value).unwrap())
        });
    });
    black_box(&records);
    let duration = start.elapsed();

    let mrecs = num_records as f64 / duration.as_secs_f64() / 1e6;

    BenchResult::new("Dataset Sort", duration, "Mrec/s", mrecs)
}

// ── Runner ──────────────────────────────────────────────────────────────────

pub fn run_all_real_world(processes: usize, intensity: u32) -> Vec<BenchResult> {
    vec![
        run_llm_inference(processes, intensity),
        run_gzip_benchmark(processes, intensity),
        run_image_blur(processes, intensity),
        run_dataset_sort(processes, intensity),
    ]
}
