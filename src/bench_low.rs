use std::hint::black_box;
use std::time::{Duration, Instant};

pub struct BenchResult {
    pub name: String,
    pub duration: Duration,
    pub metric_name: String,
    pub metric_value: f64,
    pub supported: bool,
}

impl BenchResult {
    pub fn new(name: &str, duration: Duration, metric_name: &str, metric_value: f64) -> Self {
        Self {
            name: name.to_string(),
            duration,
            metric_name: metric_name.to_string(),
            metric_value,
            supported: true,
        }
    }

    pub fn not_supported(name: &str) -> Self {
        Self {
            name: name.to_string(),
            duration: Duration::ZERO,
            metric_name: String::new(),
            metric_value: 0.0,
            supported: false,
        }
    }
}

// ── AVX2 FP Throughput ──────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub fn run_avx2_fp(processes: usize, intensity: u32) -> BenchResult {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return BenchResult::not_supported("AVX2 FP Throughput");
    }
    let base_iterations: u64 = 50_000_000;
    let iterations = base_iterations * (1u64 << intensity);
    let iters_per_thread = iterations / processes as u64;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(processes)
        .build()
        .unwrap();

    let start = Instant::now();
    pool.scope(|s| {
        for _ in 0..processes {
            s.spawn(move |_| {
                unsafe { avx2_fp_kernel(iters_per_thread) };
            });
        }
    });
    let duration = start.elapsed();

    // 4 accumulators * 4 doubles per register * 2 ops (mul+add) per FMA = 32 flops/iter
    let total_flops = iterations as f64 * 32.0;
    let gflops = total_flops / duration.as_secs_f64() / 1e9;

    BenchResult::new("AVX2 FP Throughput", duration, "GFLOPS", gflops)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_fp_kernel(iterations: u64) {
    use std::arch::x86_64::*;
    let mut acc0 = _mm256_set1_pd(1.0);
    let mut acc1 = _mm256_set1_pd(2.0);
    let mut acc2 = _mm256_set1_pd(3.0);
    let mut acc3 = _mm256_set1_pd(4.0);
    let mul = _mm256_set1_pd(1.0000001);

    for _ in 0..iterations {
        acc0 = _mm256_fmadd_pd(acc0, mul, acc1);
        acc1 = _mm256_fmadd_pd(acc1, mul, acc2);
        acc2 = _mm256_fmadd_pd(acc2, mul, acc3);
        acc3 = _mm256_fmadd_pd(acc3, mul, acc0);
    }

    black_box(acc0);
    black_box(acc1);
    black_box(acc2);
    black_box(acc3);
}

#[cfg(not(target_arch = "x86_64"))]
pub fn run_avx2_fp(_processes: usize, _intensity: u32) -> BenchResult {
    BenchResult::not_supported("AVX2 FP Throughput")
}

// ── AVX-512 FP Throughput ───────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub fn run_avx512_fp(processes: usize, intensity: u32) -> BenchResult {
    if !is_x86_feature_detected!("avx512f") {
        return BenchResult::not_supported("AVX-512 FP Throughput");
    }
    let base_iterations: u64 = 50_000_000;
    let iterations = base_iterations * (1u64 << intensity);
    let iters_per_thread = iterations / processes as u64;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(processes)
        .build()
        .unwrap();

    let start = Instant::now();
    pool.scope(|s| {
        for _ in 0..processes {
            s.spawn(move |_| {
                unsafe { avx512_fp_kernel(iters_per_thread) };
            });
        }
    });
    let duration = start.elapsed();

    // 4 accumulators * 8 doubles per register * 2 ops per FMA = 64 flops/iter
    let total_flops = iterations as f64 * 64.0;
    let gflops = total_flops / duration.as_secs_f64() / 1e9;

    BenchResult::new("AVX-512 FP Throughput", duration, "GFLOPS", gflops)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn avx512_fp_kernel(iterations: u64) {
    use std::arch::x86_64::*;
    let mut acc0 = _mm512_set1_pd(1.0);
    let mut acc1 = _mm512_set1_pd(2.0);
    let mut acc2 = _mm512_set1_pd(3.0);
    let mut acc3 = _mm512_set1_pd(4.0);
    let mul = _mm512_set1_pd(1.0000001);

    for _ in 0..iterations {
        acc0 = _mm512_fmadd_pd(acc0, mul, acc1);
        acc1 = _mm512_fmadd_pd(acc1, mul, acc2);
        acc2 = _mm512_fmadd_pd(acc2, mul, acc3);
        acc3 = _mm512_fmadd_pd(acc3, mul, acc0);
    }

    black_box(acc0);
    black_box(acc1);
    black_box(acc2);
    black_box(acc3);
}

#[cfg(not(target_arch = "x86_64"))]
pub fn run_avx512_fp(_processes: usize, _intensity: u32) -> BenchResult {
    BenchResult::not_supported("AVX-512 FP Throughput")
}

// ── AES-NI Throughput ───────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub fn run_aes_ni(processes: usize, intensity: u32) -> BenchResult {
    if !is_x86_feature_detected!("aes") {
        return BenchResult::not_supported("AES-NI Throughput");
    }
    let base_iterations: u64 = 50_000_000;
    let iterations = base_iterations * (1u64 << intensity);
    let iters_per_thread = iterations / processes as u64;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(processes)
        .build()
        .unwrap();

    let start = Instant::now();
    pool.scope(|s| {
        for _ in 0..processes {
            s.spawn(move |_| {
                unsafe { aes_ni_kernel(iters_per_thread) };
            });
        }
    });
    let duration = start.elapsed();

    // 4 blocks * 16 bytes per block per iteration
    let total_bytes = iterations as f64 * 4.0 * 16.0;
    let gbps = total_bytes / duration.as_secs_f64() / 1e9;

    BenchResult::new("AES-NI Throughput", duration, "GB/s", gbps)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "aes")]
unsafe fn aes_ni_kernel(iterations: u64) {
    use std::arch::x86_64::*;
    let key = _mm_set_epi64x(0x0123456789abcdef_u64 as i64, 0xfedcba9876543210_u64 as i64);
    let mut block0 = _mm_set_epi64x(1, 0);
    let mut block1 = _mm_set_epi64x(3, 2);
    let mut block2 = _mm_set_epi64x(5, 4);
    let mut block3 = _mm_set_epi64x(7, 6);

    for _ in 0..iterations {
        block0 = _mm_aesenc_si128(block0, key);
        block1 = _mm_aesenc_si128(block1, key);
        block2 = _mm_aesenc_si128(block2, key);
        block3 = _mm_aesenc_si128(block3, key);
    }

    black_box(block0);
    black_box(block1);
    black_box(block2);
    black_box(block3);
}

#[cfg(not(target_arch = "x86_64"))]
pub fn run_aes_ni(_processes: usize, _intensity: u32) -> BenchResult {
    BenchResult::not_supported("AES-NI Throughput")
}

// ── SHA Extensions ──────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub fn run_sha_ext(processes: usize, intensity: u32) -> BenchResult {
    if !is_x86_feature_detected!("sha") {
        return BenchResult::not_supported("SHA Extensions");
    }
    let base_iterations: u64 = 50_000_000;
    let iterations = base_iterations * (1u64 << intensity);
    let iters_per_thread = iterations / processes as u64;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(processes)
        .build()
        .unwrap();

    let start = Instant::now();
    pool.scope(|s| {
        for _ in 0..processes {
            s.spawn(move |_| {
                unsafe { sha_ext_kernel(iters_per_thread) };
            });
        }
    });
    let duration = start.elapsed();

    let total_rounds = iterations as f64 * 4.0;
    let mrounds = total_rounds / duration.as_secs_f64() / 1e6;

    BenchResult::new("SHA Extensions", duration, "Mrounds/s", mrounds)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sha")]
unsafe fn sha_ext_kernel(iterations: u64) {
    use std::arch::x86_64::*;
    let mut state0 = _mm_set_epi32(0x6a09e667_u32 as i32, 0xbb67ae85_u32 as i32, 0x3c6ef372_u32 as i32, 0xa54ff53a_u32 as i32);
    let mut state1 = _mm_set_epi32(0x510e527f_u32 as i32, 0x9b05688c_u32 as i32, 0x1f83d9ab_u32 as i32, 0x5be0cd19_u32 as i32);
    let msg = _mm_set_epi32(0x12345678_u32 as i32, 0x9abcdef0_u32 as i32, 0x0fedcba9_u32 as i32, 0x87654321_u32 as i32);

    for _ in 0..iterations {
        state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
        state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
        state0 = _mm_sha256rnds2_epu32(state0, state1, msg);
        state1 = _mm_sha256rnds2_epu32(state1, state0, msg);
    }

    black_box(state0);
    black_box(state1);
}

#[cfg(not(target_arch = "x86_64"))]
pub fn run_sha_ext(_processes: usize, _intensity: u32) -> BenchResult {
    BenchResult::not_supported("SHA Extensions")
}

// ── SSE4.2 CRC32 ───────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub fn run_crc32(processes: usize, intensity: u32) -> BenchResult {
    if !is_x86_feature_detected!("sse4.2") {
        return BenchResult::not_supported("SSE4.2 CRC32");
    }
    let base_iterations: u64 = 50_000_000;
    let iterations = base_iterations * (1u64 << intensity);
    let iters_per_thread = iterations / processes as u64;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(processes)
        .build()
        .unwrap();

    let start = Instant::now();
    pool.scope(|s| {
        for _ in 0..processes {
            s.spawn(move |_| {
                unsafe { crc32_kernel(iters_per_thread) };
            });
        }
    });
    let duration = start.elapsed();

    // 4 CRC ops * 8 bytes each per iteration
    let total_bytes = iterations as f64 * 4.0 * 8.0;
    let gbps = total_bytes / duration.as_secs_f64() / 1e9;

    BenchResult::new("SSE4.2 CRC32", duration, "GB/s", gbps)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32_kernel(iterations: u64) {
    use std::arch::x86_64::*;
    let mut crc0: u64 = 0xDEADBEEF;
    let mut crc1: u64 = 0xCAFEBABE;
    let mut crc2: u64 = 0x12345678;
    let mut crc3: u64 = 0x9ABCDEF0;
    let data: u64 = 0xFEDCBA9876543210;

    for _ in 0..iterations {
        crc0 = _mm_crc32_u64(crc0, data);
        crc1 = _mm_crc32_u64(crc1, data);
        crc2 = _mm_crc32_u64(crc2, data);
        crc3 = _mm_crc32_u64(crc3, data);
    }

    black_box(crc0);
    black_box(crc1);
    black_box(crc2);
    black_box(crc3);
}

#[cfg(not(target_arch = "x86_64"))]
pub fn run_crc32(_processes: usize, _intensity: u32) -> BenchResult {
    BenchResult::not_supported("SSE4.2 CRC32")
}

// ── AVX2 Integer SIMD ───────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub fn run_avx2_int(processes: usize, intensity: u32) -> BenchResult {
    if !is_x86_feature_detected!("avx2") {
        return BenchResult::not_supported("AVX2 Integer SIMD");
    }
    let base_iterations: u64 = 50_000_000;
    let iterations = base_iterations * (1u64 << intensity);
    let iters_per_thread = iterations / processes as u64;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(processes)
        .build()
        .unwrap();

    let start = Instant::now();
    pool.scope(|s| {
        for _ in 0..processes {
            s.spawn(move |_| {
                unsafe { avx2_int_kernel(iters_per_thread) };
            });
        }
    });
    let duration = start.elapsed();

    // 4 accumulators * 8 ints per register * 2 ops (add+mul) = 64 ops/iter
    let total_ops = iterations as f64 * 64.0;
    let gops = total_ops / duration.as_secs_f64() / 1e9;

    BenchResult::new("AVX2 Integer SIMD", duration, "GOPS", gops)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_int_kernel(iterations: u64) {
    use std::arch::x86_64::*;
    let mut acc0 = _mm256_set1_epi32(1);
    let mut acc1 = _mm256_set1_epi32(2);
    let mut acc2 = _mm256_set1_epi32(3);
    let mut acc3 = _mm256_set1_epi32(4);
    let addend = _mm256_set1_epi32(1);

    for _ in 0..iterations {
        acc0 = _mm256_add_epi32(acc0, addend);
        acc0 = _mm256_mullo_epi32(acc0, addend);
        acc1 = _mm256_add_epi32(acc1, addend);
        acc1 = _mm256_mullo_epi32(acc1, addend);
        acc2 = _mm256_add_epi32(acc2, addend);
        acc2 = _mm256_mullo_epi32(acc2, addend);
        acc3 = _mm256_add_epi32(acc3, addend);
        acc3 = _mm256_mullo_epi32(acc3, addend);
    }

    black_box(acc0);
    black_box(acc1);
    black_box(acc2);
    black_box(acc3);
}

#[cfg(not(target_arch = "x86_64"))]
pub fn run_avx2_int(_processes: usize, _intensity: u32) -> BenchResult {
    BenchResult::not_supported("AVX2 Integer SIMD")
}

// ── Runner ──────────────────────────────────────────────────────────────────

pub fn run_all_low_level(processes: usize, intensity: u32) -> Vec<BenchResult> {
    vec![
        run_avx2_fp(processes, intensity),
        run_avx512_fp(processes, intensity),
        run_aes_ni(processes, intensity),
        run_sha_ext(processes, intensity),
        run_crc32(processes, intensity),
        run_avx2_int(processes, intensity),
    ]
}
