use clap::Parser;
use std::error::Error;
use std::time::Instant;
use std::io::{self, Write};

mod bench_legacy;
mod bench_low;
mod bench_memory;
mod bench_realworld;
mod report;
mod system_info;

use bench_legacy::run_all_legacy;
use bench_low::run_all_low_level;
use bench_memory::run_all_memory;
use bench_realworld::run_all_real_world;
use report::BenchmarkReport;
use system_info::{SystemInfo, read_cpu_temp};

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    /// Number of processes to use (default: number of CPU cores)
    #[arg(short = 'p', long)]
    processes: Option<usize>,

    /// Intensity level (1-10, default: 3)
    #[arg(short = 'i', long, default_value = "3")]
    intensity: u32,

    /// Run only low-level benchmarks
    #[arg(long)]
    low_level_only: bool,

    /// Run only memory benchmarks
    #[arg(long)]
    memory_only: bool,

    /// Run only real-world benchmarks
    #[arg(long)]
    real_world_only: bool,

    /// Include legacy (v1) benchmarks
    #[arg(long)]
    legacy: bool,

    /// Run only legacy (v1) benchmarks
    #[arg(long)]
    legacy_only: bool,

    /// Disable CPU temperature monitoring
    #[arg(long)]
    no_temp: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    println!(r#"
┌───────────────────────────────────────────────────────┐
│  ____                ____                       _     │
│ / ___|___  _ __ ___ / ___|_ __ _   _ _ __   ___| |__  │
│| |   / _ \| '__/ _ \ |   | '__| | | | '_ \ / __| '_ \ │
│| |__| (_) | | |  __/ |___| |  | |_| | | | | (__| | | |│
│ \____\___/|_|  \___|\____|_|   \__,_|_| |_|\___|_| |_|│
└───────────────────────────────────────────────────────┘
                                   by ddxfish
"#);

    let system_info = SystemInfo::collect()?;
    println!("{}", system_info.format_summary());

    let max_processes = system_info.cpu_cores;
    let process_counts = match args.processes {
        Some(user_specified) => {
            if user_specified == 1 {
                vec![1]
            } else {
                vec![1, user_specified, max_processes]
            }
        }
        None => {
            if max_processes == 1 {
                vec![1]
            } else {
                vec![1, max_processes / 2, max_processes]
            }
        }
    };

    // Determine which tiers to run
    // --legacy-only is an exclusive *_only flag; --legacy is additive (runs on top of whatever else)
    let any_only = args.low_level_only || args.memory_only || args.real_world_only || args.legacy_only;
    let run_low = !any_only || args.low_level_only;
    let run_mem = !any_only || args.memory_only;
    let run_rw = !any_only || args.real_world_only;
    let run_legacy = args.legacy || args.legacy_only;

    println!("Starting benchmarks with intensity {}...", args.intensity);
    if args.intensity > 10 {
        println!("Warning: High intensity levels may result in long execution times!");
    }

    let total_start = Instant::now();

    let mut low_level_results = Vec::new();
    let mut memory_results = Vec::new();
    let mut real_world_results = Vec::new();
    let mut legacy_results = Vec::new();
    let mut temp_readings: Vec<Option<(f32, f32)>> = Vec::new();

    for &processes in &process_counts {
        println!("\nRunning tests with {} process{}..",
            processes,
            if processes > 1 { "es" } else { "" }
        );

        let temp_before = if !args.no_temp { read_cpu_temp() } else { None };

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(processes)
            .build()
            .unwrap();

        if run_low {
            println!("  CPU feature benchmarks...");
            low_level_results.push(pool.install(|| run_all_low_level(processes, args.intensity)));
        }

        if run_mem {
            println!("  Memory benchmarks...");
            memory_results.push(pool.install(|| run_all_memory(processes, args.intensity)));
        }

        if run_rw {
            println!("  Real-world benchmarks...");
            real_world_results.push(pool.install(|| run_all_real_world(processes, args.intensity)));
        }

        if run_legacy {
            println!("  Legacy benchmarks...");
            legacy_results.push(pool.install(|| run_all_legacy(processes, args.intensity)));
        }

        if !args.no_temp {
            let temp_after = read_cpu_temp();
            temp_readings.push(match (temp_before, temp_after) {
                (Some(b), Some(a)) => Some((b, a)),
                _ => None,
            });
        }
    }

    let total_duration = total_start.elapsed();

    let temp_data = if !args.no_temp { Some(temp_readings) } else { None };

    let report = BenchmarkReport::new(
        system_info.to_metrics(),
        low_level_results,
        memory_results,
        real_world_results,
        legacy_results,
        process_counts,
        temp_data,
        total_duration,
    );

    println!("\n{}", report.generate_report());

    // Prevent immediate closure by waiting for user input
    print!("\nPress Enter to exit...");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;

    Ok(())
}
