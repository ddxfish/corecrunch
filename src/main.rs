use clap::Parser;
use std::error::Error;
use std::time::Instant;
use std::io::{self, Write};

mod bench_low;
mod bench_realworld;
//mod metrics;
mod report;
mod system_info;

use bench_low::run_all_low_level;
use bench_realworld::run_all_real_world;
use report::BenchmarkReport;
use system_info::SystemInfo;

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    /// Number of processes to use (default: number of CPU cores)
    #[arg(short = 'p', long)]
    processes: Option<usize>,

    /// Intensity level (1-10, default: 5)
    #[arg(short = 'i', long, default_value = "5")]
    intensity: u32,

    /// Run only low-level benchmarks
    #[arg(long)]
    low_level_only: bool,

    /// Run only real-world benchmarks
    #[arg(long)]
    real_world_only: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    
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
    
    println!("Starting benchmarks with intensity {}...", args.intensity);
    if args.intensity > 10 {
        println!("Warning: High intensity levels may result in long execution times!");
    }
    
    let total_start = Instant::now();
    
    let mut low_level_results = Vec::new();
    let mut real_world_results = Vec::new();
    
    for &processes in &process_counts {
        println!("\nRunning tests with {} process{}..", 
            processes,
            if processes > 1 { "es" } else { "" }
        );
        
        if !args.real_world_only {
            println!("Running low-level benchmarks...");
            low_level_results.push(run_all_low_level(processes, args.intensity));
        }
        
        if !args.low_level_only {
            println!("Running real-world benchmarks...");
            real_world_results.push(run_all_real_world(processes, args.intensity));
        }
    }
    
    let total_duration = total_start.elapsed();
    
    let report = BenchmarkReport::new(
        system_info.to_metrics(),
        low_level_results,
        real_world_results,
        process_counts,
    );
    
    println!("\n{}", report.generate_report());
    println!("\nTotal benchmark time: {:.1} seconds", total_duration.as_secs_f64());
    
    // Prevent immediate closure by waiting for user input
    print!("\nPress Enter to exit...");
    io::stdout().flush()?;  // Ensure the prompt is displayed immediately
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    
    Ok(())
}