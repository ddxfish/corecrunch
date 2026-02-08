use prettytable::{Table, Row, Cell, format};
use crate::system_info::SystemMetrics;
use crate::bench_low::BenchResult;
use std::time::Duration;

// Baseline values for scoring (calibrated so a mid-range modern CPU ≈ 1000)
// CPU benchmarks — single-core baselines
const BASE_AVX2_FP: f64 = 15.0;       // GFLOPS
const BASE_AES_NI: f64 = 70.0;        // GB/s
const BASE_SHA_EXT: f64 = 900.0;      // Mrounds/s
const BASE_CRC32: f64 = 28.0;         // GB/s
const BASE_AVX2_INT: f64 = 350.0;     // GOPS
// Real-world baselines
const BASE_LLM: f64 = 2.0;           // GFLOPS
const BASE_GZIP: f64 = 100.0;         // MB/s
const BASE_BLUR: f64 = 15.0;          // Mpix/s
const BASE_SORT: f64 = 12.0;          // Mrec/s
// Memory baselines
const BASE_SEQ_READ: f64 = 7.0;       // GB/s
const BASE_SEQ_WRITE: f64 = 3.5;      // GB/s
const BASE_MEM_LATENCY: f64 = 120.0;  // ns (inverted: lower = better)

pub struct BenchmarkReport {
    system_metrics: SystemMetrics,
    low_level_results: Vec<Vec<BenchResult>>,
    memory_results: Vec<Vec<BenchResult>>,
    real_world_results: Vec<Vec<BenchResult>>,
    legacy_results: Vec<Vec<BenchResult>>,
    core_counts: Vec<usize>,
    temp_data: Option<Vec<Option<(f32, f32)>>>,
    total_duration: Duration,
}

impl BenchmarkReport {
    pub fn new(
        system_metrics: SystemMetrics,
        low_level: Vec<Vec<BenchResult>>,
        memory: Vec<Vec<BenchResult>>,
        real_world: Vec<Vec<BenchResult>>,
        legacy: Vec<Vec<BenchResult>>,
        cores: Vec<usize>,
        temp_data: Option<Vec<Option<(f32, f32)>>>,
        total_duration: Duration,
    ) -> Self {
        Self {
            system_metrics,
            low_level_results: low_level,
            memory_results: memory,
            real_world_results: real_world,
            legacy_results: legacy,
            core_counts: cores,
            temp_data,
            total_duration,
        }
    }

    fn format_duration(d: Duration) -> String {
        if d.as_secs() > 0 {
            format!("{:.1}s", d.as_secs_f64())
        } else {
            format!("{}ms", d.as_millis())
        }
    }

    fn format_metric(value: f64) -> String {
        if value >= 1_000_000.0 {
            format!("{:.1}M", value / 1_000_000.0)
        } else if value >= 1000.0 {
            format!("{:.1}K", value / 1000.0)
        } else {
            format!("{:.2}", value)
        }
    }

    fn format_score(score: f64) -> String {
        if score >= 1000.0 {
            format!("{:.0}", score)
        } else {
            format!("{:.0}", score)
        }
    }

    /// Get the baseline for a given test name. Returns (baseline, inverted).
    /// inverted=true means lower metric_value is better (e.g. latency in ns).
    fn baseline_for(name: &str) -> Option<(f64, bool)> {
        match name {
            "AVX2 FP Throughput" => Some((BASE_AVX2_FP, false)),
            "AES-NI Throughput" => Some((BASE_AES_NI, false)),
            "SHA Extensions" => Some((BASE_SHA_EXT, false)),
            "SSE4.2 CRC32" => Some((BASE_CRC32, false)),
            "AVX2 Integer SIMD" => Some((BASE_AVX2_INT, false)),
            "LLM Inference Sim" => Some((BASE_LLM, false)),
            "Gzip Compress" => Some((BASE_GZIP, false)),
            "Image Blur (5x5)" => Some((BASE_BLUR, false)),
            "Dataset Sort" => Some((BASE_SORT, false)),
            "Seq Read Bandwidth" => Some((BASE_SEQ_READ, false)),
            "Seq Write Bandwidth" => Some((BASE_SEQ_WRITE, false)),
            "Memory Latency" => Some((BASE_MEM_LATENCY, true)),
            _ => None,
        }
    }

    /// Compute geometric mean score from a set of results at a given core-count index.
    fn compute_score(results_sets: &[&Vec<Vec<BenchResult>>], idx: usize) -> Option<f64> {
        let mut log_sum = 0.0;
        let mut count = 0u32;

        for results in results_sets {
            if results.is_empty() || idx >= results.len() {
                continue;
            }
            for bench in &results[idx] {
                if !bench.supported || bench.metric_value <= 0.0 {
                    continue;
                }
                if let Some((baseline, inverted)) = Self::baseline_for(&bench.name) {
                    let ratio = if inverted {
                        baseline / bench.metric_value
                    } else {
                        bench.metric_value / baseline
                    };
                    log_sum += ratio.ln();
                    count += 1;
                }
            }
        }

        if count == 0 {
            None
        } else {
            Some((log_sum / count as f64).exp() * 1000.0)
        }
    }

    pub fn generate_report(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "System: {} @ {:.1} GHz\n",
            self.system_metrics.cpu_name,
            self.system_metrics.cpu_freq as f64 / 1000.0
        ));
        output.push_str(&format!(
            "Cores: {} | Memory: {:.1} GB\n",
            self.system_metrics.core_count,
            self.system_metrics.memory_total as f64 / 1024.0 / 1024.0 / 1024.0
        ));
        output.push_str(&format!(
            "Features: {}\n\n",
            self.system_metrics.cpu_features.format_line()
        ));

        let mut table = Table::new();
        let format = format::FormatBuilder::new()
            .column_separator('│')
            .borders('│')
            .separators(
                &[format::LinePosition::Top],
                format::LineSeparator::new('─', '┬', '┌', '┐')
            )
            .separators(
                &[format::LinePosition::Bottom],
                format::LineSeparator::new('─', '┴', '└', '┘')
            )
            .padding(1, 1)
            .build();
        table.set_format(format);

        let mut header = vec![Cell::new("Test").style_spec("b")];
        for &cores in &self.core_counts {
            header.push(Cell::new(&format!("{} Core{}",
                cores,
                if cores > 1 { "s" } else { "" }
            )).style_spec("b"));
        }
        table.add_row(Row::new(header));

        let separator = " ";

        // CPU Feature Tests section
        if !self.low_level_results.is_empty() {
            table.add_empty_row();
            table.add_row(Row::new(vec![Cell::new("===CPU Feature Tests===").style_spec("bi").with_hspan(self.core_counts.len() + 1)]));
            table.add_row(Row::new(vec![Cell::new(separator).with_hspan(self.core_counts.len() + 1)]));
            self.add_benchmark_section(&mut table, &self.low_level_results);
        }

        // Memory Tests section
        if !self.memory_results.is_empty() {
            table.add_empty_row();
            table.add_row(Row::new(vec![Cell::new("===Memory Tests===").style_spec("bi").with_hspan(self.core_counts.len() + 1)]));
            table.add_row(Row::new(vec![Cell::new(separator).with_hspan(self.core_counts.len() + 1)]));
            self.add_benchmark_section(&mut table, &self.memory_results);
        }

        // Real World Tests section
        if !self.real_world_results.is_empty() {
            table.add_empty_row();
            table.add_row(Row::new(vec![Cell::new("===Real World Tests===").style_spec("bi").with_hspan(self.core_counts.len() + 1)]));
            table.add_row(Row::new(vec![Cell::new(separator).with_hspan(self.core_counts.len() + 1)]));
            self.add_benchmark_section(&mut table, &self.real_world_results);
        }

        // Legacy Tests section
        if !self.legacy_results.is_empty() {
            table.add_empty_row();
            table.add_row(Row::new(vec![Cell::new("===Legacy Tests===").style_spec("bi").with_hspan(self.core_counts.len() + 1)]));
            table.add_row(Row::new(vec![Cell::new(separator).with_hspan(self.core_counts.len() + 1)]));
            self.add_benchmark_section(&mut table, &self.legacy_results);
        }

        // Scaling section
        if self.core_counts.len() > 1 {
            table.add_empty_row();
            table.add_row(Row::new(vec![Cell::new("===Scaling===").style_spec("bi").with_hspan(self.core_counts.len() + 1)]));
            table.add_row(Row::new(vec![Cell::new(separator).with_hspan(self.core_counts.len() + 1)]));
            self.add_scaling_metrics(&mut table);
        }

        // CPU Temperature section — only show if we got at least one reading
        if let Some(ref temps) = self.temp_data {
            if temps.iter().any(|t| t.is_some()) {
                table.add_empty_row();
                let mut row = vec![Cell::new("CPU Temp")];
                for temp in temps {
                    match temp {
                        Some((before, after)) => {
                            row.push(Cell::new(&format!("{:.0}\u{00b0}C \u{2192} {:.0}\u{00b0}C", before, after)));
                        }
                        None => {
                            row.push(Cell::new("N/A"));
                        }
                    }
                }
                table.add_row(Row::new(row));
            }
        }

        output.push_str(&table.to_string());

        // Scores section
        output.push_str(&self.generate_scores());

        output
    }

    fn generate_scores(&self) -> String {
        let mut lines = Vec::new();
        let w = 54; // inner width of the box

        // CPU tests = low-level + real-world
        let cpu_refs: Vec<&Vec<Vec<BenchResult>>> = vec![&self.low_level_results, &self.real_world_results];
        let mem_refs: Vec<&Vec<Vec<BenchResult>>> = vec![&self.memory_results];

        let last = self.core_counts.len().saturating_sub(1);

        let cpu_single = Self::compute_score(&cpu_refs, 0);
        let cpu_multi = if last > 0 { Self::compute_score(&cpu_refs, last) } else { None };
        let mem_score = Self::compute_score(&mem_refs, 0);

        // Overall = geometric mean of available scores
        let mut all_scores = Vec::new();
        if let Some(s) = cpu_single { all_scores.push(s); }
        if let Some(s) = cpu_multi { all_scores.push(s); }
        if let Some(s) = mem_score { all_scores.push(s); }

        let overall = if !all_scores.is_empty() {
            let log_sum: f64 = all_scores.iter().map(|s| s.ln()).sum();
            Some((log_sum / all_scores.len() as f64).exp())
        } else {
            None
        };

        if cpu_single.is_none() && cpu_multi.is_none() && mem_score.is_none() {
            return String::new();
        }

        let row = |s: &str| format!("║{:<w$}║", s, w = w);
        let sep: String = "═".repeat(w);
        let mid: String = "─".repeat(w);

        lines.push(String::new());
        lines.push(format!("╔{}╗", sep));
        lines.push(row("  CORECRUNCH SCORES"));

        // CPU section
        if cpu_single.is_some() || cpu_multi.is_some() {
            lines.push(format!("╠{}╣", mid));
            lines.push(row(&format!("  {}", self.system_metrics.cpu_name)));
            if let Some(s) = cpu_single {
                let ms = cpu_multi.map_or("  N/A".to_string(), |m| format!("{:>5}", Self::format_score(m)));
                lines.push(row(&format!("  Single-Core: {:>5}         Multi-Core: {}", Self::format_score(s), ms)));
            }
        }

        // Memory section
        if let Some(s) = mem_score {
            lines.push(format!("╠{}╣", mid));
            let mem_gb = self.system_metrics.memory_total as f64 / 1024.0 / 1024.0 / 1024.0;
            lines.push(row(&format!("  {:.0} GB RAM", mem_gb)));
            lines.push(row(&format!("  Memory Score: {:>5}", Self::format_score(s))));
        }

        // Overall + time section
        lines.push(format!("╠{}╣", mid));
        if let Some(g) = overall {
            lines.push(row(&format!("  ★ Overall:    {:>5}", Self::format_score(g))));
        }
        let time_str = format!("{:.1}s", self.total_duration.as_secs_f64());
        lines.push(row(&format!("  Time:         {:>5}", time_str)));

        lines.push(format!("╚{}╝", sep));

        lines.join("\n")
    }

    fn add_benchmark_section(&self, table: &mut Table, results: &[Vec<BenchResult>]) {
        if results.is_empty() || results[0].is_empty() {
            return;
        }

        let test_names: Vec<_> = results[0].iter().map(|r| r.name.clone()).collect();

        for name in test_names {
            let mut row = vec![Cell::new(&name)];

            for core_results in results {
                if let Some(result) = core_results.iter().find(|r| r.name == name) {
                    if !result.supported {
                        row.push(Cell::new("Not Supported"));
                    } else {
                        let time_str = Self::format_duration(result.duration);
                        let metric_str = format!("{} {}",
                            Self::format_metric(result.metric_value),
                            result.metric_name
                        );
                        row.push(Cell::new(&format!("{}\n{}", time_str, metric_str)));
                    }
                }
            }

            table.add_row(Row::new(row));
        }
    }

    fn add_scaling_metrics(&self, table: &mut Table) {
        let all_results = [
            &self.low_level_results,
            &self.memory_results,
            &self.real_world_results,
            &self.legacy_results,
        ];

        for (i, &core_count) in self.core_counts.iter().enumerate().skip(1) {
            let mut total_efficiency = 0.0;
            let mut count = 0;

            for results in &all_results {
                if !results.is_empty() && i < results.len() {
                    for j in 0..results[0].len() {
                        // Skip unsupported tests
                        if !results[0][j].supported || !results[i][j].supported {
                            continue;
                        }
                        let baseline = results[0][j].duration;
                        let actual = results[i][j].duration;
                        if baseline.is_zero() || actual.is_zero() {
                            continue;
                        }
                        let actual_speedup = baseline.as_secs_f64() / actual.as_secs_f64();
                        let efficiency = (actual_speedup / core_count as f64) * 100.0;
                        total_efficiency += efficiency;
                        count += 1;
                    }
                }
            }

            if count > 0 {
                let avg_efficiency = total_efficiency / count as f64;
                let speedup = core_count as f64 * avg_efficiency / 100.0;
                let mut row = vec![Cell::new(&format!("{} Cores", core_count))];
                row.push(Cell::new(&format!("{:.1}x ({:.0}% eff)", speedup, avg_efficiency)));
                table.add_row(Row::new(row));
            }
        }
    }
}
