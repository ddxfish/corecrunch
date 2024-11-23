use prettytable::{Table, Row, Cell, format};
use crate::system_info::SystemMetrics;
use crate::bench_low::BenchResult;
use std::time::Duration;

pub struct BenchmarkReport {
    system_metrics: SystemMetrics,
    low_level_results: Vec<Vec<BenchResult>>,
    real_world_results: Vec<Vec<BenchResult>>,
    core_counts: Vec<usize>,
}

impl BenchmarkReport {
    pub fn new(
        system_metrics: SystemMetrics,
        low_level: Vec<Vec<BenchResult>>,
        real_world: Vec<Vec<BenchResult>>,
        cores: Vec<usize>,
    ) -> Self {
        Self {
            system_metrics,
            low_level_results: low_level,
            real_world_results: real_world,
            core_counts: cores,
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
            format!("{:.1}", value)
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
            "Cores: {} | Memory: {:.1} GB\n\n",
            self.system_metrics.core_count,
            self.system_metrics.memory_total as f64 / 1024.0 / 1024.0 / 1024.0
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
        
        // Create a simple 10-dash separator
        let separator = " ";
        
        // Low Level Tests section
        table.add_empty_row();
        table.add_row(Row::new(vec![Cell::new("===Low Level Tests===").style_spec("bi").with_hspan(self.core_counts.len() + 1)]));
        table.add_row(Row::new(vec![Cell::new(separator).with_hspan(self.core_counts.len() + 1)]));
        self.add_benchmark_section(&mut table, &self.low_level_results);
        
        // Real World Tests section
        table.add_empty_row();
        table.add_row(Row::new(vec![Cell::new("===Real World Tests===").style_spec("bi").with_hspan(self.core_counts.len() + 1)]));
        table.add_row(Row::new(vec![Cell::new(separator).with_hspan(self.core_counts.len() + 1)]));
        self.add_benchmark_section(&mut table, &self.real_world_results);
        
        // Scaling section
        table.add_empty_row();
        table.add_row(Row::new(vec![Cell::new("===Scaling===").style_spec("bi").with_hspan(self.core_counts.len() + 1)]));
        table.add_row(Row::new(vec![Cell::new(separator).with_hspan(self.core_counts.len() + 1)]));
        self.add_scaling_metrics(&mut table);
        
        output.push_str(&table.to_string());
        output
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
                    let time_str = Self::format_duration(result.duration);
                    let metric_str = format!("{} {}", 
                        Self::format_metric(result.metric_value),
                        result.metric_name
                    );
                    row.push(Cell::new(&format!("{}\n{}", time_str, metric_str)));
                }
            }
            
            table.add_row(Row::new(row));
        }
    }
    
    fn add_scaling_metrics(&self, table: &mut Table) {
        for (i, &core_count) in self.core_counts.iter().enumerate().skip(1) {
            let mut total_efficiency = 0.0;
            let mut count = 0;
            
            for results in [&self.low_level_results, &self.real_world_results].iter() {
                if !results.is_empty() && i < results.len() {
                    for j in 0..results[0].len() {
                        let baseline = results[0][j].duration;
                        let actual = results[i][j].duration;
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
                row.push(Cell::new(&format!("{:.1}x", speedup)));
                table.add_row(Row::new(row));
            }
        }
    }
}