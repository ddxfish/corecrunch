use std::error::Error;
use sysinfo::{CpuRefreshKind, System, RefreshKind, MemoryRefreshKind};

#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub cpu_brand: String,
    pub cpu_cores: usize,
    pub cpu_freq: u64,
    pub total_memory: u64,
    pub os_name: String,
    pub os_version: String,
}

#[derive(Debug, Clone)]
pub struct SystemMetrics {
    pub cpu_name: String,
    pub cpu_freq: u64,
    pub core_count: usize,
    pub memory_total: u64,
}

impl SystemInfo {
    pub fn collect() -> Result<Self, Box<dyn Error>> {
        let mut sys = System::new_with_specifics(
            RefreshKind::new()
                .with_cpu(CpuRefreshKind::everything())
                .with_memory(MemoryRefreshKind::everything())
        );
        sys.refresh_all();

        let cpu = sys.cpus().first().ok_or("No CPU found")?;

        Ok(Self {
            cpu_brand: cpu.brand().trim().to_string(),
            cpu_cores: sys.cpus().len(),
            cpu_freq: cpu.frequency(),
            total_memory: sys.total_memory(),
            os_name: System::name().unwrap_or_else(|| "Unknown".to_string()),
            os_version: System::os_version().unwrap_or_else(|| "Unknown".to_string()),
        })
    }

    pub fn format_summary(&self) -> String {
        format!(
            "System Information:\n\
             CPU: {} ({} cores) @ {:.2} GHz\n\
             Memory: {:.1} GB\n\
             OS: {} {}\n",
            self.cpu_brand,
            self.cpu_cores,
            self.cpu_freq as f64 / 1000.0,
            self.total_memory as f64 / 1024.0 / 1024.0 / 1024.0,
            self.os_name,
            self.os_version
        )
    }

    pub fn to_metrics(&self) -> SystemMetrics {
        SystemMetrics {
            cpu_name: self.cpu_brand.clone(),
            cpu_freq: self.cpu_freq,
            core_count: self.cpu_cores,
            memory_total: self.total_memory,
        }
    }
}