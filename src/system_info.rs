use std::error::Error;
use sysinfo::{CpuRefreshKind, System, RefreshKind, MemoryRefreshKind};
#[cfg(target_os = "linux")]
use sysinfo::Components;

#[derive(Debug, Clone)]
pub struct CpuFeatures {
    pub avx2: bool,
    pub avx512f: bool,
    pub aes: bool,
    pub sha: bool,
    pub sse4_2: bool,
    pub fma: bool,
    pub bmi2: bool,
    pub popcnt: bool,
}

impl CpuFeatures {
    #[cfg(target_arch = "x86_64")]
    pub fn detect() -> Self {
        Self {
            avx2: is_x86_feature_detected!("avx2"),
            avx512f: is_x86_feature_detected!("avx512f"),
            aes: is_x86_feature_detected!("aes"),
            sha: is_x86_feature_detected!("sha"),
            sse4_2: is_x86_feature_detected!("sse4.2"),
            fma: is_x86_feature_detected!("fma"),
            bmi2: is_x86_feature_detected!("bmi2"),
            popcnt: is_x86_feature_detected!("popcnt"),
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn detect() -> Self {
        Self {
            avx2: false,
            avx512f: false,
            aes: false,
            sha: false,
            sse4_2: false,
            fma: false,
            bmi2: false,
            popcnt: false,
        }
    }

    pub fn format_line(&self) -> String {
        let yn = |b: bool| if b { "Y" } else { "N" };
        format!(
            "AVX2: {} | AVX-512: {} | AES-NI: {} | SHA: {} | SSE4.2: {} | FMA: {} | BMI2: {} | POPCNT: {}",
            yn(self.avx2), yn(self.avx512f), yn(self.aes), yn(self.sha),
            yn(self.sse4_2), yn(self.fma), yn(self.bmi2), yn(self.popcnt)
        )
    }
}

#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub cpu_brand: String,
    pub cpu_cores: usize,
    pub cpu_freq: u64,
    pub total_memory: u64,
    pub os_name: String,
    pub os_version: String,
    pub cpu_features: CpuFeatures,
}

#[derive(Debug, Clone)]
pub struct SystemMetrics {
    pub cpu_name: String,
    pub cpu_freq: u64,
    pub core_count: usize,
    pub memory_total: u64,
    pub cpu_features: CpuFeatures,
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
            cpu_features: CpuFeatures::detect(),
        })
    }

    pub fn format_summary(&self) -> String {
        format!(
            "System Information:\n\
             CPU: {} ({} cores) @ {:.2} GHz\n\
             Memory: {:.1} GB\n\
             OS: {} {}\n\
             Features: {}\n",
            self.cpu_brand,
            self.cpu_cores,
            self.cpu_freq as f64 / 1000.0,
            self.total_memory as f64 / 1024.0 / 1024.0 / 1024.0,
            self.os_name,
            self.os_version,
            self.cpu_features.format_line()
        )
    }

    pub fn to_metrics(&self) -> SystemMetrics {
        SystemMetrics {
            cpu_name: self.cpu_brand.clone(),
            cpu_freq: self.cpu_freq,
            core_count: self.cpu_cores,
            memory_total: self.total_memory,
            cpu_features: self.cpu_features.clone(),
        }
    }
}

pub fn read_cpu_temp() -> Option<f32> {
    #[cfg(target_os = "linux")]
    {
        std::panic::catch_unwind(|| {
            let components = Components::new_with_refreshed_list();
            let mut max_temp: Option<f32> = None;

            for component in &components {
                let label = component.label().to_lowercase();
                if label.contains("core") || label.contains("package") || label.contains("cpu") || label.contains("tctl") {
                    let temp = component.temperature();
                    if temp > 0.0 && temp < 150.0 {
                        max_temp = Some(max_temp.map_or(temp, |m: f32| m.max(temp)));
                    }
                }
            }

            max_temp
        }).ok().flatten()
    }

    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}
