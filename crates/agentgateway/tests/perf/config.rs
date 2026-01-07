//! Performance test configuration
//!
//! Configuration is loaded from environment variables and/or a config file,
//! allowing flexible deployment in Docker containers with varying hardware.

use std::time::Duration;

/// Configuration for performance tests
#[derive(Debug, Clone)]
pub struct PerfConfig {
    /// Whether running in verification mode (minimal iterations to check test validity)
    pub verify_mode: bool,

    /// Number of warmup iterations before measuring
    pub warmup_iterations: usize,

    /// Number of measurement iterations for latency tests
    pub measurement_iterations: usize,

    /// Number of concurrent clients for load tests
    pub concurrent_clients: usize,

    /// Duration for sustained load tests
    pub load_duration: Duration,

    /// Duration for stability/soak tests
    pub stability_duration: Duration,

    /// Target requests per second for load tests (0 = unlimited)
    pub target_rps: u64,

    /// Small payload size for baseline tests (bytes)
    pub small_payload_size: usize,

    /// Medium payload size (bytes)
    pub medium_payload_size: usize,

    /// Large payload size for stress tests (bytes)
    pub large_payload_size: usize,

    /// Extra large payload (long context) size (bytes)
    pub xl_payload_size: usize,

    /// Number of backends for multi-backend tests
    pub num_backends: usize,

    /// Backend failure simulation duration
    pub backend_failure_duration: Duration,

    /// CPU monitoring sample interval
    pub cpu_sample_interval: Duration,

    /// Memory check interval for stability tests
    pub memory_check_interval: Duration,

    /// Maximum acceptable memory growth ratio for stability tests
    pub max_memory_growth_ratio: f64,

    /// Timeout for individual requests
    pub request_timeout: Duration,

    /// Port range start for test servers
    pub port_range_start: u16,

    /// Whether to output detailed results as JSON
    pub json_output: bool,

    /// Output file path for results
    pub output_path: Option<String>,
}

impl Default for PerfConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl PerfConfig {
    /// Load configuration from environment variables with sensible defaults
    pub fn from_env() -> Self {
        let verify_mode = std::env::var("PERF_VERIFY")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);

        // In verify mode, use minimal values to just check test validity
        if verify_mode {
            return Self::verification_config();
        }

        Self {
            verify_mode: false,
            warmup_iterations: env_usize("PERF_WARMUP", 100),
            measurement_iterations: env_usize("PERF_ITERATIONS", 1000),
            concurrent_clients: env_usize("PERF_CLIENTS", 10),
            load_duration: Duration::from_secs(env_u64("PERF_LOAD_DURATION_SECS", 30)),
            stability_duration: Duration::from_secs(env_u64("PERF_STABILITY_DURATION_SECS", 300)),
            target_rps: env_u64("PERF_TARGET_RPS", 0),
            small_payload_size: env_usize("PERF_SMALL_PAYLOAD", 256),
            medium_payload_size: env_usize("PERF_MEDIUM_PAYLOAD", 4096),
            large_payload_size: env_usize("PERF_LARGE_PAYLOAD", 65536),
            xl_payload_size: env_usize("PERF_XL_PAYLOAD", 1_048_576), // 1MB
            num_backends: env_usize("PERF_NUM_BACKENDS", 3),
            backend_failure_duration: Duration::from_secs(env_u64("PERF_BACKEND_FAILURE_SECS", 5)),
            cpu_sample_interval: Duration::from_millis(env_u64("PERF_CPU_SAMPLE_MS", 100)),
            memory_check_interval: Duration::from_secs(env_u64("PERF_MEMORY_CHECK_SECS", 10)),
            max_memory_growth_ratio: env_f64("PERF_MAX_MEMORY_GROWTH", 1.5),
            request_timeout: Duration::from_secs(env_u64("PERF_REQUEST_TIMEOUT_SECS", 30)),
            port_range_start: env_u16("PERF_PORT_START", 19000),
            json_output: std::env::var("PERF_JSON_OUTPUT")
                .map(|v| v == "1" || v.to_lowercase() == "true")
                .unwrap_or(false),
            output_path: std::env::var("PERF_OUTPUT_PATH").ok(),
        }
    }

    /// Configuration for verification mode - minimal iterations to verify tests work
    pub fn verification_config() -> Self {
        Self {
            verify_mode: true,
            warmup_iterations: 2,
            measurement_iterations: 5,
            concurrent_clients: 2,
            load_duration: Duration::from_millis(500),
            stability_duration: Duration::from_secs(2),
            target_rps: 0,
            small_payload_size: 64,
            medium_payload_size: 256,
            large_payload_size: 1024,
            xl_payload_size: 4096,
            num_backends: 2,
            backend_failure_duration: Duration::from_millis(500),
            cpu_sample_interval: Duration::from_millis(50),
            memory_check_interval: Duration::from_millis(500),
            max_memory_growth_ratio: 2.0,
            request_timeout: Duration::from_secs(5),
            port_range_start: 19000,
            json_output: false,
            output_path: None,
        }
    }

    /// Configuration for CI environments - balanced between speed and coverage
    pub fn ci_config() -> Self {
        Self {
            verify_mode: false,
            warmup_iterations: 20,
            measurement_iterations: 100,
            concurrent_clients: 5,
            load_duration: Duration::from_secs(10),
            stability_duration: Duration::from_secs(30),
            target_rps: 0,
            small_payload_size: 256,
            medium_payload_size: 4096,
            large_payload_size: 32768,
            xl_payload_size: 262144, // 256KB
            num_backends: 2,
            backend_failure_duration: Duration::from_secs(2),
            cpu_sample_interval: Duration::from_millis(100),
            memory_check_interval: Duration::from_secs(5),
            max_memory_growth_ratio: 1.5,
            request_timeout: Duration::from_secs(10),
            port_range_start: 19000,
            json_output: true,
            output_path: Some("perf-results.json".to_string()),
        }
    }

    /// Configuration for full production-like testing
    pub fn production_config() -> Self {
        Self {
            verify_mode: false,
            warmup_iterations: 500,
            measurement_iterations: 5000,
            concurrent_clients: 50,
            load_duration: Duration::from_secs(120),
            stability_duration: Duration::from_secs(3600), // 1 hour
            target_rps: 0,
            small_payload_size: 256,
            medium_payload_size: 8192,
            large_payload_size: 131072, // 128KB
            xl_payload_size: 4_194_304, // 4MB
            num_backends: 5,
            backend_failure_duration: Duration::from_secs(10),
            cpu_sample_interval: Duration::from_millis(100),
            memory_check_interval: Duration::from_secs(30),
            max_memory_growth_ratio: 1.2,
            request_timeout: Duration::from_secs(60),
            port_range_start: 19000,
            json_output: true,
            output_path: Some("perf-results-production.json".to_string()),
        }
    }

    /// Get effective iteration count (respects verify mode)
    pub fn effective_iterations(&self) -> usize {
        if self.verify_mode {
            self.measurement_iterations.min(5)
        } else {
            self.measurement_iterations
        }
    }

    /// Get effective warmup count
    pub fn effective_warmup(&self) -> usize {
        if self.verify_mode {
            self.warmup_iterations.min(2)
        } else {
            self.warmup_iterations
        }
    }

    /// Get effective concurrent clients
    pub fn effective_clients(&self) -> usize {
        if self.verify_mode {
            self.concurrent_clients.min(2)
        } else {
            self.concurrent_clients
        }
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_u16(key: &str, default: u16) -> u16 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Hardware configuration for Docker-based testing
#[derive(Debug, Clone)]
pub struct HardwareConfig {
    /// Number of CPUs available
    pub cpus: f64,
    /// Memory limit in bytes
    pub memory_bytes: u64,
    /// Whether running in container
    pub containerized: bool,
}

impl HardwareConfig {
    /// Detect hardware configuration from environment
    pub fn detect() -> Self {
        let cpus = std::env::var("PERF_CPUS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| num_cpus::get() as f64);

        let memory_bytes = std::env::var("PERF_MEMORY_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(Self::detect_system_memory);

        let containerized = std::env::var("PERF_CONTAINERIZED")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or_else(|_| std::path::Path::new("/.dockerenv").exists());

        Self {
            cpus,
            memory_bytes,
            containerized,
        }
    }

    fn detect_system_memory() -> u64 {
        // Try to read from /proc/meminfo on Linux
        #[cfg(target_os = "linux")]
        {
            if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
                for line in contents.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return kb * 1024;
                            }
                        }
                    }
                }
            }
        }
        // Default to 8GB if detection fails
        8 * 1024 * 1024 * 1024
    }

    /// Get recommended concurrent clients based on hardware
    pub fn recommended_clients(&self) -> usize {
        // Rough heuristic: 2-4 clients per CPU core
        (self.cpus * 3.0).ceil() as usize
    }

    /// Get recommended memory for test payloads
    pub fn max_payload_size(&self) -> usize {
        // Don't use more than 10% of available memory for a single payload
        ((self.memory_bytes as f64 * 0.1) as usize).min(100 * 1024 * 1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PerfConfig::default();
        assert!(config.measurement_iterations > 0);
        assert!(config.warmup_iterations > 0);
    }

    #[test]
    fn test_verification_config() {
        let config = PerfConfig::verification_config();
        assert!(config.verify_mode);
        assert!(config.measurement_iterations <= 5);
        assert!(config.warmup_iterations <= 2);
    }

    #[test]
    fn test_hardware_detection() {
        let hw = HardwareConfig::detect();
        assert!(hw.cpus > 0.0);
        assert!(hw.memory_bytes > 0);
    }
}
