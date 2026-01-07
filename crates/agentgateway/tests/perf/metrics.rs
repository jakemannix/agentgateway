//! Performance metrics collection and reporting
//!
//! Provides histograms for latency measurement, CPU monitoring,
//! and unified reporting of performance results.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

/// Latency histogram for efficient percentile calculation
/// Uses a fixed-bucket approach with microsecond precision
#[derive(Debug)]
pub struct LatencyHistogram {
    /// Bucket counts (index = microseconds / bucket_size)
    buckets: Vec<AtomicU64>,
    /// Size of each bucket in microseconds
    bucket_size_us: u64,
    /// Total samples
    count: AtomicU64,
    /// Sum for mean calculation (in microseconds)
    sum_us: AtomicU64,
    /// Minimum value seen
    min_us: AtomicU64,
    /// Maximum value seen
    max_us: AtomicU64,
}

impl LatencyHistogram {
    /// Create a new histogram with default settings (1ms buckets, 60s max)
    pub fn new() -> Self {
        Self::with_params(1000, 60_000_000) // 1ms buckets, up to 60s
    }

    /// Create histogram with custom bucket size and maximum value
    pub fn with_params(bucket_size_us: u64, max_us: u64) -> Self {
        let num_buckets = ((max_us / bucket_size_us) + 1) as usize;
        let mut buckets = Vec::with_capacity(num_buckets);
        for _ in 0..num_buckets {
            buckets.push(AtomicU64::new(0));
        }
        Self {
            buckets,
            bucket_size_us,
            count: AtomicU64::new(0),
            sum_us: AtomicU64::new(0),
            min_us: AtomicU64::new(u64::MAX),
            max_us: AtomicU64::new(0),
        }
    }

    /// Record a latency value
    pub fn record(&self, duration: Duration) {
        let us = duration.as_micros() as u64;
        let bucket_idx = (us / self.bucket_size_us) as usize;
        let bucket_idx = bucket_idx.min(self.buckets.len() - 1);

        self.buckets[bucket_idx].fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.sum_us.fetch_add(us, Ordering::Relaxed);

        // Update min/max
        loop {
            let current_min = self.min_us.load(Ordering::Relaxed);
            if us >= current_min {
                break;
            }
            if self
                .min_us
                .compare_exchange(current_min, us, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
        loop {
            let current_max = self.max_us.load(Ordering::Relaxed);
            if us <= current_max {
                break;
            }
            if self
                .max_us
                .compare_exchange(current_max, us, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }

    /// Get count of samples
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get mean latency
    pub fn mean(&self) -> Duration {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return Duration::ZERO;
        }
        let sum = self.sum_us.load(Ordering::Relaxed);
        Duration::from_micros(sum / count)
    }

    /// Get minimum latency
    pub fn min(&self) -> Duration {
        let min = self.min_us.load(Ordering::Relaxed);
        if min == u64::MAX {
            Duration::ZERO
        } else {
            Duration::from_micros(min)
        }
    }

    /// Get maximum latency
    pub fn max(&self) -> Duration {
        Duration::from_micros(self.max_us.load(Ordering::Relaxed))
    }

    /// Get a specific percentile (0.0 to 1.0)
    pub fn percentile(&self, p: f64) -> Duration {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return Duration::ZERO;
        }

        let target = ((count as f64) * p).ceil() as u64;
        let mut cumulative = 0u64;

        for (idx, bucket) in self.buckets.iter().enumerate() {
            cumulative += bucket.load(Ordering::Relaxed);
            if cumulative >= target {
                // Return the upper bound of this bucket
                let us = ((idx as u64) + 1) * self.bucket_size_us;
                return Duration::from_micros(us);
            }
        }

        self.max()
    }

    /// Get p50 (median)
    pub fn p50(&self) -> Duration {
        self.percentile(0.50)
    }

    /// Get p95
    pub fn p95(&self) -> Duration {
        self.percentile(0.95)
    }

    /// Get p99
    pub fn p99(&self) -> Duration {
        self.percentile(0.99)
    }

    /// Get p999
    pub fn p999(&self) -> Duration {
        self.percentile(0.999)
    }

    /// Generate summary statistics
    pub fn summary(&self) -> LatencySummary {
        LatencySummary {
            count: self.count(),
            mean: self.mean(),
            min: self.min(),
            max: self.max(),
            p50: self.p50(),
            p95: self.p95(),
            p99: self.p99(),
            p999: self.p999(),
        }
    }

    /// Reset the histogram
    pub fn reset(&self) {
        for bucket in &self.buckets {
            bucket.store(0, Ordering::Relaxed);
        }
        self.count.store(0, Ordering::Relaxed);
        self.sum_us.store(0, Ordering::Relaxed);
        self.min_us.store(u64::MAX, Ordering::Relaxed);
        self.max_us.store(0, Ordering::Relaxed);
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of latency statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencySummary {
    pub count: u64,
    #[serde(with = "duration_micros")]
    pub mean: Duration,
    #[serde(with = "duration_micros")]
    pub min: Duration,
    #[serde(with = "duration_micros")]
    pub max: Duration,
    #[serde(with = "duration_micros")]
    pub p50: Duration,
    #[serde(with = "duration_micros")]
    pub p95: Duration,
    #[serde(with = "duration_micros")]
    pub p99: Duration,
    #[serde(with = "duration_micros")]
    pub p999: Duration,
}

impl std::fmt::Display for LatencySummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "n={} mean={:?} min={:?} max={:?} p50={:?} p95={:?} p99={:?} p999={:?}",
            self.count, self.mean, self.min, self.max, self.p50, self.p95, self.p99, self.p999
        )
    }
}

/// Serde helper for Duration as microseconds
mod duration_micros {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_micros() as u64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let us = u64::deserialize(deserializer)?;
        Ok(Duration::from_micros(us))
    }
}

/// CPU utilization monitor
#[derive(Debug)]
pub struct CpuMonitor {
    samples: Arc<RwLock<Vec<CpuSample>>>,
    running: Arc<std::sync::atomic::AtomicBool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuSample {
    pub timestamp_ms: u64,
    pub cpu_percent: f64,
    pub user_percent: f64,
    pub system_percent: f64,
}

impl CpuMonitor {
    /// Create a new CPU monitor
    pub fn new() -> Self {
        Self {
            samples: Arc::new(RwLock::new(Vec::new())),
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Start monitoring CPU at the given interval
    pub fn start(&self, interval: Duration) -> tokio::task::JoinHandle<()> {
        self.running
            .store(true, std::sync::atomic::Ordering::SeqCst);
        let samples = self.samples.clone();
        let running = self.running.clone();
        let start = Instant::now();

        tokio::spawn(async move {
            let mut last_times = Self::get_cpu_times();
            let mut interval_timer = tokio::time::interval(interval);

            while running.load(std::sync::atomic::Ordering::SeqCst) {
                interval_timer.tick().await;
                let current_times = Self::get_cpu_times();

                if let (Some(last), Some(current)) = (last_times, current_times) {
                    let total_delta = (current.0 - last.0) as f64;
                    if total_delta > 0.0 {
                        let user_delta = (current.1 - last.1) as f64;
                        let system_delta = (current.2 - last.2) as f64;
                        let idle_delta = (current.3 - last.3) as f64;

                        let busy = total_delta - idle_delta;
                        let sample = CpuSample {
                            timestamp_ms: start.elapsed().as_millis() as u64,
                            cpu_percent: (busy / total_delta) * 100.0,
                            user_percent: (user_delta / total_delta) * 100.0,
                            system_percent: (system_delta / total_delta) * 100.0,
                        };

                        samples.write().await.push(sample);
                    }
                }

                last_times = current_times;
            }
        })
    }

    /// Stop monitoring
    pub fn stop(&self) {
        self.running
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Get collected samples
    pub async fn samples(&self) -> Vec<CpuSample> {
        self.samples.read().await.clone()
    }

    /// Get CPU summary
    pub async fn summary(&self) -> CpuSummary {
        let samples = self.samples.read().await;
        if samples.is_empty() {
            return CpuSummary::default();
        }

        let mut total = 0.0;
        let mut max = 0.0f64;
        let mut user_total = 0.0;
        let mut system_total = 0.0;

        for s in samples.iter() {
            total += s.cpu_percent;
            max = max.max(s.cpu_percent);
            user_total += s.user_percent;
            system_total += s.system_percent;
        }

        let count = samples.len() as f64;
        CpuSummary {
            sample_count: samples.len(),
            avg_cpu_percent: total / count,
            max_cpu_percent: max,
            avg_user_percent: user_total / count,
            avg_system_percent: system_total / count,
        }
    }

    /// Read CPU times from /proc/stat (Linux only)
    #[cfg(target_os = "linux")]
    fn get_cpu_times() -> Option<(u64, u64, u64, u64)> {
        let contents = std::fs::read_to_string("/proc/stat").ok()?;
        let line = contents.lines().next()?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 5 || parts[0] != "cpu" {
            return None;
        }

        let user: u64 = parts[1].parse().ok()?;
        let nice: u64 = parts[2].parse().ok()?;
        let system: u64 = parts[3].parse().ok()?;
        let idle: u64 = parts[4].parse().ok()?;
        let iowait: u64 = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);
        let irq: u64 = parts.get(6).and_then(|s| s.parse().ok()).unwrap_or(0);
        let softirq: u64 = parts.get(7).and_then(|s| s.parse().ok()).unwrap_or(0);

        let total = user + nice + system + idle + iowait + irq + softirq;
        Some((total, user + nice, system, idle + iowait))
    }

    #[cfg(not(target_os = "linux"))]
    fn get_cpu_times() -> Option<(u64, u64, u64, u64)> {
        // CPU monitoring not supported on this platform
        None
    }

    /// Clear collected samples
    pub async fn clear(&self) {
        self.samples.write().await.clear();
    }
}

impl Default for CpuMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// CPU usage summary
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CpuSummary {
    pub sample_count: usize,
    pub avg_cpu_percent: f64,
    pub max_cpu_percent: f64,
    pub avg_user_percent: f64,
    pub avg_system_percent: f64,
}

impl std::fmt::Display for CpuSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "samples={} avg={:.1}% max={:.1}% user={:.1}% sys={:.1}%",
            self.sample_count,
            self.avg_cpu_percent,
            self.max_cpu_percent,
            self.avg_user_percent,
            self.avg_system_percent
        )
    }
}

/// Throughput counter
#[derive(Debug, Default)]
pub struct ThroughputCounter {
    requests: AtomicU64,
    successes: AtomicU64,
    failures: AtomicU64,
    bytes_sent: AtomicU64,
    bytes_received: AtomicU64,
    start: std::sync::Mutex<Option<Instant>>,
    end: std::sync::Mutex<Option<Instant>>,
}

impl ThroughputCounter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn start(&self) {
        *self.start.lock().unwrap() = Some(Instant::now());
    }

    pub fn stop(&self) {
        *self.end.lock().unwrap() = Some(Instant::now());
    }

    pub fn record_request(&self, success: bool, bytes_sent: usize, bytes_received: usize) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        if success {
            self.successes.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failures.fetch_add(1, Ordering::Relaxed);
        }
        self.bytes_sent
            .fetch_add(bytes_sent as u64, Ordering::Relaxed);
        self.bytes_received
            .fetch_add(bytes_received as u64, Ordering::Relaxed);
    }

    pub fn summary(&self) -> ThroughputSummary {
        let start = self.start.lock().unwrap();
        let end = self.end.lock().unwrap();
        let duration = match (*start, *end) {
            (Some(s), Some(e)) => e.duration_since(s),
            (Some(s), None) => s.elapsed(),
            _ => Duration::ZERO,
        };

        let requests = self.requests.load(Ordering::Relaxed);
        let successes = self.successes.load(Ordering::Relaxed);
        let bytes_sent = self.bytes_sent.load(Ordering::Relaxed);
        let bytes_received = self.bytes_received.load(Ordering::Relaxed);

        let secs = duration.as_secs_f64();
        ThroughputSummary {
            total_requests: requests,
            successful_requests: successes,
            failed_requests: self.failures.load(Ordering::Relaxed),
            duration,
            requests_per_second: if secs > 0.0 {
                requests as f64 / secs
            } else {
                0.0
            },
            bytes_sent,
            bytes_received,
            megabits_per_second_sent: if secs > 0.0 {
                (bytes_sent as f64 * 8.0) / (secs * 1_000_000.0)
            } else {
                0.0
            },
            megabits_per_second_received: if secs > 0.0 {
                (bytes_received as f64 * 8.0) / (secs * 1_000_000.0)
            } else {
                0.0
            },
        }
    }

    pub fn reset(&self) {
        self.requests.store(0, Ordering::Relaxed);
        self.successes.store(0, Ordering::Relaxed);
        self.failures.store(0, Ordering::Relaxed);
        self.bytes_sent.store(0, Ordering::Relaxed);
        self.bytes_received.store(0, Ordering::Relaxed);
        *self.start.lock().unwrap() = None;
        *self.end.lock().unwrap() = None;
    }
}

/// Throughput summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputSummary {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    #[serde(with = "duration_millis")]
    pub duration: Duration,
    pub requests_per_second: f64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub megabits_per_second_sent: f64,
    pub megabits_per_second_received: f64,
}

impl std::fmt::Display for ThroughputSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "total={} success={} failed={} rps={:.1} sent={:.1}Mbps recv={:.1}Mbps",
            self.total_requests,
            self.successful_requests,
            self.failed_requests,
            self.requests_per_second,
            self.megabits_per_second_sent,
            self.megabits_per_second_received
        )
    }
}

mod duration_millis {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_millis() as u64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let ms = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(ms))
    }
}

/// Memory usage tracker
#[derive(Debug, Default)]
pub struct MemoryTracker {
    initial_bytes: AtomicU64,
    peak_bytes: AtomicU64,
    samples: Arc<RwLock<Vec<MemorySample>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySample {
    pub timestamp_ms: u64,
    pub rss_bytes: u64,
    pub heap_bytes: u64,
}

impl MemoryTracker {
    pub fn new() -> Self {
        let tracker = Self::default();
        if let Some(mem) = Self::get_memory_usage() {
            tracker.initial_bytes.store(mem.0, Ordering::Relaxed);
            tracker.peak_bytes.store(mem.0, Ordering::Relaxed);
        }
        tracker
    }

    pub async fn record(&self, start: Instant) {
        if let Some((rss, heap)) = Self::get_memory_usage() {
            // Update peak
            loop {
                let current_peak = self.peak_bytes.load(Ordering::Relaxed);
                if rss <= current_peak {
                    break;
                }
                if self
                    .peak_bytes
                    .compare_exchange(current_peak, rss, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
                {
                    break;
                }
            }

            self.samples.write().await.push(MemorySample {
                timestamp_ms: start.elapsed().as_millis() as u64,
                rss_bytes: rss,
                heap_bytes: heap,
            });
        }
    }

    pub fn summary(&self) -> MemorySummary {
        let initial = self.initial_bytes.load(Ordering::Relaxed);
        let peak = self.peak_bytes.load(Ordering::Relaxed);
        MemorySummary {
            initial_bytes: initial,
            peak_bytes: peak,
            growth_bytes: peak.saturating_sub(initial),
            growth_ratio: if initial > 0 {
                peak as f64 / initial as f64
            } else {
                1.0
            },
        }
    }

    #[cfg(target_os = "linux")]
    fn get_memory_usage() -> Option<(u64, u64)> {
        // Read from /proc/self/statm
        let contents = std::fs::read_to_string("/proc/self/statm").ok()?;
        let parts: Vec<&str> = contents.split_whitespace().collect();
        if parts.len() < 2 {
            return None;
        }

        let page_size = 4096u64; // Typical page size
        let vsize: u64 = parts[0].parse().ok()?;
        let rss: u64 = parts[1].parse().ok()?;

        Some((rss * page_size, vsize * page_size))
    }

    #[cfg(not(target_os = "linux"))]
    fn get_memory_usage() -> Option<(u64, u64)> {
        None
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemorySummary {
    pub initial_bytes: u64,
    pub peak_bytes: u64,
    pub growth_bytes: u64,
    pub growth_ratio: f64,
}

impl std::fmt::Display for MemorySummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "initial={:.1}MB peak={:.1}MB growth={:.1}MB ratio={:.2}x",
            self.initial_bytes as f64 / 1_048_576.0,
            self.peak_bytes as f64 / 1_048_576.0,
            self.growth_bytes as f64 / 1_048_576.0,
            self.growth_ratio
        )
    }
}

/// Combined performance metrics for a test run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfMetrics {
    pub test_name: String,
    pub config_summary: String,
    pub latency: Option<LatencySummary>,
    pub throughput: Option<ThroughputSummary>,
    pub cpu: Option<CpuSummary>,
    pub memory: Option<MemorySummary>,
    pub custom_metrics: BTreeMap<String, f64>,
}

impl PerfMetrics {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            test_name: name.into(),
            config_summary: String::new(),
            latency: None,
            throughput: None,
            cpu: None,
            memory: None,
            custom_metrics: BTreeMap::new(),
        }
    }

    pub fn with_config(mut self, summary: impl Into<String>) -> Self {
        self.config_summary = summary.into();
        self
    }

    pub fn with_latency(mut self, latency: LatencySummary) -> Self {
        self.latency = Some(latency);
        self
    }

    pub fn with_throughput(mut self, throughput: ThroughputSummary) -> Self {
        self.throughput = Some(throughput);
        self
    }

    pub fn with_cpu(mut self, cpu: CpuSummary) -> Self {
        self.cpu = Some(cpu);
        self
    }

    pub fn with_memory(mut self, memory: MemorySummary) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn with_custom(mut self, key: impl Into<String>, value: f64) -> Self {
        self.custom_metrics.insert(key.into(), value);
        self
    }

    /// Print a human-readable report
    pub fn print_report(&self) {
        println!("\n=== {} ===", self.test_name);
        if !self.config_summary.is_empty() {
            println!("Config: {}", self.config_summary);
        }
        if let Some(ref lat) = self.latency {
            println!("Latency: {}", lat);
        }
        if let Some(ref tp) = self.throughput {
            println!("Throughput: {}", tp);
        }
        if let Some(ref cpu) = self.cpu {
            println!("CPU: {}", cpu);
        }
        if let Some(ref mem) = self.memory {
            println!("Memory: {}", mem);
        }
        for (key, value) in &self.custom_metrics {
            println!("{}: {:.3}", key, value);
        }
        println!();
    }

    /// Convert to JSON string
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

/// Collection of metrics from multiple test runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfReport {
    pub timestamp: String,
    pub hardware: String,
    pub results: Vec<PerfMetrics>,
}

impl PerfReport {
    pub fn new() -> Self {
        Self {
            timestamp: chrono::Utc::now().to_rfc3339(),
            hardware: format!(
                "cpus={}, containerized={}",
                num_cpus::get(),
                std::path::Path::new("/.dockerenv").exists()
            ),
            results: Vec::new(),
        }
    }

    pub fn add(&mut self, metrics: PerfMetrics) {
        self.results.push(metrics);
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }
}

impl Default for PerfReport {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_basic() {
        let hist = LatencyHistogram::new();
        hist.record(Duration::from_millis(10));
        hist.record(Duration::from_millis(20));
        hist.record(Duration::from_millis(30));

        assert_eq!(hist.count(), 3);
        assert!(hist.min() >= Duration::from_millis(10));
        assert!(hist.max() <= Duration::from_millis(31)); // bucket boundary
    }

    #[test]
    fn test_histogram_percentiles() {
        let hist = LatencyHistogram::new();

        // Add 100 samples from 1ms to 100ms
        for i in 1..=100 {
            hist.record(Duration::from_millis(i));
        }

        // p50 should be around 50ms
        let p50 = hist.p50();
        assert!(p50 >= Duration::from_millis(49) && p50 <= Duration::from_millis(52));

        // p99 should be around 99ms
        let p99 = hist.p99();
        assert!(p99 >= Duration::from_millis(98) && p99 <= Duration::from_millis(101));
    }

    #[test]
    fn test_throughput_counter() {
        let counter = ThroughputCounter::new();
        counter.start();
        counter.record_request(true, 100, 200);
        counter.record_request(true, 100, 200);
        counter.record_request(false, 100, 0);
        counter.stop();

        let summary = counter.summary();
        assert_eq!(summary.total_requests, 3);
        assert_eq!(summary.successful_requests, 2);
        assert_eq!(summary.failed_requests, 1);
    }

    #[test]
    fn test_perf_metrics() {
        let metrics = PerfMetrics::new("test")
            .with_config("iterations=100")
            .with_custom("overhead_percent", 5.5);

        assert_eq!(metrics.test_name, "test");
        assert_eq!(metrics.custom_metrics.get("overhead_percent"), Some(&5.5));
    }
}
