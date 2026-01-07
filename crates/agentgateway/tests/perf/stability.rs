//! Long-Running Stability Tests
//!
//! These tests verify the MCP server's stability under sustained operation:
//! - Memory leak detection
//! - Performance consistency over time
//! - Resource exhaustion handling
//! - Connection pool health
//!
//! Essential for production deployments running 24/7.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::config::PerfConfig;
use super::harness::MockMcpServer;
use super::metrics::{
    LatencyHistogram, MemoryTracker, PerfMetrics, PerfReport, ThroughputCounter,
};

/// Run all stability tests
pub async fn run_stability_tests(config: &PerfConfig) -> PerfReport {
    let mut report = PerfReport::new();

    println!("\n=== Long-Running Stability Tests ===\n");

    // Test 1: Memory stability
    let memory_metrics = test_memory_stability(config).await;
    memory_metrics.print_report();
    report.add(memory_metrics);

    // Test 2: Performance consistency
    let consistency_metrics = test_performance_consistency(config).await;
    for m in &consistency_metrics {
        m.print_report();
    }
    for m in consistency_metrics {
        report.add(m);
    }

    // Test 3: Connection pool stability
    let pool_metrics = test_connection_pool_stability(config).await;
    pool_metrics.print_report();
    report.add(pool_metrics);

    // Test 4: Error rate over time
    let error_metrics = test_error_rate_stability(config).await;
    error_metrics.print_report();
    report.add(error_metrics);

    report
}

/// Test memory stability over time (detect memory leaks)
pub async fn test_memory_stability(config: &PerfConfig) -> PerfMetrics {
    let server = MockMcpServer::start(Duration::ZERO).await;
    let url = format!("http://{}/mcp", server.addr);

    let memory_tracker = MemoryTracker::new();
    let counter = Arc::new(ThroughputCounter::new());
    let running = Arc::new(AtomicBool::new(true));

    let concurrency = config.effective_clients();
    let duration = config.stability_duration;
    let memory_check_interval = config.memory_check_interval;

    counter.start();
    let start = Instant::now();

    // Spawn worker tasks
    let mut handles = Vec::new();

    for client_id in 0..concurrency {
        let counter = counter.clone();
        let running = running.clone();
        let url = url.clone();

        let handle = tokio::spawn(async move {
            let client = reqwest::Client::new();

            while running.load(Ordering::Relaxed) {
                let result = client
                    .post(&url)
                    .json(&serde_json::json!({
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": "echo",
                            "arguments": {"client": client_id}
                        },
                        "id": client_id
                    }))
                    .send()
                    .await;

                match result {
                    Ok(resp) => {
                        if let Ok(bytes) = resp.bytes().await {
                            counter.record_request(true, 100, bytes.len());
                        } else {
                            counter.record_request(false, 100, 0);
                        }
                    }
                    Err(_) => {
                        counter.record_request(false, 0, 0);
                    }
                }
            }
        });

        handles.push(handle);
    }

    // Memory sampling loop
    let memory_samples = Arc::new(tokio::sync::RwLock::new(Vec::new()));
    let memory_samples_clone = memory_samples.clone();

    let memory_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(memory_check_interval);
        while start.elapsed() < duration {
            interval.tick().await;
            memory_tracker.record(start).await;
            memory_samples_clone.write().await.push(memory_tracker.summary());
        }
        memory_tracker.summary()
    });

    // Wait for duration
    tokio::time::sleep(duration).await;

    running.store(false, Ordering::Relaxed);
    tokio::time::sleep(Duration::from_millis(100)).await;

    for handle in handles {
        handle.abort();
    }

    counter.stop();

    let memory_summary = memory_task.await.unwrap_or_default();
    server.shutdown().await;

    // Analyze memory trend
    let samples = memory_samples.read().await;
    let (growth_trend, is_stable) = analyze_memory_trend(&samples, config.max_memory_growth_ratio);

    PerfMetrics::new("Memory Stability")
        .with_config(format!(
            "duration={:?}, concurrency={}",
            duration, concurrency
        ))
        .with_throughput(counter.summary())
        .with_memory(memory_summary)
        .with_custom("memory_samples", samples.len() as f64)
        .with_custom("growth_trend_percent_per_hour", growth_trend)
        .with_custom("is_stable", if is_stable { 1.0 } else { 0.0 })
}

/// Test performance consistency over time
pub async fn test_performance_consistency(config: &PerfConfig) -> Vec<PerfMetrics> {
    let server = MockMcpServer::start(Duration::ZERO).await;
    let url = format!("http://{}/mcp", server.addr);

    let num_windows = if config.verify_mode { 3 } else { 10 };
    let window_duration = config.stability_duration / num_windows as u32;
    let concurrency = config.effective_clients();

    let mut results = Vec::new();

    for window in 0..num_windows {
        let histogram = Arc::new(LatencyHistogram::new());
        let counter = Arc::new(ThroughputCounter::new());
        let running = Arc::new(AtomicBool::new(true));

        counter.start();

        let mut handles = Vec::new();

        for client_id in 0..concurrency {
            let histogram = histogram.clone();
            let counter = counter.clone();
            let running = running.clone();
            let url = url.clone();

            let handle = tokio::spawn(async move {
                let client = reqwest::Client::new();

                while running.load(Ordering::Relaxed) {
                    let req_start = Instant::now();

                    let result = client
                        .post(&url)
                        .json(&serde_json::json!({
                            "jsonrpc": "2.0",
                            "method": "tools/call",
                            "params": {
                                "name": "echo",
                                "arguments": {"client": client_id, "window": window}
                            },
                            "id": client_id
                        }))
                        .send()
                        .await;

                    match result {
                        Ok(resp) => {
                            if let Ok(bytes) = resp.bytes().await {
                                histogram.record(req_start.elapsed());
                                counter.record_request(true, 100, bytes.len());
                            } else {
                                counter.record_request(false, 100, 0);
                            }
                        }
                        Err(_) => {
                            counter.record_request(false, 0, 0);
                        }
                    }
                }
            });

            handles.push(handle);
        }

        tokio::time::sleep(window_duration).await;

        running.store(false, Ordering::Relaxed);
        tokio::time::sleep(Duration::from_millis(50)).await;

        for handle in handles {
            handle.abort();
        }

        counter.stop();

        results.push(
            PerfMetrics::new(format!("Performance Window {}/{}", window + 1, num_windows))
                .with_config(format!(
                    "window_duration={:?}, concurrency={}",
                    window_duration, concurrency
                ))
                .with_latency(histogram.summary())
                .with_throughput(counter.summary())
                .with_custom("window_number", (window + 1) as f64),
        );
    }

    server.shutdown().await;

    // Calculate consistency score
    if results.len() >= 2 {
        let p50_values: Vec<f64> = results
            .iter()
            .filter_map(|r| r.latency.as_ref())
            .map(|l| l.p50.as_micros() as f64)
            .collect();

        if !p50_values.is_empty() {
            let mean = p50_values.iter().sum::<f64>() / p50_values.len() as f64;
            let variance = p50_values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / p50_values.len() as f64;
            let std_dev = variance.sqrt();
            let cv = if mean > 0.0 { std_dev / mean * 100.0 } else { 0.0 };

            results.push(
                PerfMetrics::new("Performance Consistency Summary")
                    .with_config(format!("windows={}", num_windows))
                    .with_custom("mean_p50_us", mean)
                    .with_custom("std_dev_us", std_dev)
                    .with_custom("coefficient_of_variation_percent", cv)
                    .with_custom(
                        "is_consistent",
                        if cv < 20.0 { 1.0 } else { 0.0 },
                    ),
            );
        }
    }

    results
}

/// Test connection pool stability under sustained load
pub async fn test_connection_pool_stability(config: &PerfConfig) -> PerfMetrics {
    let server = MockMcpServer::start(Duration::ZERO).await;
    let url = format!("http://{}/mcp", server.addr);

    let concurrency = config.effective_clients() * 2;
    let duration = config.stability_duration;
    let connection_churn_interval = Duration::from_secs(5);

    let counter = Arc::new(ThroughputCounter::new());
    let histogram = Arc::new(LatencyHistogram::new());
    let running = Arc::new(AtomicBool::new(true));
    let client_rotations = Arc::new(AtomicU64::new(0));

    counter.start();

    // Spawn workers that periodically recreate their HTTP clients
    let mut handles = Vec::new();

    for client_id in 0..concurrency {
        let counter = counter.clone();
        let histogram = histogram.clone();
        let running = running.clone();
        let client_rotations = client_rotations.clone();
        let url = url.clone();

        let handle = tokio::spawn(async move {
            let mut client = reqwest::Client::new();
            let mut last_rotation = Instant::now();

            while running.load(Ordering::Relaxed) {
                // Periodically rotate the client to simulate connection churn
                if last_rotation.elapsed() > connection_churn_interval {
                    client = reqwest::Client::new();
                    client_rotations.fetch_add(1, Ordering::Relaxed);
                    last_rotation = Instant::now();
                }

                let req_start = Instant::now();

                let result = client
                    .post(&url)
                    .json(&serde_json::json!({
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": "echo",
                            "arguments": {"client": client_id}
                        },
                        "id": client_id
                    }))
                    .send()
                    .await;

                match result {
                    Ok(resp) => {
                        if let Ok(bytes) = resp.bytes().await {
                            histogram.record(req_start.elapsed());
                            counter.record_request(true, 100, bytes.len());
                        } else {
                            counter.record_request(false, 100, 0);
                        }
                    }
                    Err(_) => {
                        counter.record_request(false, 0, 0);
                    }
                }
            }
        });

        handles.push(handle);
    }

    tokio::time::sleep(duration).await;

    running.store(false, Ordering::Relaxed);
    tokio::time::sleep(Duration::from_millis(100)).await;

    for handle in handles {
        handle.abort();
    }

    counter.stop();
    server.shutdown().await;

    let throughput = counter.summary();
    let rotations = client_rotations.load(Ordering::Relaxed);

    let total_requests = throughput.total_requests;
    let successful_requests = throughput.successful_requests;
    let success_rate_percent = if total_requests > 0 {
        (successful_requests as f64 / total_requests as f64) * 100.0
    } else {
        0.0
    };

    PerfMetrics::new("Connection Pool Stability")
        .with_config(format!(
            "duration={:?}, concurrency={}, churn_interval={:?}",
            duration, concurrency, connection_churn_interval
        ))
        .with_latency(histogram.summary())
        .with_throughput(throughput)
        .with_custom("client_rotations", rotations as f64)
        .with_custom("success_rate_percent", success_rate_percent)
}

/// Test error rate stability over time
pub async fn test_error_rate_stability(config: &PerfConfig) -> PerfMetrics {
    let server = MockMcpServer::start(Duration::ZERO).await;
    let url = format!("http://{}/mcp", server.addr);

    let concurrency = config.effective_clients();
    let duration = config.stability_duration;
    let sample_interval = Duration::from_secs(5);

    let running = Arc::new(AtomicBool::new(true));
    let success_count = Arc::new(AtomicU64::new(0));
    let failure_count = Arc::new(AtomicU64::new(0));
    let error_rate_samples = Arc::new(tokio::sync::RwLock::new(Vec::new()));

    let start = Instant::now();

    // Spawn workers
    let mut handles = Vec::new();

    for client_id in 0..concurrency {
        let running = running.clone();
        let success_count = success_count.clone();
        let failure_count = failure_count.clone();
        let url = url.clone();

        let handle = tokio::spawn(async move {
            let client = reqwest::Client::new();

            while running.load(Ordering::Relaxed) {
                let result = client
                    .post(&url)
                    .json(&serde_json::json!({
                        "jsonrpc": "2.0",
                        "method": "tools/call",
                        "params": {
                            "name": "echo",
                            "arguments": {"client": client_id}
                        },
                        "id": client_id
                    }))
                    .send()
                    .await;

                match result {
                    Ok(resp) if resp.status().is_success() => {
                        success_count.fetch_add(1, Ordering::Relaxed);
                    }
                    _ => {
                        failure_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        });

        handles.push(handle);
    }

    // Sample error rates periodically
    let samples_clone = error_rate_samples.clone();
    let success_clone = success_count.clone();
    let failure_clone = failure_count.clone();

    let sampling_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(sample_interval);
        let mut last_success = 0u64;
        let mut last_failure = 0u64;

        while start.elapsed() < duration {
            interval.tick().await;

            let current_success = success_clone.load(Ordering::Relaxed);
            let current_failure = failure_clone.load(Ordering::Relaxed);

            let interval_success = current_success - last_success;
            let interval_failure = current_failure - last_failure;
            let interval_total = interval_success + interval_failure;

            let error_rate = if interval_total > 0 {
                (interval_failure as f64 / interval_total as f64) * 100.0
            } else {
                0.0
            };

            samples_clone.write().await.push(error_rate);

            last_success = current_success;
            last_failure = current_failure;
        }
    });

    tokio::time::sleep(duration).await;

    running.store(false, Ordering::Relaxed);
    tokio::time::sleep(Duration::from_millis(100)).await;

    for handle in handles {
        handle.abort();
    }

    sampling_task.abort();
    server.shutdown().await;

    let samples = error_rate_samples.read().await;
    let avg_error_rate = if samples.is_empty() {
        0.0
    } else {
        samples.iter().sum::<f64>() / samples.len() as f64
    };
    let max_error_rate = samples.iter().cloned().fold(0.0f64, f64::max);
    let total_success = success_count.load(Ordering::Relaxed);
    let total_failure = failure_count.load(Ordering::Relaxed);

    // Check for error rate spikes
    let spike_threshold = avg_error_rate * 3.0 + 1.0; // 3x average or at least 1%
    let spikes = samples.iter().filter(|&&r| r > spike_threshold).count();

    PerfMetrics::new("Error Rate Stability")
        .with_config(format!(
            "duration={:?}, concurrency={}",
            duration, concurrency
        ))
        .with_custom("avg_error_rate_percent", avg_error_rate)
        .with_custom("max_error_rate_percent", max_error_rate)
        .with_custom("total_success", total_success as f64)
        .with_custom("total_failure", total_failure as f64)
        .with_custom("sample_count", samples.len() as f64)
        .with_custom("error_spikes", spikes as f64)
        .with_custom(
            "is_stable",
            if spikes == 0 && avg_error_rate < 1.0 {
                1.0
            } else {
                0.0
            },
        )
}

/// Analyze memory growth trend
fn analyze_memory_trend(
    samples: &[super::metrics::MemorySummary],
    max_growth_ratio: f64,
) -> (f64, bool) {
    if samples.len() < 2 {
        return (0.0, true);
    }

    let first_memory = samples.first().map(|s| s.peak_bytes as f64).unwrap_or(1.0);
    let last_memory = samples.last().map(|s| s.peak_bytes as f64).unwrap_or(1.0);

    let growth_ratio = last_memory / first_memory;
    let is_stable = growth_ratio <= max_growth_ratio;

    // Calculate growth trend as percent per hour
    // (assuming samples are evenly spaced)
    let growth_percent = (growth_ratio - 1.0) * 100.0;
    let samples_per_hour = 3600.0 / 10.0; // Assuming 10-second intervals
    let growth_per_hour = growth_percent * (samples_per_hour / samples.len() as f64);

    (growth_per_hour, is_stable)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stability_memory() {
        let config = PerfConfig::verification_config();
        let metrics = test_memory_stability(&config).await;
        assert!(metrics.memory.is_some() || metrics.custom_metrics.contains_key("memory_samples"));
    }

    #[tokio::test]
    async fn test_stability_consistency() {
        let config = PerfConfig::verification_config();
        let results = test_performance_consistency(&config).await;
        assert!(!results.is_empty());
    }
}
