//! Load and Throughput Performance Tests
//!
//! These tests measure the MCP server's behavior under sustained load:
//! - Maximum throughput (requests per second)
//! - Latency under load (latency degradation)
//! - CPU utilization as throughput increases
//! - Scalability with increasing client count
//!
//! Critical for capacity planning in enterprise deployments.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::config::PerfConfig;
use super::harness::MockMcpServer;
use super::metrics::{CpuMonitor, LatencyHistogram, PerfMetrics, PerfReport, ThroughputCounter};

/// Run all load performance tests
pub async fn run_load_tests(config: &PerfConfig) -> PerfReport {
    let mut report = PerfReport::new();

    println!("\n=== Load and Throughput Tests ===\n");

    // Test 1: Maximum throughput
    let max_throughput = test_maximum_throughput(config).await;
    max_throughput.print_report();
    report.add(max_throughput);

    // Test 2: Latency under increasing load
    let latency_under_load = test_latency_under_load(config).await;
    for m in &latency_under_load {
        m.print_report();
    }
    for m in latency_under_load {
        report.add(m);
    }

    // Test 3: CPU utilization curve
    let cpu_curve = test_cpu_utilization_curve(config).await;
    cpu_curve.print_report();
    report.add(cpu_curve);

    // Test 4: Scalability test
    let scalability = test_scalability(config).await;
    for m in &scalability {
        m.print_report();
    }
    for m in scalability {
        report.add(m);
    }

    // Test 5: Burst load handling
    let burst_metrics = test_burst_load(config).await;
    burst_metrics.print_report();
    report.add(burst_metrics);

    report
}

/// Test maximum sustainable throughput
pub async fn test_maximum_throughput(config: &PerfConfig) -> PerfMetrics {
    let server = MockMcpServer::start(Duration::ZERO).await;
    let url = format!("http://{}/mcp", server.addr);

    let counter = Arc::new(ThroughputCounter::new());
    let histogram = Arc::new(LatencyHistogram::new());
    let cpu_monitor = CpuMonitor::new();
    let running = Arc::new(AtomicBool::new(true));

    let concurrency = config.effective_clients() * 2; // Higher concurrency for max throughput
    let duration = config.load_duration;

    // Start CPU monitoring
    let cpu_handle = cpu_monitor.start(config.cpu_sample_interval);
    counter.start();

    let mut handles = Vec::new();

    for client_id in 0..concurrency {
        let counter = counter.clone();
        let histogram = histogram.clone();
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
                            "arguments": {"client": client_id, "ts": req_start.elapsed().as_nanos()}
                        },
                        "id": client_id
                    }))
                    .send()
                    .await;

                match result {
                    Ok(resp) => {
                        if let Ok(bytes) = resp.bytes().await {
                            histogram.record(req_start.elapsed());
                            counter.record_request(true, 150, bytes.len());
                        } else {
                            counter.record_request(false, 150, 0);
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

    // Run for duration
    tokio::time::sleep(duration).await;

    // Stop workers
    running.store(false, Ordering::Relaxed);
    tokio::time::sleep(Duration::from_millis(100)).await;

    for handle in handles {
        handle.abort();
    }

    counter.stop();
    cpu_monitor.stop();
    cpu_handle.abort();

    server.shutdown().await;

    let cpu_summary = cpu_monitor.summary().await;

    PerfMetrics::new("Maximum Throughput")
        .with_config(format!(
            "duration={:?}, concurrency={}",
            duration, concurrency
        ))
        .with_latency(histogram.summary())
        .with_throughput(counter.summary())
        .with_cpu(cpu_summary)
}

/// Test latency characteristics under increasing load levels
pub async fn test_latency_under_load(config: &PerfConfig) -> Vec<PerfMetrics> {
    let server = MockMcpServer::start(Duration::ZERO).await;
    let url = format!("http://{}/mcp", server.addr);

    let base_concurrency = config.effective_clients();
    let load_levels = if config.verify_mode {
        vec![1, 2]
    } else {
        vec![1, 2, 5, 10, 20, 50]
    };

    let mut results = Vec::new();
    let measurement_duration = if config.verify_mode {
        Duration::from_millis(500)
    } else {
        Duration::from_secs(5)
    };

    for &multiplier in &load_levels {
        let concurrency = (base_concurrency * multiplier).min(100);
        let counter = Arc::new(ThroughputCounter::new());
        let histogram = Arc::new(LatencyHistogram::new());
        let running = Arc::new(AtomicBool::new(true));

        counter.start();

        let mut handles = Vec::new();

        for client_id in 0..concurrency {
            let counter = counter.clone();
            let histogram = histogram.clone();
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

        tokio::time::sleep(measurement_duration).await;

        running.store(false, Ordering::Relaxed);
        tokio::time::sleep(Duration::from_millis(50)).await;

        for handle in handles {
            handle.abort();
        }

        counter.stop();

        let throughput = counter.summary();
        let latency = histogram.summary();

        results.push(
            PerfMetrics::new(format!("Latency at {}x Load", multiplier))
                .with_config(format!("concurrency={}", concurrency))
                .with_latency(latency)
                .with_throughput(throughput)
                .with_custom("load_multiplier", multiplier as f64)
                .with_custom("concurrency", concurrency as f64),
        );

        // Brief pause between load levels
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    server.shutdown().await;

    results
}

/// Test CPU utilization as throughput increases
pub async fn test_cpu_utilization_curve(config: &PerfConfig) -> PerfMetrics {
    let server = MockMcpServer::start(Duration::ZERO).await;
    let url = format!("http://{}/mcp", server.addr);

    let cpu_monitor = CpuMonitor::new();
    let rps_samples = Arc::new(tokio::sync::RwLock::new(Vec::<(f64, f64)>::new()));
    let running = Arc::new(AtomicBool::new(true));

    let concurrency = config.effective_clients() * 2;
    let duration = config.load_duration;

    // Start CPU monitoring
    let cpu_handle = cpu_monitor.start(config.cpu_sample_interval);

    // Request counter for RPS calculation
    let request_count = Arc::new(AtomicU64::new(0));

    let mut handles = Vec::new();

    for client_id in 0..concurrency {
        let running = running.clone();
        let request_count = request_count.clone();
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

                if result.is_ok() {
                    request_count.fetch_add(1, Ordering::Relaxed);
                }
            }
        });

        handles.push(handle);
    }

    // Sample RPS and CPU periodically
    let sample_interval = Duration::from_secs(1);
    let start = Instant::now();

    while start.elapsed() < duration {
        let count_before = request_count.load(Ordering::Relaxed);
        tokio::time::sleep(sample_interval).await;
        let count_after = request_count.load(Ordering::Relaxed);

        let rps = (count_after - count_before) as f64;
        let cpu = cpu_monitor.summary().await.avg_cpu_percent;

        rps_samples.write().await.push((rps, cpu));
    }

    running.store(false, Ordering::Relaxed);
    tokio::time::sleep(Duration::from_millis(100)).await;

    for handle in handles {
        handle.abort();
    }

    cpu_monitor.stop();
    cpu_handle.abort();

    server.shutdown().await;

    let samples = rps_samples.read().await;
    let avg_rps = if samples.is_empty() {
        0.0
    } else {
        samples.iter().map(|(r, _)| r).sum::<f64>() / samples.len() as f64
    };
    let avg_cpu = if samples.is_empty() {
        0.0
    } else {
        samples.iter().map(|(_, c)| c).sum::<f64>() / samples.len() as f64
    };
    let max_rps = samples.iter().map(|(r, _)| *r).fold(0.0f64, f64::max);
    let max_cpu = samples.iter().map(|(_, c)| *c).fold(0.0f64, f64::max);

    // Calculate efficiency (RPS per % CPU)
    let efficiency = if avg_cpu > 0.0 {
        avg_rps / avg_cpu
    } else {
        0.0
    };

    PerfMetrics::new("CPU Utilization Curve")
        .with_config(format!(
            "duration={:?}, concurrency={}",
            duration, concurrency
        ))
        .with_cpu(cpu_monitor.summary().await)
        .with_custom("avg_rps", avg_rps)
        .with_custom("max_rps", max_rps)
        .with_custom("avg_cpu_percent", avg_cpu)
        .with_custom("max_cpu_percent", max_cpu)
        .with_custom("rps_per_cpu_percent", efficiency)
        .with_custom("sample_count", samples.len() as f64)
}

/// Test scalability with increasing client count
pub async fn test_scalability(config: &PerfConfig) -> Vec<PerfMetrics> {
    let server = MockMcpServer::start(Duration::ZERO).await;
    let url = format!("http://{}/mcp", server.addr);

    let client_counts = if config.verify_mode {
        vec![1, 2, 4]
    } else {
        vec![1, 2, 4, 8, 16, 32, 64]
    };

    let mut results = Vec::new();
    let measurement_duration = if config.verify_mode {
        Duration::from_millis(500)
    } else {
        Duration::from_secs(5)
    };

    for &client_count in &client_counts {
        let counter = Arc::new(ThroughputCounter::new());
        let histogram = Arc::new(LatencyHistogram::new());
        let running = Arc::new(AtomicBool::new(true));

        counter.start();

        let mut handles = Vec::new();

        for client_id in 0..client_count {
            let counter = counter.clone();
            let histogram = histogram.clone();
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

        tokio::time::sleep(measurement_duration).await;

        running.store(false, Ordering::Relaxed);
        tokio::time::sleep(Duration::from_millis(50)).await;

        for handle in handles {
            handle.abort();
        }

        counter.stop();

        let throughput = counter.summary();
        let rps_per_client = throughput.requests_per_second / client_count as f64;

        results.push(
            PerfMetrics::new(format!("Scalability - {} Clients", client_count))
                .with_latency(histogram.summary())
                .with_throughput(throughput)
                .with_custom("client_count", client_count as f64)
                .with_custom("rps_per_client", rps_per_client),
        );

        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    server.shutdown().await;

    results
}

/// Test burst load handling (sudden spike in traffic)
pub async fn test_burst_load(config: &PerfConfig) -> PerfMetrics {
    let server = MockMcpServer::start(Duration::ZERO).await;
    let url = format!("http://{}/mcp", server.addr);

    let baseline_clients = config.effective_clients();
    let burst_clients = baseline_clients * 5;
    let burst_duration = if config.verify_mode {
        Duration::from_millis(200)
    } else {
        Duration::from_secs(2)
    };

    let counter = Arc::new(ThroughputCounter::new());
    let baseline_histogram = Arc::new(LatencyHistogram::new());
    let burst_histogram = Arc::new(LatencyHistogram::new());
    let running = Arc::new(AtomicBool::new(true));
    let in_burst = Arc::new(AtomicBool::new(false));

    counter.start();

    // Start baseline clients
    let mut handles = Vec::new();

    for client_id in 0..baseline_clients {
        let counter = counter.clone();
        let baseline_histogram = baseline_histogram.clone();
        let burst_histogram = burst_histogram.clone();
        let running = running.clone();
        let in_burst = in_burst.clone();
        let url = url.clone();

        let handle = tokio::spawn(async move {
            let client = reqwest::Client::new();

            while running.load(Ordering::Relaxed) {
                let req_start = Instant::now();
                let is_burst = in_burst.load(Ordering::Relaxed);

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
                            let elapsed = req_start.elapsed();
                            if is_burst {
                                burst_histogram.record(elapsed);
                            } else {
                                baseline_histogram.record(elapsed);
                            }
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

    // Baseline measurement
    tokio::time::sleep(burst_duration).await;

    // Trigger burst
    in_burst.store(true, Ordering::Relaxed);

    // Add burst clients
    for client_id in baseline_clients..(baseline_clients + burst_clients) {
        let counter = counter.clone();
        let burst_histogram = burst_histogram.clone();
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
                            "arguments": {"client": client_id, "burst": true}
                        },
                        "id": client_id
                    }))
                    .send()
                    .await;

                match result {
                    Ok(resp) => {
                        if let Ok(bytes) = resp.bytes().await {
                            burst_histogram.record(req_start.elapsed());
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

    // Burst measurement
    tokio::time::sleep(burst_duration).await;

    running.store(false, Ordering::Relaxed);
    tokio::time::sleep(Duration::from_millis(100)).await;

    for handle in handles {
        handle.abort();
    }

    counter.stop();
    server.shutdown().await;

    let baseline_summary = baseline_histogram.summary();
    let burst_summary = burst_histogram.summary();

    let baseline_p50_us = baseline_summary.p50.as_micros() as f64;
    let burst_p50_us = burst_summary.p50.as_micros() as f64;

    let latency_degradation = if baseline_p50_us > 0.0 {
        ((burst_p50_us - baseline_p50_us) / baseline_p50_us) * 100.0
    } else {
        0.0
    };

    PerfMetrics::new("Burst Load Handling")
        .with_config(format!(
            "baseline_clients={}, burst_clients={}",
            baseline_clients, burst_clients
        ))
        .with_latency(burst_summary)
        .with_throughput(counter.summary())
        .with_custom("baseline_p50_us", baseline_p50_us)
        .with_custom("burst_p50_us", burst_p50_us)
        .with_custom("latency_degradation_percent", latency_degradation)
        .with_custom("baseline_clients", baseline_clients as f64)
        .with_custom("burst_clients", burst_clients as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_load_max_throughput() {
        let config = PerfConfig::verification_config();
        let metrics = test_maximum_throughput(&config).await;
        assert!(metrics.throughput.is_some());
    }

    #[tokio::test]
    async fn test_load_scalability() {
        let config = PerfConfig::verification_config();
        let results = test_scalability(&config).await;
        assert!(!results.is_empty());
    }
}
