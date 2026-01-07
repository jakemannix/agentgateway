//! Large Payload Performance Tests
//!
//! These tests measure performance with varying payload sizes, critical for:
//! - Long context window LLM interactions
//! - Large tool responses (e.g., file contents, database results)
//! - Streaming large datasets through MCP
//!
//! Key considerations for enterprise multi-agent systems:
//! - Memory pressure under large payloads
//! - Latency scaling with payload size
//! - Throughput degradation with increased payload sizes

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::config::PerfConfig;
use super::harness::{MockMcpServer, call_tool_timed, create_mcp_client, verify_test_basics};
use super::metrics::{LatencyHistogram, MemoryTracker, PerfMetrics, PerfReport, ThroughputCounter};

/// Run all payload performance tests
pub async fn run_payload_tests(config: &PerfConfig) -> PerfReport {
    let mut report = PerfReport::new();

    println!("\n=== Large Payload Performance Tests ===\n");

    // Test 1: Request payload size impact
    let request_metrics = test_request_payload_sizes(config).await;
    for m in &request_metrics {
        m.print_report();
    }
    for m in request_metrics {
        report.add(m);
    }

    // Test 2: Response payload size impact
    let response_metrics = test_response_payload_sizes(config).await;
    for m in &response_metrics {
        m.print_report();
    }
    for m in response_metrics {
        report.add(m);
    }

    // Test 3: Bidirectional large payload
    let bidirectional_metrics = test_bidirectional_large_payload(config).await;
    bidirectional_metrics.print_report();
    report.add(bidirectional_metrics);

    // Test 4: Memory pressure under large payloads
    let memory_metrics = test_memory_pressure_large_payloads(config).await;
    memory_metrics.print_report();
    report.add(memory_metrics);

    // Test 5: Throughput with varying payload sizes
    let throughput_metrics = test_throughput_vs_payload_size(config).await;
    for m in &throughput_metrics {
        m.print_report();
    }
    for m in throughput_metrics {
        report.add(m);
    }

    report
}

/// Test latency impact of varying request payload sizes
pub async fn test_request_payload_sizes(config: &PerfConfig) -> Vec<PerfMetrics> {
    let server = MockMcpServer::start(Duration::ZERO).await;
    let client = create_mcp_client(server.addr).await.expect("Failed to create client");

    let payload_sizes = if config.verify_mode {
        vec![
            ("tiny", 64),
            ("small", config.small_payload_size),
            ("medium", config.medium_payload_size),
        ]
    } else {
        vec![
            ("tiny", 64),
            ("small", config.small_payload_size),
            ("medium", config.medium_payload_size),
            ("large", config.large_payload_size),
            ("xl", config.xl_payload_size),
        ]
    };

    let iterations = config.effective_iterations();
    let warmup = config.effective_warmup();
    let mut results = Vec::new();

    for (name, size) in payload_sizes {
        let histogram = LatencyHistogram::new();
        let payload = generate_request_payload(size);

        // Warmup
        for _ in 0..warmup {
            let _ = call_tool_timed(&client, "echo", payload.clone()).await;
        }

        // Measure
        for _ in 0..iterations {
            if let Ok((elapsed, _)) = call_tool_timed(&client, "echo", payload.clone()).await {
                histogram.record(elapsed);
            }
        }

        let summary = histogram.summary();
        let p50_ms = summary.p50.as_millis();
        let bytes_per_ms = if p50_ms > 0 {
            size as f64 / p50_ms as f64
        } else {
            0.0
        };
        results.push(
            PerfMetrics::new(format!("Request Payload - {}", name))
                .with_config(format!("size={} bytes, iterations={}", size, iterations))
                .with_latency(summary)
                .with_custom("payload_size_bytes", size as f64)
                .with_custom("bytes_per_ms", bytes_per_ms),
        );
    }

    server.shutdown().await;

    results
}

/// Test latency impact of varying response payload sizes
pub async fn test_response_payload_sizes(config: &PerfConfig) -> Vec<PerfMetrics> {
    let server = MockMcpServer::start(Duration::ZERO).await;
    let client = create_mcp_client(server.addr).await.expect("Failed to create client");

    let payload_sizes = if config.verify_mode {
        vec![
            ("tiny", 64),
            ("small", config.small_payload_size),
            ("medium", config.medium_payload_size),
        ]
    } else {
        vec![
            ("tiny", 64),
            ("small", config.small_payload_size),
            ("medium", config.medium_payload_size),
            ("large", config.large_payload_size),
            ("xl", config.xl_payload_size),
        ]
    };

    let iterations = config.effective_iterations();
    let warmup = config.effective_warmup();
    let mut results = Vec::new();

    for (name, size) in payload_sizes {
        let histogram = LatencyHistogram::new();

        // Warmup
        for _ in 0..warmup {
            let _ = call_tool_timed(
                &client,
                "large_response",
                serde_json::json!({"size": size}),
            )
            .await;
        }

        // Measure
        let mut total_bytes_received = 0usize;
        for _ in 0..iterations {
            if let Ok((elapsed, bytes)) = call_tool_timed(
                &client,
                "large_response",
                serde_json::json!({"size": size}),
            )
            .await
            {
                histogram.record(elapsed);
                total_bytes_received += bytes;
            }
        }

        let summary = histogram.summary();
        let sample_count = summary.count;
        let p50_secs = summary.p50.as_secs_f64();
        let avg_bytes = if sample_count > 0 {
            total_bytes_received as f64 / sample_count as f64
        } else {
            0.0
        };
        let throughput_mbps = if p50_secs > 0.0 {
            (avg_bytes * 8.0) / (p50_secs * 1_000_000.0)
        } else {
            0.0
        };

        results.push(
            PerfMetrics::new(format!("Response Payload - {}", name))
                .with_config(format!(
                    "requested_size={} bytes, iterations={}",
                    size, iterations
                ))
                .with_latency(summary)
                .with_custom("requested_size_bytes", size as f64)
                .with_custom("avg_received_bytes", avg_bytes)
                .with_custom("throughput_mbps", throughput_mbps),
        );
    }

    server.shutdown().await;

    results
}

/// Test with large payloads in both directions
pub async fn test_bidirectional_large_payload(config: &PerfConfig) -> PerfMetrics {
    let server = MockMcpServer::start(Duration::ZERO).await;
    let client = create_mcp_client(server.addr).await.expect("Failed to create client");

    let request_size = config.large_payload_size;
    let response_size = config.large_payload_size;
    let histogram = LatencyHistogram::new();
    let iterations = config.effective_iterations();
    let warmup = config.effective_warmup();

    // Create large request payload
    let request_data = "x".repeat(request_size);

    // Warmup
    for _ in 0..warmup {
        let _ = call_tool_timed(
            &client,
            "large_response",
            serde_json::json!({
                "size": response_size,
                "data": &request_data[..request_size.min(1000)] // Truncate for warmup
            }),
        )
        .await;
    }

    // Measure
    let mut total_sent = 0usize;
    let mut total_received = 0usize;

    for _ in 0..iterations {
        let args = serde_json::json!({
            "size": response_size,
            "data": &request_data
        });

        let sent_size = serde_json::to_string(&args).map(|s| s.len()).unwrap_or(0);

        if let Ok((elapsed, received_size)) =
            call_tool_timed(&client, "large_response", args).await
        {
            histogram.record(elapsed);
            total_sent += sent_size;
            total_received += received_size;
        }
    }

    server.shutdown().await;

    let summary = histogram.summary();
    let avg_sent = if summary.count > 0 {
        total_sent as f64 / summary.count as f64
    } else {
        0.0
    };
    let avg_received = if summary.count > 0 {
        total_received as f64 / summary.count as f64
    } else {
        0.0
    };

    PerfMetrics::new("Bidirectional Large Payload")
        .with_config(format!(
            "request_size={}, response_size={}, iterations={}",
            request_size, response_size, iterations
        ))
        .with_latency(summary)
        .with_custom("request_size_bytes", request_size as f64)
        .with_custom("response_size_bytes", response_size as f64)
        .with_custom("avg_sent_bytes", avg_sent)
        .with_custom("avg_received_bytes", avg_received)
        .with_custom("total_bytes_transferred", (total_sent + total_received) as f64)
}

/// Test memory pressure under large payloads
pub async fn test_memory_pressure_large_payloads(config: &PerfConfig) -> PerfMetrics {
    let server = MockMcpServer::start(Duration::ZERO).await;
    let url = format!("http://{}/mcp", server.addr);

    let memory_tracker = MemoryTracker::new();
    let counter = Arc::new(ThroughputCounter::new());
    let running = Arc::new(AtomicBool::new(true));

    let concurrency = config.effective_clients();
    let duration = if config.verify_mode {
        Duration::from_secs(2)
    } else {
        Duration::from_secs(30)
    };
    let payload_size = config.large_payload_size;

    let start = Instant::now();
    counter.start();

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
                            "name": "large_response",
                            "arguments": {"size": payload_size}
                        },
                        "id": client_id
                    }))
                    .send()
                    .await;

                match result {
                    Ok(resp) => {
                        if let Ok(bytes) = resp.bytes().await {
                            counter.record_request(true, 200, bytes.len());
                        } else {
                            counter.record_request(false, 200, 0);
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

    // Memory sampling
    let memory_samples = Arc::new(tokio::sync::RwLock::new(Vec::new()));
    let memory_samples_clone = memory_samples.clone();

    let memory_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        while start.elapsed() < duration {
            interval.tick().await;
            memory_tracker.record(start).await;
            memory_samples_clone.write().await.push(memory_tracker.summary());
        }
        memory_tracker.summary()
    });

    tokio::time::sleep(duration).await;

    running.store(false, Ordering::Relaxed);
    tokio::time::sleep(Duration::from_millis(100)).await;

    for handle in handles {
        handle.abort();
    }

    counter.stop();

    let memory_summary = memory_task.await.unwrap_or_default();
    server.shutdown().await;

    let throughput = counter.summary();
    let total_data_transferred_mb = throughput.bytes_received as f64 / 1_048_576.0;

    PerfMetrics::new("Memory Pressure Large Payloads")
        .with_config(format!(
            "duration={:?}, concurrency={}, payload_size={}",
            duration, concurrency, payload_size
        ))
        .with_throughput(throughput)
        .with_memory(memory_summary)
        .with_custom("payload_size_bytes", payload_size as f64)
        .with_custom("total_data_transferred_mb", total_data_transferred_mb)
}

/// Test throughput at different payload sizes
pub async fn test_throughput_vs_payload_size(config: &PerfConfig) -> Vec<PerfMetrics> {
    let server = MockMcpServer::start(Duration::ZERO).await;
    let url = format!("http://{}/mcp", server.addr);

    let payload_sizes = if config.verify_mode {
        vec![("small", config.small_payload_size), ("medium", config.medium_payload_size)]
    } else {
        vec![
            ("small", config.small_payload_size),
            ("medium", config.medium_payload_size),
            ("large", config.large_payload_size),
        ]
    };

    let concurrency = config.effective_clients();
    let duration = if config.verify_mode {
        Duration::from_millis(500)
    } else {
        Duration::from_secs(10)
    };

    let mut results = Vec::new();

    for (name, size) in payload_sizes {
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
                                "name": "large_response",
                                "arguments": {"size": size}
                            },
                            "id": client_id
                        }))
                        .send()
                        .await;

                    match result {
                        Ok(resp) => {
                            if let Ok(bytes) = resp.bytes().await {
                                histogram.record(req_start.elapsed());
                                counter.record_request(true, 200, bytes.len());
                            } else {
                                counter.record_request(false, 200, 0);
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
        tokio::time::sleep(Duration::from_millis(50)).await;

        for handle in handles {
            handle.abort();
        }

        counter.stop();

        let throughput = counter.summary();
        let effective_throughput_mbps = throughput.megabits_per_second_received;

        results.push(
            PerfMetrics::new(format!("Throughput - {} Payload", name))
                .with_config(format!(
                    "size={}, duration={:?}, concurrency={}",
                    size, duration, concurrency
                ))
                .with_latency(histogram.summary())
                .with_throughput(throughput)
                .with_custom("payload_size_bytes", size as f64)
                .with_custom("effective_throughput_mbps", effective_throughput_mbps),
        );

        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    server.shutdown().await;

    results
}

/// Generate a request payload of the specified size
fn generate_request_payload(size: usize) -> serde_json::Value {
    // Create a payload that's approximately the requested size
    let data_size = size.saturating_sub(50); // Account for JSON overhead
    let data = "x".repeat(data_size);
    serde_json::json!({
        "data": data,
        "size": size
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_payload_request_sizes() {
        let config = PerfConfig::verification_config();
        let results = test_request_payload_sizes(&config).await;
        assert!(!results.is_empty());
        for r in &results {
            verify_test_basics(r).expect("Request payload test should pass verification");
        }
    }

    #[tokio::test]
    async fn test_payload_response_sizes() {
        let config = PerfConfig::verification_config();
        let results = test_response_payload_sizes(&config).await;
        assert!(!results.is_empty());
    }

    #[test]
    fn test_generate_payload() {
        let payload = generate_request_payload(1000);
        let serialized = serde_json::to_string(&payload).unwrap();
        // Should be approximately 1000 bytes
        assert!(serialized.len() >= 900 && serialized.len() <= 1100);
    }
}
