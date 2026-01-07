//! HTTP Streaming Performance Tests
//!
//! These tests measure the performance characteristics of HTTP streaming
//! and MCP interactions, which is critical for:
//! - Server-Sent Events (SSE) for MCP notifications
//! - Streaming LLM responses
//! - Real-time data feeds
//!
//! Key metrics:
//! - Time to first byte (TTFB)
//! - Chunk delivery latency
//! - Throughput under streaming load

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::config::PerfConfig;
use super::harness::{MockMcpServer, verify_test_basics};
use super::metrics::{LatencyHistogram, PerfMetrics, PerfReport, ThroughputCounter};

/// Run all streaming performance tests
pub async fn run_streaming_tests(config: &PerfConfig) -> PerfReport {
    let mut report = PerfReport::new();

    println!("\n=== HTTP Streaming Performance Tests ===\n");

    // Test 1: Time to first byte
    let ttfb_metrics = test_time_to_first_byte(config).await;
    ttfb_metrics.print_report();
    verify_test_basics(&ttfb_metrics).expect("TTFB test failed validation");
    report.add(ttfb_metrics);

    // Test 2: Chunk delivery latency (via large responses)
    let chunk_metrics = test_chunk_delivery_latency(config).await;
    chunk_metrics.print_report();
    report.add(chunk_metrics);

    // Test 3: Streaming throughput
    let throughput_metrics = test_streaming_throughput(config).await;
    throughput_metrics.print_report();
    report.add(throughput_metrics);

    // Test 4: Multiple concurrent streams
    let concurrent_metrics = test_concurrent_streams(config).await;
    concurrent_metrics.print_report();
    report.add(concurrent_metrics);

    report
}

/// Test time to first byte for MCP requests
pub async fn test_time_to_first_byte(config: &PerfConfig) -> PerfMetrics {
    let server = MockMcpServer::start(Duration::ZERO).await;

    let ttfb_histogram = LatencyHistogram::new();
    let iterations = config.effective_iterations();
    let warmup = config.effective_warmup();

    // Use reqwest for HTTP-level measurements
    let client = reqwest::Client::new();
    let url = format!("http://{}/mcp", server.addr);

    // Warmup
    for _ in 0..warmup {
        let _ = client
            .post(&url)
            .json(&serde_json::json!({
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "echo",
                    "arguments": {"test": "warmup"}
                },
                "id": 1
            }))
            .send()
            .await;
    }

    // Measure TTFB
    for i in 0..iterations {
        let start = Instant::now();

        let result = client
            .post(&url)
            .json(&serde_json::json!({
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "echo",
                    "arguments": {"iteration": i}
                },
                "id": i
            }))
            .send()
            .await;

        if result.is_ok() {
            // TTFB is time until headers received
            ttfb_histogram.record(start.elapsed());
        }
    }

    server.shutdown().await;

    PerfMetrics::new("Time to First Byte (TTFB)")
        .with_config(format!("iterations={}", iterations))
        .with_latency(ttfb_histogram.summary())
}

/// Test latency between chunks in streaming responses
pub async fn test_chunk_delivery_latency(config: &PerfConfig) -> PerfMetrics {
    // For this test, we measure response delivery for large payloads
    let server = MockMcpServer::start(Duration::ZERO).await;

    let chunk_histogram = LatencyHistogram::new();
    let iterations = config.effective_iterations();

    let client = reqwest::Client::new();
    let url = format!("http://{}/mcp", server.addr);

    for i in 0..iterations {
        // Request a larger response
        let response_size = config.medium_payload_size;

        let start = Instant::now();
        let result = client
            .post(&url)
            .json(&serde_json::json!({
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "large_response",
                    "arguments": {"size": response_size}
                },
                "id": i
            }))
            .send()
            .await;

        if let Ok(resp) = result {
            // Consume all bytes and measure total time
            if let Ok(bytes) = resp.bytes().await {
                let elapsed = start.elapsed();
                // Calculate per-chunk latency assuming typical 8KB chunks
                let num_chunks = (bytes.len() / 8192).max(1);
                let per_chunk = elapsed / num_chunks as u32;
                chunk_histogram.record(per_chunk);
            }
        }
    }

    server.shutdown().await;

    PerfMetrics::new("Chunk Delivery Latency")
        .with_config(format!(
            "iterations={}, payload_size={}",
            iterations, config.medium_payload_size
        ))
        .with_latency(chunk_histogram.summary())
}

/// Test streaming throughput
pub async fn test_streaming_throughput(config: &PerfConfig) -> PerfMetrics {
    let server = MockMcpServer::start(Duration::ZERO).await;

    let counter = ThroughputCounter::new();
    let duration = config.load_duration;
    let client = reqwest::Client::new();
    let url = format!("http://{}/mcp", server.addr);

    counter.start();
    let start = Instant::now();
    let deadline = start + duration;

    while Instant::now() < deadline {
        let response_size = config.large_payload_size;

        let result = client
            .post(&url)
            .json(&serde_json::json!({
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": "large_response",
                    "arguments": {"size": response_size}
                },
                "id": 1
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

    counter.stop();
    server.shutdown().await;

    PerfMetrics::new("Streaming Throughput")
        .with_config(format!(
            "duration={:?}, payload_size={}",
            duration, config.large_payload_size
        ))
        .with_throughput(counter.summary())
}

/// Test concurrent streaming connections
pub async fn test_concurrent_streams(config: &PerfConfig) -> PerfMetrics {
    let server = MockMcpServer::start(Duration::ZERO).await;

    let counter = Arc::new(ThroughputCounter::new());
    let histogram = Arc::new(LatencyHistogram::new());
    let concurrency = config.effective_clients();
    let duration = config.load_duration;
    let url = format!("http://{}/mcp", server.addr);

    counter.start();
    let start = Instant::now();
    let deadline = start + duration;
    let running = Arc::new(AtomicBool::new(true));

    let mut handles = Vec::new();

    for client_id in 0..concurrency {
        let counter = counter.clone();
        let histogram = histogram.clone();
        let running = running.clone();
        let response_size = config.medium_payload_size;
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
                            "arguments": {"size": response_size}
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

    // Wait for duration
    tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)).await;

    // Signal workers to stop
    running.store(false, Ordering::Relaxed);
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Abort remaining tasks
    for handle in handles {
        handle.abort();
    }

    counter.stop();
    server.shutdown().await;

    PerfMetrics::new("Concurrent Streams")
        .with_config(format!(
            "duration={:?}, concurrency={}",
            duration, concurrency
        ))
        .with_latency(histogram.summary())
        .with_throughput(counter.summary())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_streaming_ttfb() {
        let config = PerfConfig::verification_config();
        let metrics = test_time_to_first_byte(&config).await;
        verify_test_basics(&metrics).expect("TTFB test should pass verification");
    }

    #[tokio::test]
    async fn test_streaming_throughput_basic() {
        let config = PerfConfig::verification_config();
        let metrics = test_streaming_throughput(&config).await;
        // In verification mode, just check it completes
        assert!(metrics.throughput.is_some());
    }
}
