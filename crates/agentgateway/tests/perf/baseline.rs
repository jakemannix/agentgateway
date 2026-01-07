//! Baseline MCP latency tests
//!
//! These tests measure the baseline latency for MCP operations, establishing
//! performance baselines that can be compared against gateway-proxied scenarios.
//!
//! Key metrics:
//! - Direct server latency (no gateway)
//! - MCP tool call latency distribution (p50/p95/p99)

use std::time::Duration;

use super::config::PerfConfig;
use super::harness::{MockMcpServer, call_tool_timed, create_mcp_client, verify_test_basics};
use super::metrics::{LatencyHistogram, PerfMetrics, PerfReport};

/// Run all baseline performance tests
pub async fn run_baseline_tests(config: &PerfConfig) -> PerfReport {
    let mut report = PerfReport::new();

    println!("\n=== Baseline MCP Latency Tests ===\n");

    // Test 1: Direct server latency (no gateway)
    let direct_metrics = test_direct_server_latency(config).await;
    direct_metrics.print_report();
    verify_test_basics(&direct_metrics).expect("Direct server test failed validation");
    report.add(direct_metrics);

    // Test 2: Tool call with varying argument sizes
    let args_metrics = test_argument_size_impact(config).await;
    for m in &args_metrics {
        m.print_report();
    }
    for m in args_metrics {
        report.add(m);
    }

    // Test 3: Backend response delay simulation
    let delay_metrics = test_backend_delay_impact(config).await;
    for m in &delay_metrics {
        m.print_report();
    }
    for m in delay_metrics {
        report.add(m);
    }

    // Test 4: Connection reuse vs new connections
    let conn_metrics = test_connection_reuse(config).await;
    conn_metrics.print_report();
    report.add(conn_metrics);

    report
}

/// Test direct MCP server latency (baseline without gateway)
pub async fn test_direct_server_latency(config: &PerfConfig) -> PerfMetrics {
    let server = MockMcpServer::start(Duration::ZERO).await;
    let client = create_mcp_client(server.addr).await.expect("Failed to create client");

    let histogram = LatencyHistogram::new();
    let iterations = config.effective_iterations();
    let warmup = config.effective_warmup();

    // Warmup
    for _ in 0..warmup {
        let _ = call_tool_timed(&client, "echo", serde_json::json!({"test": "warmup"})).await;
    }

    // Measure
    for i in 0..iterations {
        let args = serde_json::json!({"iteration": i, "data": "test"});
        if let Ok((elapsed, _)) = call_tool_timed(&client, "echo", args).await {
            histogram.record(elapsed);
        }
    }

    server.shutdown().await;

    PerfMetrics::new("Direct MCP Server Latency")
        .with_config(format!("iterations={}, warmup={}", iterations, warmup))
        .with_latency(histogram.summary())
}

/// Test impact of argument size on latency
pub async fn test_argument_size_impact(config: &PerfConfig) -> Vec<PerfMetrics> {
    let server = MockMcpServer::start(Duration::ZERO).await;
    let client = create_mcp_client(server.addr).await.expect("Failed to create client");

    let sizes = if config.verify_mode {
        vec![("tiny", 10), ("small", 100)]
    } else {
        vec![
            ("tiny", 10),
            ("small", 100),
            ("medium", 1000),
            ("large", 10000),
        ]
    };

    let iterations = config.effective_iterations();
    let warmup = config.effective_warmup();
    let mut results = Vec::new();

    for (name, size) in sizes {
        let histogram = LatencyHistogram::new();
        let data = "x".repeat(size);

        // Warmup
        for _ in 0..warmup {
            let _ = call_tool_timed(&client, "echo", serde_json::json!({"data": &data})).await;
        }

        // Measure
        for _ in 0..iterations {
            let args = serde_json::json!({"data": &data});
            if let Ok((elapsed, _)) = call_tool_timed(&client, "echo", args).await {
                histogram.record(elapsed);
            }
        }

        results.push(
            PerfMetrics::new(format!("Argument Size - {}", name))
                .with_config(format!("size={} bytes, iterations={}", size, iterations))
                .with_latency(histogram.summary())
                .with_custom("argument_size_bytes", size as f64),
        );
    }

    server.shutdown().await;
    results
}

/// Test impact of backend processing delay on latency
pub async fn test_backend_delay_impact(config: &PerfConfig) -> Vec<PerfMetrics> {
    let delays = if config.verify_mode {
        vec![
            ("no_delay", Duration::ZERO),
            ("1ms", Duration::from_millis(1)),
        ]
    } else {
        vec![
            ("no_delay", Duration::ZERO),
            ("1ms", Duration::from_millis(1)),
            ("5ms", Duration::from_millis(5)),
            ("10ms", Duration::from_millis(10)),
            ("50ms", Duration::from_millis(50)),
        ]
    };

    let iterations = config.effective_iterations();
    let warmup = config.effective_warmup();
    let mut results = Vec::new();

    for (name, delay) in delays {
        let server = MockMcpServer::start(delay).await;
        let client = create_mcp_client(server.addr).await.expect("Failed to create client");

        let histogram = LatencyHistogram::new();

        // Warmup
        for _ in 0..warmup {
            let _ = call_tool_timed(&client, "echo", serde_json::json!({"test": "warmup"})).await;
        }

        // Measure
        for i in 0..iterations {
            let args = serde_json::json!({"iteration": i});
            if let Ok((elapsed, _)) = call_tool_timed(&client, "echo", args).await {
                histogram.record(elapsed);
            }
        }

        server.shutdown().await;

        let summary = histogram.summary();
        let overhead = summary.p50.saturating_sub(delay);

        results.push(
            PerfMetrics::new(format!("Backend Delay - {}", name))
                .with_config(format!(
                    "delay={:?}, iterations={}",
                    delay, iterations
                ))
                .with_latency(summary)
                .with_custom("configured_delay_us", delay.as_micros() as f64)
                .with_custom("overhead_us", overhead.as_micros() as f64),
        );
    }

    results
}

/// Test connection reuse vs creating new connections
pub async fn test_connection_reuse(config: &PerfConfig) -> PerfMetrics {
    let server = MockMcpServer::start(Duration::ZERO).await;
    let iterations = config.effective_iterations();
    let warmup = config.effective_warmup();

    // Test with single connection (reuse)
    let reuse_histogram = LatencyHistogram::new();
    {
        let client = create_mcp_client(server.addr).await.expect("Failed to create client");

        // Warmup
        for _ in 0..warmup {
            let _ = call_tool_timed(&client, "echo", serde_json::json!({"test": "warmup"})).await;
        }

        // Measure with reused connection
        for i in 0..iterations {
            let args = serde_json::json!({"iteration": i});
            if let Ok((elapsed, _)) = call_tool_timed(&client, "echo", args).await {
                reuse_histogram.record(elapsed);
            }
        }
    }

    // Test with new connections each time (limited iterations)
    let new_conn_histogram = LatencyHistogram::new();
    let new_conn_iterations = (iterations / 10).max(5); // Fewer iterations since this is slow

    for i in 0..new_conn_iterations {
        let client = create_mcp_client(server.addr).await.expect("Failed to create client");
        let args = serde_json::json!({"iteration": i});
        if let Ok((elapsed, _)) = call_tool_timed(&client, "echo", args).await {
            new_conn_histogram.record(elapsed);
        }
    }

    server.shutdown().await;

    let reuse_summary = reuse_histogram.summary();
    let new_conn_summary = new_conn_histogram.summary();

    let connection_overhead = new_conn_summary
        .p50
        .saturating_sub(reuse_summary.p50);

    PerfMetrics::new("Connection Reuse Impact")
        .with_config(format!(
            "reuse_iterations={}, new_conn_iterations={}",
            iterations, new_conn_iterations
        ))
        .with_latency(reuse_summary.clone())
        .with_custom("reused_p50_us", reuse_summary.p50.as_micros() as f64)
        .with_custom("new_conn_p50_us", new_conn_summary.p50.as_micros() as f64)
        .with_custom(
            "connection_overhead_us",
            connection_overhead.as_micros() as f64,
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_baseline_direct_latency() {
        let config = PerfConfig::verification_config();
        let metrics = test_direct_server_latency(&config).await;
        verify_test_basics(&metrics).expect("Direct latency test should pass verification");
        assert!(metrics.latency.is_some());
    }

    #[tokio::test]
    async fn test_baseline_argument_sizes() {
        let config = PerfConfig::verification_config();
        let results = test_argument_size_impact(&config).await;
        assert!(!results.is_empty());
    }

    #[tokio::test]
    async fn test_baseline_backend_delays() {
        let config = PerfConfig::verification_config();
        let results = test_backend_delay_impact(&config).await;
        assert!(!results.is_empty());
    }
}
