//! Multi-Backend Failover Performance Tests
//!
//! These tests measure performance characteristics when:
//! - Multiple MCP servers are available
//! - Servers fail and require reconnection
//! - Load is distributed across servers
//!
//! Key scenarios for enterprise multi-agent systems:
//! - Backend failure detection latency
//! - Reconnection overhead
//! - Impact on ongoing requests during failover
//! - Recovery time to steady state

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::config::PerfConfig;
use super::harness::{MockMcpServer, call_tool_timed, create_mcp_client, verify_test_basics};
use super::metrics::{LatencyHistogram, PerfMetrics, PerfReport, ThroughputCounter};

/// Run all failover performance tests
pub async fn run_failover_tests(config: &PerfConfig) -> PerfReport {
    let mut report = PerfReport::new();

    println!("\n=== Multi-Backend Failover Tests ===\n");

    // Test 1: Multiple server baseline
    let multi_baseline = test_multi_server_baseline(config).await;
    multi_baseline.print_report();
    verify_test_basics(&multi_baseline).expect("Multi-server baseline test failed validation");
    report.add(multi_baseline);

    // Test 2: Server failure impact
    let failure_metrics = test_server_failure_impact(config).await;
    failure_metrics.print_report();
    report.add(failure_metrics);

    // Test 3: Recovery time
    let recovery_metrics = test_recovery_time(config).await;
    recovery_metrics.print_report();
    report.add(recovery_metrics);

    // Test 4: Continuous load during failover
    let continuous_metrics = test_continuous_load_failover(config).await;
    continuous_metrics.print_report();
    report.add(continuous_metrics);

    report
}

/// Test baseline performance with multiple MCP servers
pub async fn test_multi_server_baseline(config: &PerfConfig) -> PerfMetrics {
    let num_servers = config.num_backends.min(3);
    let mut servers = Vec::with_capacity(num_servers);
    let mut clients = Vec::with_capacity(num_servers);

    // Start multiple servers
    for _ in 0..num_servers {
        let server = MockMcpServer::start(Duration::ZERO).await;
        let client = create_mcp_client(server.addr).await.expect("Failed to create client");
        servers.push(server);
        clients.push(client);
    }

    let histogram = LatencyHistogram::new();
    let iterations = config.effective_iterations();
    let warmup = config.effective_warmup();

    // Warmup - round robin across servers
    for i in 0..warmup {
        let client = &clients[i % num_servers];
        let _ = call_tool_timed(client, "echo", serde_json::json!({"test": "warmup"})).await;
    }

    // Measure - distribute requests across servers
    for i in 0..iterations {
        let client = &clients[i % num_servers];
        let args = serde_json::json!({"iteration": i, "server": i % num_servers});
        if let Ok((elapsed, _)) = call_tool_timed(client, "echo", args).await {
            histogram.record(elapsed);
        }
    }

    // Shutdown servers
    for server in servers {
        server.shutdown().await;
    }

    PerfMetrics::new("Multi-Server Baseline")
        .with_config(format!(
            "iterations={}, servers={}",
            iterations, num_servers
        ))
        .with_latency(histogram.summary())
        .with_custom("num_servers", num_servers as f64)
}

/// Test impact of a server failure on request latency
pub async fn test_server_failure_impact(config: &PerfConfig) -> PerfMetrics {
    let num_servers = 2;
    let mut servers = Vec::with_capacity(num_servers);
    let mut clients = Vec::with_capacity(num_servers);

    for _ in 0..num_servers {
        let server = MockMcpServer::start(Duration::ZERO).await;
        let client = create_mcp_client(server.addr).await.expect("Failed to create client");
        servers.push(server);
        clients.push(client);
    }

    let before_failure_histogram = LatencyHistogram::new();
    let during_failure_histogram = LatencyHistogram::new();
    let iterations = config.effective_iterations() / 3;

    // Phase 1: Measure before failure
    for i in 0..iterations {
        let client = &clients[i % num_servers];
        let args = serde_json::json!({"phase": "before", "iteration": i});
        if let Ok((elapsed, _)) = call_tool_timed(client, "echo", args).await {
            before_failure_histogram.record(elapsed);
        }
    }

    // Phase 2: Kill one server and measure with remaining
    servers[0].shutdown().await;
    tokio::time::sleep(Duration::from_millis(50)).await;

    let mut failures_during = 0u64;
    for i in 0..iterations {
        // Only use remaining server
        let args = serde_json::json!({"phase": "during", "iteration": i});
        match call_tool_timed(&clients[1], "echo", args).await {
            Ok((elapsed, _)) => {
                during_failure_histogram.record(elapsed);
            }
            Err(_) => {
                failures_during += 1;
            }
        }
    }

    // Shutdown remaining server
    servers[1].shutdown().await;

    let before_summary = before_failure_histogram.summary();
    let during_summary = during_failure_histogram.summary();

    let before_p50_us = before_summary.p50.as_micros() as f64;
    let during_p50_us = during_summary.p50.as_micros() as f64;
    let latency_increase_pct = if before_p50_us > 0.0 {
        ((during_p50_us - before_p50_us) / before_p50_us) * 100.0
    } else {
        0.0
    };

    PerfMetrics::new("Server Failure Impact")
        .with_config(format!("iterations_per_phase={}", iterations))
        .with_latency(during_summary)
        .with_custom("before_failure_p50_us", before_p50_us)
        .with_custom("during_failure_p50_us", during_p50_us)
        .with_custom("latency_increase_percent", latency_increase_pct)
        .with_custom("failures_during_failover", failures_during as f64)
        .with_custom(
            "failure_rate_percent",
            (failures_during as f64 / iterations as f64) * 100.0,
        )
}

/// Test recovery time after server comes back
pub async fn test_recovery_time(config: &PerfConfig) -> PerfMetrics {
    let server = MockMcpServer::start(Duration::ZERO).await;
    let server_addr = server.addr;

    // Create initial client and verify it works
    let client = create_mcp_client(server_addr).await.expect("Failed to create client");
    let _ = call_tool_timed(&client, "echo", serde_json::json!({"test": "initial"})).await;

    // Simulate server restart
    server.shutdown().await;
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Start new server (simulating recovery)
    let new_server = MockMcpServer::start(Duration::ZERO).await;

    // Measure time to successful reconnection
    let reconnect_start = Instant::now();
    let mut reconnect_attempts = 0;
    let mut reconnect_time = Duration::ZERO;

    loop {
        reconnect_attempts += 1;
        if reconnect_attempts > 100 {
            break;
        }

        match create_mcp_client(new_server.addr).await {
            Ok(new_client) => {
                if call_tool_timed(&new_client, "echo", serde_json::json!({"test": "recovery"}))
                    .await
                    .is_ok()
                {
                    reconnect_time = reconnect_start.elapsed();
                    break;
                }
            }
            Err(_) => {}
        }

        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    new_server.shutdown().await;

    PerfMetrics::new("Recovery Time")
        .with_config("server_restart_simulation")
        .with_custom("reconnect_time_ms", reconnect_time.as_millis() as f64)
        .with_custom("reconnect_attempts", reconnect_attempts as f64)
}

/// Test continuous load during failover scenario
pub async fn test_continuous_load_failover(config: &PerfConfig) -> PerfMetrics {
    let num_servers = 2;
    let mut servers = Vec::with_capacity(num_servers);

    for _ in 0..num_servers {
        servers.push(MockMcpServer::start(Duration::ZERO).await);
    }

    let counter = Arc::new(ThroughputCounter::new());
    let histogram = Arc::new(LatencyHistogram::new());
    let running = Arc::new(AtomicBool::new(true));
    let failover_triggered = Arc::new(AtomicBool::new(false));

    let concurrency = config.effective_clients();
    let duration = config.load_duration;
    let failure_delay = duration / 3; // Trigger failure at 1/3 of duration

    counter.start();

    // Collect server URLs
    let server_urls: Vec<String> = servers
        .iter()
        .map(|s| format!("http://{}/mcp", s.addr))
        .collect();

    // Spawn worker tasks
    let mut handles = Vec::new();
    for client_id in 0..concurrency {
        let counter = counter.clone();
        let histogram = histogram.clone();
        let running = running.clone();
        let failover_triggered = failover_triggered.clone();
        let server_urls = server_urls.clone();

        let handle = tokio::spawn(async move {
            let client = reqwest::Client::new();

            while running.load(Ordering::Relaxed) {
                let req_start = Instant::now();

                // Use second server if failover triggered
                let url = if failover_triggered.load(Ordering::Relaxed) {
                    &server_urls[1]
                } else {
                    &server_urls[client_id % server_urls.len()]
                };

                let result = client
                    .post(url)
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
                        if resp.status().is_success() {
                            histogram.record(req_start.elapsed());
                            counter.record_request(true, 100, 100);
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

    // Schedule failover
    let servers_arc = Arc::new(tokio::sync::Mutex::new(servers));
    let servers_clone = servers_arc.clone();
    let failover_triggered_clone = failover_triggered.clone();

    tokio::spawn(async move {
        tokio::time::sleep(failure_delay).await;
        let mut servers = servers_clone.lock().await;
        if !servers.is_empty() {
            servers[0].shutdown().await;
            failover_triggered_clone.store(true, Ordering::Relaxed);
        }
    });

    // Wait for duration
    tokio::time::sleep(duration).await;

    // Signal workers to stop
    running.store(false, Ordering::Relaxed);
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Abort remaining tasks
    for handle in handles {
        handle.abort();
    }

    counter.stop();

    // Shutdown remaining servers
    let mut servers = servers_arc.lock().await;
    for server in servers.drain(..) {
        server.shutdown().await;
    }

    let throughput = counter.summary();
    let success_rate = if throughput.total_requests > 0 {
        (throughput.successful_requests as f64 / throughput.total_requests as f64) * 100.0
    } else {
        0.0
    };

    PerfMetrics::new("Continuous Load During Failover")
        .with_config(format!(
            "duration={:?}, concurrency={}, failover_at={:?}",
            duration, concurrency, failure_delay
        ))
        .with_latency(histogram.summary())
        .with_throughput(throughput)
        .with_custom("success_rate_percent", success_rate)
        .with_custom(
            "failover_triggered",
            if failover_triggered.load(Ordering::Relaxed) {
                1.0
            } else {
                0.0
            },
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_failover_multi_server_baseline() {
        let config = PerfConfig::verification_config();
        let metrics = test_multi_server_baseline(&config).await;
        verify_test_basics(&metrics).expect("Multi-server baseline should pass verification");
    }

    #[tokio::test]
    async fn test_failover_recovery_time() {
        let config = PerfConfig::verification_config();
        let metrics = test_recovery_time(&config).await;
        assert!(
            metrics.custom_metrics.contains_key("reconnect_time_ms"),
            "Should track reconnect time"
        );
    }
}
