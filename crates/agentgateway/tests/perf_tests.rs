//! Performance Test Suite Entry Point
//!
//! This module serves as the main entry point for running performance tests.
//! Tests can be run in verification mode (fast, minimal iterations) or
//! full mode (comprehensive, production-like parameters).
//!
//! # Running Tests
//!
//! ```bash
//! # Verification mode (fast, for CI/sandboxed environments)
//! PERF_VERIFY=1 cargo test --test perf_tests --release -- --nocapture
//!
//! # Full performance tests
//! cargo test --test perf_tests --release -- --nocapture
//!
//! # Specific test category
//! cargo test --test perf_tests --release baseline -- --nocapture
//! cargo test --test perf_tests --release virtual_tools -- --nocapture
//! cargo test --test perf_tests --release streaming -- --nocapture
//! cargo test --test perf_tests --release failover -- --nocapture
//! cargo test --test perf_tests --release load -- --nocapture
//! cargo test --test perf_tests --release stability -- --nocapture
//! cargo test --test perf_tests --release payload -- --nocapture
//!
//! # With custom configuration
//! PERF_ITERATIONS=5000 PERF_CLIENTS=50 cargo test --test perf_tests --release -- --nocapture
//!
//! # Output results to JSON
//! PERF_JSON_OUTPUT=1 PERF_OUTPUT_PATH=results.json cargo test --test perf_tests --release
//! ```

mod perf;

use perf::config::PerfConfig;
use perf::metrics::PerfReport;

/// Run all performance tests and generate a comprehensive report
#[tokio::test]
async fn all_perf_tests() {
    let config = PerfConfig::from_env();
    let mut report = PerfReport::new();

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║          AgentGateway Performance Test Suite                 ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║ Mode: {:56} ║",
        if config.verify_mode {
            "VERIFICATION (minimal iterations)"
        } else {
            "FULL (comprehensive testing)"
        }
    );
    println!(
        "║ Iterations: {:50} ║",
        config.effective_iterations()
    );
    println!(
        "║ Concurrent Clients: {:42} ║",
        config.effective_clients()
    );
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Run baseline tests
    let baseline_report = perf::baseline::run_baseline_tests(&config).await;
    for r in baseline_report.results {
        report.add(r);
    }

    // Run virtual tools tests
    let vt_report = perf::virtual_tools::run_virtual_tools_tests(&config).await;
    for r in vt_report.results {
        report.add(r);
    }

    // Run streaming tests
    let streaming_report = perf::streaming::run_streaming_tests(&config).await;
    for r in streaming_report.results {
        report.add(r);
    }

    // Run failover tests
    let failover_report = perf::failover::run_failover_tests(&config).await;
    for r in failover_report.results {
        report.add(r);
    }

    // Run load tests
    let load_report = perf::load::run_load_tests(&config).await;
    for r in load_report.results {
        report.add(r);
    }

    // Run stability tests (shorter in verify mode)
    let stability_report = perf::stability::run_stability_tests(&config).await;
    for r in stability_report.results {
        report.add(r);
    }

    // Run payload tests
    let payload_report = perf::payload::run_payload_tests(&config).await;
    for r in payload_report.results {
        report.add(r);
    }

    // Output summary
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    Test Suite Complete                       ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Total Tests: {:49} ║", report.results.len());
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Save results if configured
    if config.json_output {
        if let Some(ref path) = config.output_path {
            if let Err(e) = report.save(path) {
                eprintln!("Failed to save results to {}: {}", path, e);
            } else {
                println!("Results saved to: {}", path);
            }
        }
    }

    // In verification mode, ensure all tests ran successfully
    if config.verify_mode {
        assert!(
            !report.results.is_empty(),
            "No test results collected in verification mode"
        );
        println!("✓ All verification tests passed");
    }
}

/// Run only baseline latency tests
#[tokio::test]
async fn baseline_tests() {
    let config = PerfConfig::from_env();
    let report = perf::baseline::run_baseline_tests(&config).await;

    if config.json_output {
        println!("{}", report.to_json());
    }
}

/// Run only virtual tools overhead tests
#[tokio::test]
async fn virtual_tools_tests() {
    let config = PerfConfig::from_env();
    let report = perf::virtual_tools::run_virtual_tools_tests(&config).await;

    if config.json_output {
        println!("{}", report.to_json());
    }
}

/// Run only streaming performance tests
#[tokio::test]
async fn streaming_tests() {
    let config = PerfConfig::from_env();
    let report = perf::streaming::run_streaming_tests(&config).await;

    if config.json_output {
        println!("{}", report.to_json());
    }
}

/// Run only failover tests
#[tokio::test]
async fn failover_tests() {
    let config = PerfConfig::from_env();
    let report = perf::failover::run_failover_tests(&config).await;

    if config.json_output {
        println!("{}", report.to_json());
    }
}

/// Run only load/throughput tests
#[tokio::test]
async fn load_tests() {
    let config = PerfConfig::from_env();
    let report = perf::load::run_load_tests(&config).await;

    if config.json_output {
        println!("{}", report.to_json());
    }
}

/// Run only stability tests
#[tokio::test]
async fn stability_tests() {
    let config = PerfConfig::from_env();
    let report = perf::stability::run_stability_tests(&config).await;

    if config.json_output {
        println!("{}", report.to_json());
    }
}

/// Run only payload size tests
#[tokio::test]
async fn payload_tests() {
    let config = PerfConfig::from_env();
    let report = perf::payload::run_payload_tests(&config).await;

    if config.json_output {
        println!("{}", report.to_json());
    }
}

/// Quick smoke test to verify test infrastructure works
#[tokio::test]
async fn perf_smoke_test() {
    // Force verification mode for smoke test
    // SAFETY: This is a test, and we're setting an env var before any threading
    unsafe { std::env::set_var("PERF_VERIFY", "1"); }
    let config = PerfConfig::verification_config();

    // Just run a single baseline test
    let metrics = perf::baseline::test_direct_server_latency(&config).await;

    // Verify basic sanity
    assert!(metrics.latency.is_some(), "Should have latency data");
    let lat = metrics.latency.unwrap();
    assert!(lat.count > 0, "Should have recorded samples");
    assert!(lat.p50 <= lat.p99, "p50 should be <= p99");

    println!("✓ Smoke test passed");
}
