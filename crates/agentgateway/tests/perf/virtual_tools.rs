//! Virtual Tools Overhead Tests
//!
//! These tests measure the performance overhead of virtual tool features:
//! - Tool renaming/description changes
//! - Default argument injection
//! - Schema field hiding/projection
//! - Output schema transformation (JSON parsing for strongly-typed outputs)
//!
//! Understanding these overheads helps teams make informed decisions about
//! which virtual tool features to enable for their use cases.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use agentgateway::mcp::registry::{
    CompiledRegistry, OutputField, OutputSchema, Registry, VirtualToolDef,
};

use super::config::PerfConfig;
use super::harness::{MockMcpServer, call_tool_timed, create_mcp_client, verify_test_basics};
use super::metrics::{LatencyHistogram, PerfMetrics, PerfReport};

/// Run all virtual tools performance tests
pub async fn run_virtual_tools_tests(config: &PerfConfig) -> PerfReport {
    let mut report = PerfReport::new();

    println!("\n=== Virtual Tools Overhead Tests ===\n");

    // Test 1: Direct MCP baseline (without any virtual tools)
    let baseline_metrics = test_direct_mcp_baseline(config).await;
    baseline_metrics.print_report();
    verify_test_basics(&baseline_metrics).expect("Baseline test failed validation");
    report.add(baseline_metrics);

    // Test 2: Registry compilation overhead
    let compile_metrics = test_registry_compilation_overhead(config).await;
    compile_metrics.print_report();
    report.add(compile_metrics);

    // Test 3: Default argument injection (compile-time)
    let defaults_metrics = test_default_injection_compile_time(config).await;
    defaults_metrics.print_report();
    report.add(defaults_metrics);

    // Test 4: Output transformation overhead (compile-time)
    let output_metrics = test_output_transformation_compile_time(config).await;
    output_metrics.print_report();
    report.add(output_metrics);

    // Test 5: Combined virtual tool overhead (compile-time)
    let combined_metrics = test_combined_virtual_tool_compile_time(config).await;
    combined_metrics.print_report();
    report.add(combined_metrics);

    // Test 6: prepare_call_args overhead at runtime
    let runtime_metrics = test_prepare_call_args_overhead(config).await;
    runtime_metrics.print_report();
    report.add(runtime_metrics);

    report
}

/// Test direct MCP server baseline without any virtual tools
pub async fn test_direct_mcp_baseline(config: &PerfConfig) -> PerfMetrics {
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

    PerfMetrics::new("Direct MCP Baseline (No Virtual Tools)")
        .with_config(format!("iterations={}, warmup={}", iterations, warmup))
        .with_latency(histogram.summary())
}

/// Test registry compilation overhead with varying registry sizes
pub async fn test_registry_compilation_overhead(config: &PerfConfig) -> PerfMetrics {
    let iterations = config.effective_iterations();

    // Small registry (10 tools)
    let small_compile_times = measure_compilation_time(10, iterations);

    // Medium registry (100 tools)
    let medium_compile_times = measure_compilation_time(100, iterations);

    // Large registry (1000 tools) - only if not in verify mode
    let large_compile_times = if config.verify_mode {
        measure_compilation_time(50, iterations)
    } else {
        measure_compilation_time(1000, iterations)
    };

    PerfMetrics::new("Registry Compilation Overhead")
        .with_config(format!("iterations={}", iterations))
        .with_custom("small_10_tools_mean_us", small_compile_times.0)
        .with_custom("small_10_tools_p99_us", small_compile_times.1)
        .with_custom("medium_100_tools_mean_us", medium_compile_times.0)
        .with_custom("medium_100_tools_p99_us", medium_compile_times.1)
        .with_custom("large_tools_mean_us", large_compile_times.0)
        .with_custom("large_tools_p99_us", large_compile_times.1)
}

/// Measure compilation time for a registry with N tools
fn measure_compilation_time(num_tools: usize, iterations: usize) -> (f64, f64) {
    let histogram = LatencyHistogram::new();

    for i in 0..iterations {
        let tools: Vec<VirtualToolDef> = (0..num_tools)
            .map(|j| {
                VirtualToolDef::new(
                    format!("tool_{}_{}", i, j),
                    "backend",
                    format!("original_tool_{}", j),
                )
                .with_description(format!("Test tool {} for iteration {}", j, i))
                .with_default("default_key", serde_json::json!("default_value"))
            })
            .collect();

        let registry = Registry::with_tools(tools);

        let start = Instant::now();
        let _ = CompiledRegistry::compile(registry);
        histogram.record(start.elapsed());
    }

    let summary = histogram.summary();
    (
        summary.mean.as_micros() as f64,
        summary.p99.as_micros() as f64,
    )
}

/// Test overhead of default argument injection at compile time
pub async fn test_default_injection_compile_time(config: &PerfConfig) -> PerfMetrics {
    let iterations = config.effective_iterations();

    // Without defaults
    let without_defaults_times = {
        let histogram = LatencyHistogram::new();
        for i in 0..iterations {
            let tools = vec![VirtualToolDef::new(
                format!("tool_{}", i),
                "backend",
                "original_tool",
            )];
            let registry = Registry::with_tools(tools);
            let start = Instant::now();
            let _ = CompiledRegistry::compile(registry);
            histogram.record(start.elapsed());
        }
        histogram.summary()
    };

    // With 5 defaults
    let with_defaults_times = {
        let histogram = LatencyHistogram::new();
        for i in 0..iterations {
            let tools = vec![VirtualToolDef::new(
                format!("tool_{}", i),
                "backend",
                "original_tool",
            )
            .with_default("api_version", serde_json::json!("v2"))
            .with_default("debug", serde_json::json!(false))
            .with_default("format", serde_json::json!("json"))
            .with_default("timeout", serde_json::json!(30))
            .with_default("retry_count", serde_json::json!(3))];
            let registry = Registry::with_tools(tools);
            let start = Instant::now();
            let _ = CompiledRegistry::compile(registry);
            histogram.record(start.elapsed());
        }
        histogram.summary()
    };

    let without_p50_us = without_defaults_times.p50.as_micros() as f64;
    let with_p50_us = with_defaults_times.p50.as_micros() as f64;
    let overhead_us = with_p50_us - without_p50_us;
    let overhead_pct = if without_p50_us > 0.0 {
        (overhead_us / without_p50_us) * 100.0
    } else {
        0.0
    };

    PerfMetrics::new("Default Injection Compile-time Overhead")
        .with_config(format!("iterations={}, num_defaults=5", iterations))
        .with_latency(with_defaults_times)
        .with_custom("without_defaults_p50_us", without_p50_us)
        .with_custom("with_defaults_p50_us", with_p50_us)
        .with_custom("overhead_us", overhead_us)
        .with_custom("overhead_percent", overhead_pct)
}

/// Test overhead of output transformation at compile time
pub async fn test_output_transformation_compile_time(config: &PerfConfig) -> PerfMetrics {
    let iterations = config.effective_iterations();

    // Without output schema
    let without_output_times = {
        let histogram = LatencyHistogram::new();
        for i in 0..iterations {
            let tools = vec![VirtualToolDef::new(
                format!("tool_{}", i),
                "backend",
                "original_tool",
            )];
            let registry = Registry::with_tools(tools);
            let start = Instant::now();
            let _ = CompiledRegistry::compile(registry);
            histogram.record(start.elapsed());
        }
        histogram.summary()
    };

    // With output schema (4 fields with JSONPath)
    let with_output_times = {
        let histogram = LatencyHistogram::new();
        for i in 0..iterations {
            let mut properties = HashMap::new();
            properties.insert("result".to_string(), OutputField::new("string", "$.data.result"));
            properties.insert("query".to_string(), OutputField::new("string", "$.data.query"));
            properties.insert(
                "api_version".to_string(),
                OutputField::new("string", "$.data.metadata.api_version"),
            );
            properties.insert(
                "debug".to_string(),
                OutputField::new("boolean", "$.data.metadata.debug"),
            );

            let tools = vec![VirtualToolDef::new(
                format!("tool_{}", i),
                "backend",
                "original_tool",
            )
            .with_output_schema(OutputSchema::new(properties))];
            let registry = Registry::with_tools(tools);
            let start = Instant::now();
            let _ = CompiledRegistry::compile(registry);
            histogram.record(start.elapsed());
        }
        histogram.summary()
    };

    let without_p50_us = without_output_times.p50.as_micros() as f64;
    let with_p50_us = with_output_times.p50.as_micros() as f64;
    let overhead_us = with_p50_us - without_p50_us;
    let overhead_pct = if without_p50_us > 0.0 {
        (overhead_us / without_p50_us) * 100.0
    } else {
        0.0
    };

    PerfMetrics::new("Output Transformation Compile-time Overhead")
        .with_config(format!("iterations={}, output_fields=4", iterations))
        .with_latency(with_output_times)
        .with_custom("without_output_p50_us", without_p50_us)
        .with_custom("with_output_p50_us", with_p50_us)
        .with_custom("overhead_us", overhead_us)
        .with_custom("overhead_percent", overhead_pct)
}

/// Test combined overhead of all virtual tool features at compile time
pub async fn test_combined_virtual_tool_compile_time(config: &PerfConfig) -> PerfMetrics {
    let iterations = config.effective_iterations();

    // Baseline (no features)
    let baseline_times = {
        let histogram = LatencyHistogram::new();
        for i in 0..iterations {
            let tools = vec![VirtualToolDef::new(
                format!("tool_{}", i),
                "backend",
                "original_tool",
            )];
            let registry = Registry::with_tools(tools);
            let start = Instant::now();
            let _ = CompiledRegistry::compile(registry);
            histogram.record(start.elapsed());
        }
        histogram.summary()
    };

    // Full virtual tool (all features)
    let combined_times = {
        let histogram = LatencyHistogram::new();
        for i in 0..iterations {
            let mut properties = HashMap::new();
            properties.insert("result".to_string(), OutputField::new("string", "$.data.result"));
            properties.insert("query".to_string(), OutputField::new("string", "$.data.query"));

            let tools = vec![VirtualToolDef::new(
                format!("my_search_tool_{}", i),
                "backend",
                "original_tool",
            )
            .with_description(
                "A fully customized search tool with defaults, hidden fields, and output transformation",
            )
            .with_default("api_version", serde_json::json!("v2"))
            .with_default("debug", serde_json::json!(false))
            .with_default("format", serde_json::json!("json"))
            .with_hidden_fields(vec!["internal_id".to_string(), "trace_id".to_string()])
            .with_output_schema(OutputSchema::new(properties))];
            let registry = Registry::with_tools(tools);
            let start = Instant::now();
            let _ = CompiledRegistry::compile(registry);
            histogram.record(start.elapsed());
        }
        histogram.summary()
    };

    let baseline_p50_us = baseline_times.p50.as_micros() as f64;
    let combined_p50_us = combined_times.p50.as_micros() as f64;
    let overhead_us = combined_p50_us - baseline_p50_us;
    let overhead_pct = if baseline_p50_us > 0.0 {
        (overhead_us / baseline_p50_us) * 100.0
    } else {
        0.0
    };

    PerfMetrics::new("Combined Virtual Tool Compile-time Overhead")
        .with_config(format!(
            "iterations={}, features=rename+defaults+hide+output",
            iterations
        ))
        .with_latency(combined_times)
        .with_custom("baseline_p50_us", baseline_p50_us)
        .with_custom("combined_p50_us", combined_p50_us)
        .with_custom("overhead_us", overhead_us)
        .with_custom("overhead_percent", overhead_pct)
}

/// Test runtime overhead of prepare_call_args
pub async fn test_prepare_call_args_overhead(config: &PerfConfig) -> PerfMetrics {
    let iterations = config.effective_iterations();

    // Create a compiled registry with defaults
    let tools = vec![VirtualToolDef::new("my_tool", "backend", "original_tool")
        .with_default("api_version", serde_json::json!("v2"))
        .with_default("debug", serde_json::json!(false))
        .with_default("format", serde_json::json!("json"))
        .with_default("timeout", serde_json::json!(30))
        .with_default("retry_count", serde_json::json!(3))];
    let registry = Registry::with_tools(tools);
    let compiled = CompiledRegistry::compile(registry).expect("Failed to compile registry");

    let histogram = LatencyHistogram::new();

    // Measure prepare_call_args overhead
    for i in 0..iterations {
        let user_args = serde_json::json!({
            "query": format!("query_{}", i)
        });

        let start = Instant::now();
        let _ = compiled.prepare_call_args("my_tool", user_args);
        histogram.record(start.elapsed());
    }

    let summary = histogram.summary();

    PerfMetrics::new("prepare_call_args Runtime Overhead")
        .with_config(format!("iterations={}, num_defaults=5", iterations))
        .with_latency(summary.clone())
        .with_custom("mean_us", summary.mean.as_micros() as f64)
        .with_custom("p50_us", summary.p50.as_micros() as f64)
        .with_custom("p99_us", summary.p99.as_micros() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_virtual_tools_direct_baseline() {
        let config = PerfConfig::verification_config();
        let metrics = test_direct_mcp_baseline(&config).await;
        verify_test_basics(&metrics).expect("Direct baseline test should pass verification");
    }

    #[test]
    fn test_registry_compilation_baseline() {
        let (mean, p99) = measure_compilation_time(10, 5);
        assert!(mean > 0.0, "Mean compilation time should be positive");
        assert!(p99 >= mean, "p99 should be >= mean");
    }

    #[tokio::test]
    async fn test_prepare_call_args_runs() {
        let config = PerfConfig::verification_config();
        let metrics = test_prepare_call_args_overhead(&config).await;
        assert!(metrics.latency.is_some());
    }
}
