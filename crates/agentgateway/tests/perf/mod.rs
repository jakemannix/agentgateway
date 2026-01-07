//! Performance Test Suite for AgentGateway
//!
//! This module provides comprehensive performance testing for the AgentGateway,
//! covering various scenarios relevant to enterprise multi-agent systems.
//!
//! # Test Categories
//!
//! - **Baseline**: Measure baseline MCP latency through the gateway (p50/p95/p99)
//! - **Virtual Tools**: Measure overhead of virtual tool features (renaming, defaults, schema projection)
//! - **Streaming**: Measure HTTP streaming performance and latency characteristics
//! - **Failover**: Test multi-backend failover scenarios and reconnection behavior
//! - **Load**: Throughput tests with CPU utilization metrics
//! - **Stability**: Long-running stability and memory leak detection
//! - **Payload**: Large context window and response payload tests
//!
//! # Running Tests
//!
//! ```bash
//! # Run all perf tests (verification mode - limited iterations)
//! PERF_VERIFY=1 cargo test --test perf_tests --release
//!
//! # Run full perf tests
//! cargo test --test perf_tests --release -- --nocapture
//!
//! # Run specific category
//! cargo test --test perf_tests --release baseline -- --nocapture
//!
//! # Run with Docker for controlled hardware
//! ./perf/run-perf-tests.sh --cpus 4 --memory 8g
//! ```

pub mod config;
pub mod metrics;
pub mod harness;
pub mod baseline;
pub mod virtual_tools;
pub mod streaming;
pub mod failover;
pub mod load;
pub mod stability;
pub mod payload;
