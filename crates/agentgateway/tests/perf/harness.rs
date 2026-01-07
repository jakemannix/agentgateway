//! Performance test harness
//!
//! Provides common infrastructure for running performance tests, including:
//! - Gateway setup and teardown
//! - Mock MCP server management
//! - Client connection pooling
//! - Coordinated test execution

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rmcp::model::{CallToolRequestParam, InitializeRequestParam};
use rmcp::service::RunningService;
use rmcp::{RoleClient, ServiceExt};
use rmcp::transport::StreamableHttpClientTransport;
use tokio::sync::Semaphore;

use super::config::PerfConfig;
use super::metrics::{CpuMonitor, LatencyHistogram, MemoryTracker, PerfMetrics, ThroughputCounter};

/// Performance test harness
pub struct PerfTestHarness {
    pub config: PerfConfig,
    pub gateway_addr: Option<SocketAddr>,
    mock_servers: Vec<MockMcpServer>,
    cpu_monitor: CpuMonitor,
    memory_tracker: MemoryTracker,
}

impl PerfTestHarness {
    /// Create a new test harness with the given configuration
    pub fn new(config: PerfConfig) -> Self {
        Self {
            config,
            gateway_addr: None,
            mock_servers: Vec::new(),
            cpu_monitor: CpuMonitor::new(),
            memory_tracker: MemoryTracker::new(),
        }
    }

    /// Create harness with default configuration (checks PERF_VERIFY env)
    pub fn default_config() -> Self {
        Self::new(PerfConfig::from_env())
    }

    /// Start CPU monitoring
    pub fn start_cpu_monitoring(&self) -> tokio::task::JoinHandle<()> {
        self.cpu_monitor.start(self.config.cpu_sample_interval)
    }

    /// Stop CPU monitoring and get summary
    pub async fn stop_cpu_monitoring(&self) -> super::metrics::CpuSummary {
        self.cpu_monitor.stop();
        tokio::time::sleep(Duration::from_millis(100)).await;
        self.cpu_monitor.summary().await
    }

    /// Record memory sample
    pub async fn record_memory(&self, start: Instant) {
        self.memory_tracker.record(start).await;
    }

    /// Get memory summary
    pub fn memory_summary(&self) -> super::metrics::MemorySummary {
        self.memory_tracker.summary()
    }

    /// Create a mock MCP server that responds with configurable latency
    pub async fn create_mock_server(&mut self, response_delay: Duration) -> MockMcpServer {
        let server = MockMcpServer::start(response_delay).await;
        self.mock_servers.push(server.clone());
        server
    }

    /// Create multiple mock servers
    pub async fn create_mock_servers(
        &mut self,
        count: usize,
        response_delay: Duration,
    ) -> Vec<MockMcpServer> {
        let mut servers = Vec::with_capacity(count);
        for _ in 0..count {
            servers.push(self.create_mock_server(response_delay).await);
        }
        servers
    }

    /// Run a latency benchmark
    pub async fn run_latency_benchmark<F, Fut>(
        &self,
        name: &str,
        iterations: usize,
        warmup: usize,
        f: F,
    ) -> PerfMetrics
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<(), anyhow::Error>>,
    {
        let histogram = LatencyHistogram::new();

        // Warmup phase
        for _ in 0..warmup {
            let _ = f().await;
        }

        // Measurement phase
        for _ in 0..iterations {
            let start = Instant::now();
            if f().await.is_ok() {
                histogram.record(start.elapsed());
            }
        }

        PerfMetrics::new(name)
            .with_config(format!(
                "iterations={}, warmup={}",
                iterations, warmup
            ))
            .with_latency(histogram.summary())
    }

    /// Run a throughput benchmark with concurrent clients
    pub async fn run_throughput_benchmark<F, Fut>(
        &self,
        name: &str,
        duration: Duration,
        concurrency: usize,
        f: F,
    ) -> PerfMetrics
    where
        F: Fn(usize) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<(usize, usize), anyhow::Error>> + Send,
    {
        let counter = Arc::new(ThroughputCounter::new());
        let histogram = Arc::new(LatencyHistogram::new());
        let f = Arc::new(f);

        // Start CPU monitoring
        let cpu_handle = self.start_cpu_monitoring();

        counter.start();
        let start = Instant::now();
        let deadline = start + duration;

        let semaphore = Arc::new(Semaphore::new(concurrency));
        let mut handles = Vec::new();

        let running = Arc::new(std::sync::atomic::AtomicBool::new(true));

        // Spawn worker tasks
        for client_id in 0..concurrency {
            let sem = semaphore.clone();
            let counter = counter.clone();
            let histogram = histogram.clone();
            let f = f.clone();
            let running = running.clone();

            let handle = tokio::spawn(async move {
                while running.load(std::sync::atomic::Ordering::Relaxed) {
                    let _permit = sem.acquire().await.unwrap();
                    let req_start = Instant::now();

                    match f(client_id).await {
                        Ok((sent, recv)) => {
                            histogram.record(req_start.elapsed());
                            counter.record_request(true, sent, recv);
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
        running.store(false, std::sync::atomic::Ordering::Relaxed);

        // Give workers time to finish current requests
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Abort any remaining tasks
        for handle in handles {
            handle.abort();
        }

        counter.stop();

        // Get CPU summary
        let cpu_summary = self.stop_cpu_monitoring().await;
        cpu_handle.abort();

        PerfMetrics::new(name)
            .with_config(format!(
                "duration={:?}, concurrency={}",
                duration, concurrency
            ))
            .with_latency(histogram.summary())
            .with_throughput(counter.summary())
            .with_cpu(cpu_summary)
    }

    /// Run a comparison benchmark (e.g., with vs without virtual tools)
    pub async fn run_comparison_benchmark<F1, F2, Fut1, Fut2>(
        &self,
        name: &str,
        baseline: F1,
        comparison: F2,
    ) -> (PerfMetrics, PerfMetrics, f64)
    where
        F1: Fn() -> Fut1,
        F2: Fn() -> Fut2,
        Fut1: std::future::Future<Output = Result<(), anyhow::Error>>,
        Fut2: std::future::Future<Output = Result<(), anyhow::Error>>,
    {
        let iterations = self.config.effective_iterations();
        let warmup = self.config.effective_warmup();

        let baseline_metrics = self
            .run_latency_benchmark(&format!("{} (baseline)", name), iterations, warmup, baseline)
            .await;

        let comparison_metrics = self
            .run_latency_benchmark(
                &format!("{} (with feature)", name),
                iterations,
                warmup,
                comparison,
            )
            .await;

        // Calculate overhead percentage
        let baseline_p50 = baseline_metrics
            .latency
            .as_ref()
            .map(|l| l.p50.as_micros() as f64)
            .unwrap_or(1.0);
        let comparison_p50 = comparison_metrics
            .latency
            .as_ref()
            .map(|l| l.p50.as_micros() as f64)
            .unwrap_or(1.0);

        let overhead_percent = ((comparison_p50 - baseline_p50) / baseline_p50) * 100.0;

        (baseline_metrics, comparison_metrics, overhead_percent)
    }

    /// Shutdown all mock servers
    pub async fn shutdown(&mut self) {
        for server in self.mock_servers.drain(..) {
            server.shutdown().await;
        }
    }
}

impl Drop for PerfTestHarness {
    fn drop(&mut self) {
        self.cpu_monitor.stop();
    }
}

/// Mock MCP server for performance testing
#[derive(Clone)]
pub struct MockMcpServer {
    pub addr: SocketAddr,
    response_delay: Duration,
    shutdown_tx: Arc<tokio::sync::Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
}

impl MockMcpServer {
    /// Start a mock MCP server with configurable response delay
    pub async fn start(response_delay: Duration) -> Self {
        use rmcp::transport::StreamableHttpServerConfig;
        use rmcp::transport::streamable_http_server::StreamableHttpService;
        use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;

        let service = StreamableHttpService::new(
            move || Ok(MockMcpHandler::new(response_delay)),
            LocalSessionManager::default().into(),
            StreamableHttpServerConfig {
                sse_keep_alive: None,
                stateful_mode: true,
                cancellation_token: Default::default(),
            },
        );

        let (tx, rx) = tokio::sync::oneshot::channel();
        let router = axum::Router::new().nest_service("/mcp", service);
        let tcp_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = tcp_listener.local_addr().unwrap();

        tokio::spawn(async move {
            let _ = axum::serve(tcp_listener, router)
                .with_graceful_shutdown(async { rx.await.unwrap_or(()) })
                .await;
        });

        // Brief delay to ensure server is ready
        tokio::time::sleep(Duration::from_millis(10)).await;

        Self {
            addr,
            response_delay,
            shutdown_tx: Arc::new(tokio::sync::Mutex::new(Some(tx))),
        }
    }

    /// Get the server address
    pub fn address(&self) -> SocketAddr {
        self.addr
    }

    /// Shutdown the server
    pub async fn shutdown(&self) {
        if let Some(tx) = self.shutdown_tx.lock().await.take() {
            let _ = tx.send(());
        }
    }
}

/// Mock MCP handler for performance testing
#[derive(Clone)]
pub struct MockMcpHandler {
    response_delay: Duration,
}

impl MockMcpHandler {
    pub fn new(response_delay: Duration) -> Self {
        Self { response_delay }
    }

    async fn handle_echo(&self, args: serde_json::Value) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        if !self.response_delay.is_zero() {
            tokio::time::sleep(self.response_delay).await;
        }
        Ok(rmcp::model::CallToolResult::success(vec![
            rmcp::model::Content::text(serde_json::to_string(&args).unwrap_or_default()),
        ]))
    }

    async fn handle_large_response(&self, args: serde_json::Value) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        if !self.response_delay.is_zero() {
            tokio::time::sleep(self.response_delay).await;
        }
        let size = args.get("size").and_then(|v| v.as_u64()).unwrap_or(1024) as usize;
        let response = "x".repeat(size);
        Ok(rmcp::model::CallToolResult::success(vec![
            rmcp::model::Content::text(response),
        ]))
    }

    async fn handle_complex_tool(&self, args: serde_json::Value) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        if !self.response_delay.is_zero() {
            tokio::time::sleep(self.response_delay).await;
        }

        let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("unknown");
        let api_version = args.get("api_version").and_then(|v| v.as_str()).unwrap_or("v1");
        let debug = args.get("debug").and_then(|v| v.as_bool()).unwrap_or(false);

        let response = serde_json::json!({
            "data": {
                "query": query,
                "result": format!("Result for: {}", query),
                "metadata": {
                    "api_version": api_version,
                    "debug": debug,
                }
            }
        });

        Ok(rmcp::model::CallToolResult::success(vec![
            rmcp::model::Content::text(serde_json::to_string(&response).unwrap_or_default()),
        ]))
    }

    fn get_tools() -> Vec<rmcp::model::Tool> {
        vec![
            rmcp::model::Tool {
                name: "echo".into(),
                description: Some("Echo tool for performance testing".into()),
                input_schema: serde_json::from_str(r#"{"type": "object"}"#).unwrap(),
                annotations: None,
                icons: None,
                meta: None,
                output_schema: None,
                title: None,
            },
            rmcp::model::Tool {
                name: "large_response".into(),
                description: Some("Tool that returns large payloads".into()),
                input_schema: serde_json::from_str(r#"{"type": "object", "properties": {"size": {"type": "integer"}}}"#).unwrap(),
                annotations: None,
                icons: None,
                meta: None,
                output_schema: None,
                title: None,
            },
            rmcp::model::Tool {
                name: "complex_tool".into(),
                description: Some("Tool with complex schema for virtual tool testing".into()),
                input_schema: serde_json::from_str(r#"{"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}"#).unwrap(),
                annotations: None,
                icons: None,
                meta: None,
                output_schema: None,
                title: None,
            },
        ]
    }
}

impl rmcp::ServerHandler for MockMcpHandler {
    fn get_info(&self) -> rmcp::model::ServerInfo {
        rmcp::model::ServerInfo {
            protocol_version: rmcp::model::ProtocolVersion::V_2025_06_18,
            capabilities: rmcp::model::ServerCapabilities::builder()
                .enable_tools()
                .build(),
            server_info: rmcp::model::Implementation {
                name: "perf-test-server".to_string(),
                version: "1.0.0".to_string(),
                title: None,
                website_url: None,
                icons: None,
            },
            instructions: None,
        }
    }

    async fn list_tools(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParam>,
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::ListToolsResult, rmcp::ErrorData> {
        Ok(rmcp::model::ListToolsResult {
            tools: Self::get_tools(),
            next_cursor: None,
            meta: None,
        })
    }

    async fn call_tool(
        &self,
        request: rmcp::model::CallToolRequestParam,
        _context: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        let args = serde_json::Value::Object(request.arguments.unwrap_or_default());
        let name: &str = &request.name;
        match name {
            "echo" => self.handle_echo(args).await,
            "large_response" => self.handle_large_response(args).await,
            "complex_tool" => self.handle_complex_tool(args).await,
            _ => Err(rmcp::ErrorData::invalid_request("Unknown tool", None)),
        }
    }

    async fn initialize(
        &self,
        _request: rmcp::model::InitializeRequestParam,
        _: rmcp::service::RequestContext<rmcp::RoleServer>,
    ) -> Result<rmcp::model::InitializeResult, rmcp::ErrorData> {
        Ok(self.get_info())
    }
}

/// Create an MCP client connected to the given address
pub async fn create_mcp_client(
    addr: SocketAddr,
) -> Result<RunningService<RoleClient, InitializeRequestParam>, anyhow::Error> {
    let transport =
        StreamableHttpClientTransport::<reqwest::Client>::from_uri(format!("http://{}/mcp", addr));

    let client_info = rmcp::model::ClientInfo {
        protocol_version: Default::default(),
        capabilities: rmcp::model::ClientCapabilities::default(),
        client_info: rmcp::model::Implementation {
            name: "perf-test-client".to_string(),
            version: "1.0.0".to_string(),
            title: None,
            website_url: None,
            icons: None,
        },
    };

    Ok(client_info.serve(transport).await?)
}

/// Helper to call an MCP tool and measure the operation
pub async fn call_tool_timed(
    client: &RunningService<RoleClient, InitializeRequestParam>,
    tool_name: &str,
    args: serde_json::Value,
) -> Result<(Duration, usize), anyhow::Error> {
    let start = Instant::now();
    let result = client
        .call_tool(CallToolRequestParam {
            name: tool_name.to_string().into(),
            arguments: args.as_object().cloned(),
        })
        .await?;

    let elapsed = start.elapsed();
    let response_size = result
        .content
        .iter()
        .map(|c| {
            c.raw
                .as_text()
                .map(|t| t.text.len())
                .unwrap_or(0)
        })
        .sum();

    Ok((elapsed, response_size))
}

/// Generate a payload of the specified size
pub fn generate_payload(size: usize) -> serde_json::Value {
    let data = "x".repeat(size);
    serde_json::json!({ "data": data })
}

/// Helper to verify tests work by checking basic assertions
pub fn verify_test_basics(metrics: &PerfMetrics) -> Result<(), String> {
    if let Some(ref lat) = metrics.latency {
        if lat.count == 0 {
            return Err("No latency samples recorded".to_string());
        }
        if lat.p50 > lat.p99 {
            return Err("p50 should not be greater than p99".to_string());
        }
        if lat.min > lat.max {
            return Err("min should not be greater than max".to_string());
        }
    }

    if let Some(ref tp) = metrics.throughput {
        if tp.total_requests == 0 {
            return Err("No requests recorded".to_string());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_server_creation() {
        let server = MockMcpServer::start(Duration::ZERO).await;
        assert!(server.addr.port() > 0);
        server.shutdown().await;
    }

    #[tokio::test]
    async fn test_harness_creation() {
        let harness = PerfTestHarness::new(PerfConfig::verification_config());
        assert!(harness.config.verify_mode);
    }

    #[test]
    fn test_generate_payload() {
        let payload = generate_payload(100);
        let data = payload["data"].as_str().unwrap();
        assert_eq!(data.len(), 100);
    }
}
