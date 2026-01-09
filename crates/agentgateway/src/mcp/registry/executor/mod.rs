// Composition Executor Module
//
// Executes tool compositions at runtime, handling:
// - Pattern execution (pipeline, scatter-gather, filter, schema-map, map-each)
// - Tool invocation via backend pool
// - Result aggregation and transformation
// - Tracing and observability

mod context;
mod filter;
mod map_each;
mod pipeline;
mod scatter_gather;
mod schema_map;

pub use context::ExecutionContext;
pub use filter::FilterExecutor;
pub use map_each::MapEachExecutor;
pub use pipeline::PipelineExecutor;
pub use scatter_gather::ScatterGatherExecutor;
pub use schema_map::SchemaMapExecutor;

use std::sync::Arc;

use serde_json::Value;
use thiserror::Error;

use super::compiled::{CompiledComposition, CompiledRegistry, CompiledTool};
use super::patterns::PatternSpec;

/// Errors that can occur during composition execution
#[derive(Error, Debug)]
pub enum ExecutionError {
	#[error("tool not found: {0}")]
	ToolNotFound(String),

	#[error("tool execution failed: {0}")]
	ToolExecutionFailed(String),

	#[error("pattern execution failed: {0}")]
	PatternExecutionFailed(String),

	#[error("invalid input: {0}")]
	InvalidInput(String),

	#[error("timeout after {0}ms")]
	Timeout(u32),

	#[error("all scatter-gather targets failed")]
	AllTargetsFailed,

	#[error("JSONPath evaluation failed: {0}")]
	JsonPathError(String),

	#[error("predicate evaluation failed: {0}")]
	PredicateError(String),

	#[error("type error: expected {expected}, got {actual}")]
	TypeError { expected: String, actual: String },

	#[error("internal error: {0}")]
	Internal(String),
}

/// Composition executor - executes tool compositions
pub struct CompositionExecutor {
	/// Compiled registry for tool lookups
	registry: Arc<CompiledRegistry>,
	/// Tool invocation callback
	tool_invoker: Arc<dyn ToolInvoker>,
}

/// Trait for invoking tools (abstraction over actual backend calls)
#[async_trait::async_trait]
pub trait ToolInvoker: Send + Sync {
	/// Invoke a tool by name with the given arguments
	async fn invoke(&self, tool_name: &str, args: Value) -> Result<Value, ExecutionError>;
}

impl CompositionExecutor {
	/// Create a new composition executor
	pub fn new(registry: Arc<CompiledRegistry>, tool_invoker: Arc<dyn ToolInvoker>) -> Self {
		Self { registry, tool_invoker }
	}

	/// Execute a composition by name
	pub async fn execute(&self, composition_name: &str, input: Value) -> Result<Value, ExecutionError> {
		let tool = self
			.registry
			.get_tool(composition_name)
			.ok_or_else(|| ExecutionError::ToolNotFound(composition_name.to_string()))?;

		let composition = tool
			.composition_info()
			.ok_or_else(|| ExecutionError::InvalidInput(format!("{} is not a composition", composition_name)))?;

		self.execute_composition(tool, composition, input).await
	}

	/// Execute a compiled composition
	async fn execute_composition(
		&self,
		_tool: &CompiledTool,
		composition: &CompiledComposition,
		input: Value,
	) -> Result<Value, ExecutionError> {
		let ctx = ExecutionContext::new(input.clone(), self.registry.clone(), self.tool_invoker.clone());

		let result = self.execute_pattern(&composition.spec, input, &ctx).await?;

		// Apply output transform if present
		if let Some(ref transform) = composition.output_transform {
			transform.apply(&result).map_err(|e| ExecutionError::PatternExecutionFailed(e.to_string()))
		} else {
			Ok(result)
		}
	}

	/// Execute a pattern
	pub async fn execute_pattern(
		&self,
		spec: &PatternSpec,
		input: Value,
		ctx: &ExecutionContext,
	) -> Result<Value, ExecutionError> {
		match spec {
			PatternSpec::Pipeline(p) => PipelineExecutor::execute(p, input, ctx, self).await,
			PatternSpec::ScatterGather(sg) => ScatterGatherExecutor::execute(sg, input, ctx, self).await,
			PatternSpec::Filter(f) => FilterExecutor::execute(f, input).await,
			PatternSpec::SchemaMap(sm) => SchemaMapExecutor::execute(sm, input).await,
			PatternSpec::MapEach(me) => MapEachExecutor::execute(me, input, ctx, self).await,
		}
	}

	/// Execute a tool by name
	pub async fn execute_tool(&self, name: &str, args: Value, ctx: &ExecutionContext) -> Result<Value, ExecutionError> {
		// First, check if it's a composition in the registry
		if let Some(tool) = self.registry.get_tool(name) {
			if let Some(composition) = tool.composition_info() {
				return self.execute_composition(tool, composition, args).await;
			}
		}

		// Otherwise, invoke via the tool invoker
		ctx.tool_invoker.invoke(name, args).await
	}
}

/// Mock tool invoker for testing
#[cfg(test)]
pub struct MockToolInvoker {
	responses: std::sync::Mutex<std::collections::HashMap<String, Value>>,
}

#[cfg(test)]
impl MockToolInvoker {
	pub fn new() -> Self {
		Self { responses: std::sync::Mutex::new(std::collections::HashMap::new()) }
	}

	pub fn with_response(self, tool_name: &str, response: Value) -> Self {
		self.responses.lock().unwrap().insert(tool_name.to_string(), response);
		self
	}
}

#[cfg(test)]
#[async_trait::async_trait]
impl ToolInvoker for MockToolInvoker {
	async fn invoke(&self, tool_name: &str, _args: Value) -> Result<Value, ExecutionError> {
		self.responses
			.lock()
			.unwrap()
			.get(tool_name)
			.cloned()
			.ok_or_else(|| ExecutionError::ToolNotFound(tool_name.to_string()))
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::mcp::registry::patterns::{PipelineSpec, PipelineStep, StepOperation, ToolCall};
	use crate::mcp::registry::types::{Registry, ToolDefinition, ToolImplementation};

	#[tokio::test]
	async fn test_execute_simple_composition() {
		// Create a simple pipeline composition
		let composition = ToolDefinition::composition(
			"test_pipeline",
			PatternSpec::Pipeline(PipelineSpec {
				steps: vec![PipelineStep {
					id: "step1".to_string(),
					operation: StepOperation::Tool(ToolCall { name: "echo".to_string() }),
					input: None,
				}],
			}),
		);

		let registry = Registry::with_tool_definitions(vec![composition]);
		let compiled = CompiledRegistry::compile(registry).unwrap();

		let invoker = MockToolInvoker::new().with_response("echo", serde_json::json!({"echoed": true}));

		let executor = CompositionExecutor::new(Arc::new(compiled), Arc::new(invoker));

		let result = executor.execute("test_pipeline", serde_json::json!({"input": "test"})).await;

		assert!(result.is_ok());
		assert_eq!(result.unwrap()["echoed"], true);
	}

	#[tokio::test]
	async fn test_execute_nonexistent_composition() {
		let registry = Registry::new();
		let compiled = CompiledRegistry::compile(registry).unwrap();
		let invoker = MockToolInvoker::new();

		let executor = CompositionExecutor::new(Arc::new(compiled), Arc::new(invoker));

		let result = executor.execute("nonexistent", serde_json::json!({})).await;

		assert!(result.is_err());
		assert!(matches!(result.unwrap_err(), ExecutionError::ToolNotFound(_)));
	}
}

