// Pattern type definitions for tool compositions
//
// These types correspond to the registry.proto schema and are used
// for deserializing composition definitions from JSON.

mod filter;
mod map_each;
mod pipeline;
mod scatter_gather;
mod schema_map;

pub use filter::{FieldPredicate, FilterSpec, PredicateValue};
pub use map_each::{MapEachInner, MapEachSpec};
pub use pipeline::{DataBinding, InputBinding, PipelineSpec, PipelineStep, StepBinding, StepOperation, ToolCall};
pub use scatter_gather::{
	AggregationOp, AggregationStrategy, DedupeOp, LimitOp, ScatterGatherSpec, ScatterTarget, SortOp,
};
pub use schema_map::{CoalesceSource, ConcatSource, FieldSource, LiteralValue, SchemaMapSpec, TemplateSource};

use serde::{Deserialize, Serialize};

/// PatternSpec defines a composition pattern
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum PatternSpec {
	/// Sequential execution of steps
	Pipeline(PipelineSpec),

	/// Parallel fan-out with aggregation
	ScatterGather(ScatterGatherSpec),

	/// Filter array elements by predicate
	Filter(FilterSpec),

	/// Transform fields using mappings
	SchemaMap(SchemaMapSpec),

	/// Apply operation to each array element
	MapEach(MapEachSpec),
}

impl PatternSpec {
	/// Get the names of tools referenced by this pattern
	pub fn referenced_tools(&self) -> Vec<&str> {
		match self {
			PatternSpec::Pipeline(p) => p.referenced_tools(),
			PatternSpec::ScatterGather(sg) => sg.referenced_tools(),
			PatternSpec::Filter(_) => vec![],
			PatternSpec::SchemaMap(_) => vec![],
			PatternSpec::MapEach(me) => me.referenced_tools(),
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_parse_pipeline_pattern() {
		let json = r#"{
			"pipeline": {
				"steps": [
					{
						"id": "step1",
						"operation": { "tool": { "name": "search" } },
						"input": { "input": { "path": "$" } }
					}
				]
			}
		}"#;

		let spec: PatternSpec = serde_json::from_str(json).unwrap();
		assert!(matches!(spec, PatternSpec::Pipeline(_)));
	}

	#[test]
	fn test_parse_scatter_gather_pattern() {
		let json = r#"{
			"scatterGather": {
				"targets": [
					{ "tool": "search1" },
					{ "tool": "search2" }
				],
				"aggregation": {
					"ops": [
						{ "flatten": true }
					]
				}
			}
		}"#;

		let spec: PatternSpec = serde_json::from_str(json).unwrap();
		assert!(matches!(spec, PatternSpec::ScatterGather(_)));
	}

	#[test]
	fn test_parse_filter_pattern() {
		let json = r#"{
			"filter": {
				"predicate": {
					"field": "$.score",
					"op": "gt",
					"value": { "numberValue": 0.5 }
				}
			}
		}"#;

		let spec: PatternSpec = serde_json::from_str(json).unwrap();
		assert!(matches!(spec, PatternSpec::Filter(_)));
	}

	#[test]
	fn test_parse_schema_map_pattern() {
		let json = r#"{
			"schemaMap": {
				"mappings": {
					"title": { "path": "$.name" },
					"source": { "literal": { "stringValue": "web" } }
				}
			}
		}"#;

		let spec: PatternSpec = serde_json::from_str(json).unwrap();
		assert!(matches!(spec, PatternSpec::SchemaMap(_)));
	}

	#[test]
	fn test_parse_map_each_pattern() {
		let json = r#"{
			"mapEach": {
				"inner": { "tool": "fetch" }
			}
		}"#;

		let spec: PatternSpec = serde_json::from_str(json).unwrap();
		assert!(matches!(spec, PatternSpec::MapEach(_)));
	}
}

