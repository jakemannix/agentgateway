# AgentGateway Performance Test Suite

Comprehensive performance testing for AgentGateway, measuring latency, throughput, and resource utilization across various scenarios relevant to enterprise multi-agent systems.

## Test Categories

### 1. Baseline Tests (`baseline`)
Measures the fundamental gateway overhead for MCP requests:
- Direct server latency (no gateway)
- Gateway passthrough latency
- Gateway overhead percentage (p50, p95, p99)
- Stateful vs stateless proxy modes

### 2. Virtual Tools Tests (`virtual_tools`)
Measures overhead of virtual tool features:
- Tool renaming/description changes
- Default argument injection
- Schema field hiding/projection
- Output transformation with JSONPath
- Registry compilation time

### 3. Streaming Tests (`streaming`)
Measures HTTP streaming performance:
- Time to first byte (TTFB)
- Chunk delivery latency
- Streaming throughput
- Concurrent streams handling

### 4. Failover Tests (`failover`)
Tests multi-backend scenarios:
- Multi-backend baseline performance
- Backend failure impact on latency
- Recovery time after backend restart
- Continuous load during failover events

### 5. Load Tests (`load`)
Throughput and scalability testing:
- Maximum sustainable throughput
- Latency under increasing load
- CPU utilization curve
- Scalability with client count
- Burst load handling

### 6. Stability Tests (`stability`)
Long-running stability verification:
- Memory leak detection
- Performance consistency over time
- Connection pool stability
- Error rate stability

### 7. Payload Tests (`payload`)
Large payload handling:
- Request payload size impact
- Response payload size impact
- Bidirectional large payloads
- Memory pressure under large payloads
- Throughput vs payload size

## Running Tests

### Quick Start

```bash
# Verification mode (minimal iterations, ~1-2 minutes)
PERF_VERIFY=1 cargo test --test perf_tests --release -- --nocapture

# Full performance tests
cargo test --test perf_tests --release -- --nocapture

# Specific test category
cargo test --test perf_tests --release baseline -- --nocapture
```

### Using the Runner Script

```bash
# Quick verification
./perf/run-perf-tests.sh --verify

# Custom hardware constraints
./perf/run-perf-tests.sh --cpus 4 --memory 8g

# Run specific category locally
./perf/run-perf-tests.sh --test baseline --local

# Full production tests in Docker
./perf/run-perf-tests.sh --production
```

### Using Docker Compose

```bash
# Standard tests (4 CPUs, 8GB RAM)
docker-compose -f perf/docker-compose.perf.yml up

# CI configuration
docker-compose -f perf/docker-compose.perf.yml --profile ci up

# Production configuration (8 CPUs, 16GB RAM)
docker-compose -f perf/docker-compose.perf.yml --profile production up

# Resource-constrained (1 CPU, 1GB RAM)
docker-compose -f perf/docker-compose.perf.yml --profile constrained up
```

## Configuration

Tests are configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PERF_VERIFY` | `0` | Enable verification mode (minimal iterations) |
| `PERF_ITERATIONS` | `1000` | Number of measurement iterations |
| `PERF_WARMUP` | `100` | Number of warmup iterations |
| `PERF_CLIENTS` | `10` | Concurrent clients for load tests |
| `PERF_LOAD_DURATION_SECS` | `30` | Duration for sustained load tests |
| `PERF_STABILITY_DURATION_SECS` | `300` | Duration for stability tests |
| `PERF_SMALL_PAYLOAD` | `256` | Small payload size (bytes) |
| `PERF_MEDIUM_PAYLOAD` | `4096` | Medium payload size (bytes) |
| `PERF_LARGE_PAYLOAD` | `65536` | Large payload size (bytes) |
| `PERF_XL_PAYLOAD` | `1048576` | Extra large payload size (bytes) |
| `PERF_JSON_OUTPUT` | `0` | Output results as JSON |
| `PERF_OUTPUT_PATH` | - | File path for JSON results |

### Preset Configurations

The test suite includes several preset configurations:

1. **Verification** - Minimal iterations to verify tests work
2. **CI** - Balanced for CI pipelines (moderate iterations, shorter durations)
3. **Production** - Full production-like testing (high iterations, long durations)
4. **Constrained** - For resource-limited environments

## Output Format

Results can be output as JSON for integration with CI/CD pipelines and monitoring systems:

```bash
PERF_JSON_OUTPUT=1 PERF_OUTPUT_PATH=results.json cargo test --test perf_tests --release
```

The JSON output includes:
- Test name and configuration
- Latency statistics (mean, min, max, p50, p95, p99, p999)
- Throughput metrics (requests/second, MB/s)
- CPU utilization
- Memory usage
- Custom metrics per test

## Interpreting Results

### Key Metrics

1. **Gateway Overhead** - Look for `p50_overhead_percent` in baseline tests
   - < 10% is excellent
   - 10-25% is acceptable
   - > 25% may need investigation

2. **Virtual Tool Overhead** - Check `overhead_percent` in virtual tools tests
   - Renaming: Should be negligible (< 1%)
   - Defaults: < 5% expected
   - Output transformation: < 10% expected

3. **Throughput** - `requests_per_second` in load tests
   - Compare against baseline
   - Watch for degradation under load

4. **Memory Stability** - `growth_ratio` in stability tests
   - < 1.5x is stable
   - > 2x may indicate a memory leak

### Example Report

```
=== Gateway Overhead Analysis ===
Config: iterations=1000, direct_p50=1.2ms, gateway_p50=1.5ms
Latency: n=1000 mean=1.5ms p50=1.5ms p95=2.1ms p99=3.2ms
p50_overhead_percent: 25.0
p95_overhead_percent: 18.5
p99_overhead_percent: 15.2
```

## Hardware Recommendations

For accurate performance measurements:

| Environment | CPUs | Memory | Use Case |
|-------------|------|--------|----------|
| Development | 2 | 4GB | Quick verification |
| CI | 4 | 8GB | Standard testing |
| Staging | 8 | 16GB | Production-like |
| Benchmarking | 16+ | 32GB+ | Maximum throughput |

## Troubleshooting

### Tests fail with timeouts
- Increase `PERF_REQUEST_TIMEOUT_SECS`
- Check system resource availability
- Reduce `PERF_CLIENTS` for resource-constrained environments

### Inconsistent results
- Increase `PERF_WARMUP` iterations
- Ensure no other CPU-intensive processes running
- Use Docker with CPU pinning for isolation

### Memory errors in payload tests
- Reduce `PERF_XL_PAYLOAD` for constrained environments
- Increase memory allocation if using Docker

## Contributing

When adding new performance tests:

1. Add test module in `tests/perf/`
2. Create a `run_*_tests()` function that returns `PerfReport`
3. Add individual test functions with `test_*` prefix
4. Include unit tests for verification mode
5. Update the main runner in `perf_tests.rs`
6. Add documentation to this README
