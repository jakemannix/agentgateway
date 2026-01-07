#!/bin/bash
# AgentGateway Performance Test Runner
#
# This script provides an easy way to run performance tests with various
# configurations and hardware constraints.
#
# Usage:
#   ./perf/run-perf-tests.sh                    # Run with defaults
#   ./perf/run-perf-tests.sh --verify           # Quick verification
#   ./perf/run-perf-tests.sh --ci               # CI configuration
#   ./perf/run-perf-tests.sh --production       # Full production tests
#   ./perf/run-perf-tests.sh --cpus 4 --memory 8g
#   ./perf/run-perf-tests.sh --test baseline    # Specific test category
#   ./perf/run-perf-tests.sh --local            # Run locally without Docker

set -e

# Default values
CPUS="4"
MEMORY="8g"
TEST_CATEGORY=""
PROFILE=""
LOCAL_RUN=false
VERIFY_MODE=false
JSON_OUTPUT=false
OUTPUT_DIR="./perf/results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_banner() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║          AgentGateway Performance Test Runner                ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
}

print_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --verify          Quick verification mode (minimal iterations)"
    echo "  --ci              CI-optimized configuration"
    echo "  --production      Full production-like tests"
    echo "  --constrained     Resource-constrained environment test"
    echo "  --cpus N          Number of CPUs to allocate (default: 4)"
    echo "  --memory SIZE     Memory limit (e.g., 8g, 4096m) (default: 8g)"
    echo "  --test CATEGORY   Run specific test category:"
    echo "                    baseline, virtual_tools, streaming, failover,"
    echo "                    load, stability, payload"
    echo "  --local           Run locally without Docker"
    echo "  --json            Output results as JSON"
    echo "  --output DIR      Output directory for results (default: ./perf/results)"
    echo "  --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --verify                    # Quick verification"
    echo "  $0 --cpus 8 --memory 16g       # Custom hardware"
    echo "  $0 --test baseline --local     # Run baseline tests locally"
    echo "  $0 --production                # Full production tests"
    echo ""
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --verify)
                VERIFY_MODE=true
                PROFILE="verify"
                shift
                ;;
            --ci)
                PROFILE="ci"
                shift
                ;;
            --production)
                PROFILE="production"
                shift
                ;;
            --constrained)
                PROFILE="constrained"
                shift
                ;;
            --cpus)
                CPUS="$2"
                shift 2
                ;;
            --memory)
                MEMORY="$2"
                shift 2
                ;;
            --test)
                TEST_CATEGORY="$2"
                shift 2
                ;;
            --local)
                LOCAL_RUN=true
                shift
                ;;
            --json)
                JSON_OUTPUT=true
                shift
                ;;
            --output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --help)
                print_help
                exit 0
                ;;
            *)
                echo -e "${RED}Unknown option: $1${NC}"
                print_help
                exit 1
                ;;
        esac
    done
}

run_local() {
    echo -e "${GREEN}Running performance tests locally...${NC}"
    echo ""

    # Set up environment
    export PERF_JSON_OUTPUT="${JSON_OUTPUT}"
    export PERF_OUTPUT_PATH="${OUTPUT_DIR}/perf-results-local.json"

    if [ "$VERIFY_MODE" = true ]; then
        export PERF_VERIFY=1
    fi

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Build if needed
    echo "Building release binary..."
    cargo build --release --tests -p agentgateway

    # Run tests
    if [ -n "$TEST_CATEGORY" ]; then
        echo "Running ${TEST_CATEGORY} tests..."
        cargo test --test perf_tests --release "${TEST_CATEGORY}_tests" -- --nocapture --test-threads=1
    else
        echo "Running all performance tests..."
        cargo test --test perf_tests --release all_perf_tests -- --nocapture --test-threads=1
    fi

    echo ""
    echo -e "${GREEN}Tests completed!${NC}"
    if [ "$JSON_OUTPUT" = true ] && [ -f "$PERF_OUTPUT_PATH" ]; then
        echo "Results saved to: $PERF_OUTPUT_PATH"
    fi
}

run_docker() {
    echo -e "${GREEN}Running performance tests in Docker...${NC}"
    echo "  CPUs: ${CPUS}"
    echo "  Memory: ${MEMORY}"
    if [ -n "$PROFILE" ]; then
        echo "  Profile: ${PROFILE}"
    fi
    echo ""

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Set environment variables for docker-compose
    export PERF_CPUS="$CPUS"
    export PERF_MEMORY="$MEMORY"

    # Determine which service to run
    local SERVICE="perf-tests"
    local COMPOSE_ARGS=""

    if [ -n "$PROFILE" ]; then
        COMPOSE_ARGS="--profile $PROFILE"
        case $PROFILE in
            verify)
                SERVICE="perf-tests-verify"
                ;;
            ci)
                SERVICE="perf-tests-ci"
                ;;
            production)
                SERVICE="perf-tests-production"
                ;;
            constrained)
                SERVICE="perf-tests-constrained"
                ;;
        esac
    fi

    # Build the image
    echo "Building Docker image..."
    docker-compose -f perf/docker-compose.perf.yml $COMPOSE_ARGS build $SERVICE

    # Run tests
    if [ -n "$TEST_CATEGORY" ]; then
        echo "Running ${TEST_CATEGORY} tests..."
        docker-compose -f perf/docker-compose.perf.yml $COMPOSE_ARGS run --rm $SERVICE \
            "${TEST_CATEGORY}_tests" --test-threads=1 --nocapture
    else
        echo "Running all performance tests..."
        docker-compose -f perf/docker-compose.perf.yml $COMPOSE_ARGS run --rm $SERVICE \
            --test-threads=1 --nocapture
    fi

    echo ""
    echo -e "${GREEN}Tests completed!${NC}"
    echo "Results available in: ${OUTPUT_DIR}/"
}

check_dependencies() {
    if [ "$LOCAL_RUN" = true ]; then
        if ! command -v cargo &> /dev/null; then
            echo -e "${RED}Error: cargo is required for local runs${NC}"
            exit 1
        fi
    else
        if ! command -v docker &> /dev/null; then
            echo -e "${RED}Error: docker is required for containerized runs${NC}"
            exit 1
        fi
        if ! command -v docker-compose &> /dev/null; then
            echo -e "${RED}Error: docker-compose is required for containerized runs${NC}"
            exit 1
        fi
    fi
}

main() {
    print_banner
    parse_args "$@"
    check_dependencies

    echo "Configuration:"
    echo "  Mode: $([ "$LOCAL_RUN" = true ] && echo "Local" || echo "Docker")"
    echo "  Verify: ${VERIFY_MODE}"
    if [ -n "$TEST_CATEGORY" ]; then
        echo "  Test Category: ${TEST_CATEGORY}"
    else
        echo "  Test Category: all"
    fi
    echo ""

    if [ "$LOCAL_RUN" = true ]; then
        run_local
    else
        run_docker
    fi
}

main "$@"
