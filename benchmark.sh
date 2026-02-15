#!/usr/bin/env bash
set -e

echo "=== microgpt benchmark ==="
echo ""

# Build Rust (exclude compile time)
echo "Building Rust (release)..."
cargo build --release 2>&1
echo ""

# Build Go (exclude compile time)
echo "Building Go..."
go build -o microgpt_go src/microgpt.go
echo ""

# Run Python
echo "Running Python..."
py_out=$(python3 src/microgpt.py)
py_loss=$(echo "$py_out" | grep "step 1000" | awk '{print $NF}')
py_time=$( { time python3 src/microgpt.py > /dev/null; } 2>&1 | grep real | awk '{print $2}' )
py_min=$(echo "$py_time" | sed 's/m.*//')
py_sec=$(echo "$py_time" | sed 's/.*m//' | sed 's/s//')
py_secs=$(echo "$py_min * 60 + $py_sec" | bc)
echo "  time: ${py_secs}s | final loss: $py_loss"
echo ""

# Run Rust
echo "Running Rust..."
rs_out=$(./target/release/microgpt_rs)
rs_loss=$(echo "$rs_out" | grep "step 1000" | awk '{print $NF}')
rs_time=$( { time ./target/release/microgpt_rs > /dev/null; } 2>&1 | grep real | awk '{print $2}' )
rs_min=$(echo "$rs_time" | sed 's/m.*//')
rs_sec=$(echo "$rs_time" | sed 's/.*m//' | sed 's/s//')
rs_secs=$(echo "$rs_min * 60 + $rs_sec" | bc)
echo "  time: ${rs_secs}s | final loss: $rs_loss"
echo ""

# Run Go
echo "Running Go..."
go_out=$(./microgpt_go)
go_loss=$(echo "$go_out" | grep "step 1000" | awk '{print $NF}')
go_time=$( { time ./microgpt_go > /dev/null; } 2>&1 | grep real | awk '{print $2}' )
go_min=$(echo "$go_time" | sed 's/m.*//')
go_sec=$(echo "$go_time" | sed 's/.*m//' | sed 's/s//')
go_secs=$(echo "$go_min * 60 + $go_sec" | bc)
echo "  time: ${go_secs}s | final loss: $go_loss"
echo ""

# Results
rs_speedup=$(echo "scale=1; $py_secs / $rs_secs" | bc)
go_speedup=$(echo "scale=1; $py_secs / $go_secs" | bc)
echo "=== Results ==="
echo "Python | ${py_secs}s | loss $py_loss"
echo "Rust   | ${rs_secs}s | loss $rs_loss | ${rs_speedup}x faster"
echo "Go     | ${go_secs}s | loss $go_loss | ${go_speedup}x faster"
