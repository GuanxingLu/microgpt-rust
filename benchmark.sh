#!/usr/bin/env bash
set -e

echo "=== microgpt benchmark ==="
echo ""

# Build Rust (exclude compile time)
echo "Building Rust (release)..."
cargo build --release 2>&1
echo ""

# Run Python
echo "Running Python..."
py_out=$(python3 microgpt.py)
py_loss=$(echo "$py_out" | grep "step 1000" | awk '{print $NF}')
py_time=$( { time python3 microgpt.py > /dev/null; } 2>&1 | grep real | awk '{print $2}' )
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

# Results
speedup=$(echo "scale=1; $py_secs / $rs_secs" | bc)
echo "=== Results ==="
echo "Python | ${py_secs}s | loss $py_loss"
echo "Rust   | ${rs_secs}s | loss $rs_loss"
echo "Speedup: ${speedup}x"
