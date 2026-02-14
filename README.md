# microgpt

An atomic way to train and inference a GPT in Rust. Inspired by [@karpathy](https://github.com/karpathy).

Both implementations train a 1-layer transformer on character-level name generation using the [makemore](https://github.com/karpathy/makemore) dataset.

## Quick start

```bash
cargo run --release
```

## Benchmark

Run both Python and Rust implementations and compare:

```bash
bash benchmark.sh
```

Results on my own Apple M4 Max:

| | Time | Final Loss |
|---|---|---|
| Python | 44.210s | 2.6497 |
| Rust | 4.698s | 2.3251 |
| **Speedup** | **~9.4x** | |

NOTE: Losses differ slightly due to different RNG implementations (Python's Mersenne Twister vs Rust's xorshift64), which produces different initial weights and data shuffle order.

## License

MIT
