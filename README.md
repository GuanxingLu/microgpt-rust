# microgpt

An atomic way to train and inference a GPT in Python, Rust, and Go. Inspired by [@karpathy](https://github.com/karpathy).

All three implementations train a 1-layer transformer on character-level name generation using the [makemore](https://github.com/karpathy/makemore) dataset.

## Quick start

Run all three implementations and compare:

```bash
bash benchmark.sh
```

Results on my own Apple M4 Max:

| | Lines | Time | Speedup | Final Loss |
|---|---|---|---|---|
| Python | 215 | 44.210s | 1.0x | 2.6497 |
| Rust | 617 | 4.698s | 9.4x | 2.3251 |
| Go | 635 | 4.000s | 11.1x | 2.3251 |

NOTE: Losses differ slightly due to different RNG implementations (Python's Mersenne Twister vs Rust/Go's xorshift64), which produces different initial weights and data shuffle order.

## License

MIT
