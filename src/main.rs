///
/// The most atomic way to train and inference a GPT in pure, dependency-free Rust.
/// This file is the complete algorithm.
/// Everything else is just efficiency.
///
/// Ported from microgpt.py by @karpathy
///
use std::fs; // fs::read_to_string
use std::path::Path; // Path::new
use std::process::Command; // for downloading input.txt

// ---- Simple RNG (xorshift64) seeded deterministically ----
// random.seed, random.choices, random.gauss, random.shuffle
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    /// Box-Muller transform for normal distribution
    fn gauss(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = self.next_f64().max(1e-30); // avoid log(0)
        let u2 = self.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std * z
    }

    /// Weighted random choice, returns index
    fn choices(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        let mut r = self.next_f64() * total;
        for (i, &w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return i;
            }
        }
        weights.len() - 1
    }

    /// Fisher-Yates shuffle
    fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len();
        for i in (1..n).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

// Let there be Autograd, to recursively apply the chain rule through a computation graph
type ValIdx = usize;

struct Node {
    data: f64,             // scalar value of this node calculated during forward pass
    grad: f64,             // derivative of the loss w.r.t. this node, calculated in backward pass
    children: Vec<ValIdx>, // children of this node in the computation graph
    local_grads: Vec<f64>, // local derivative of this node w.r.t. its children
}

struct Tape {
    nodes: Vec<Node>,
}

impl Tape {
    fn new() -> Self {
        Tape { nodes: Vec::new() }
    }

    fn new_val(&mut self, data: f64) -> ValIdx {
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: 0.0,
            children: Vec::new(),
            local_grads: Vec::new(),
        });
        idx
    }

    fn data(&self, idx: ValIdx) -> f64 {
        self.nodes[idx].data
    }

    fn set_data(&mut self, idx: ValIdx, val: f64) {
        self.nodes[idx].data = val;
    }

    fn grad(&self, idx: ValIdx) -> f64 {
        self.nodes[idx].grad
    }

    fn add(&mut self, a: ValIdx, b: ValIdx) -> ValIdx {
        let data = self.nodes[a].data + self.nodes[b].data;
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data,
            grad: 0.0,
            children: vec![a, b],
            local_grads: vec![1.0, 1.0],
        });
        idx
    }

    fn mul(&mut self, a: ValIdx, b: ValIdx) -> ValIdx {
        let ad = self.nodes[a].data;
        let bd = self.nodes[b].data;
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data: ad * bd,
            grad: 0.0,
            children: vec![a, b],
            local_grads: vec![bd, ad],
        });
        idx
    }

    fn pow_const(&mut self, a: ValIdx, exponent: f64) -> ValIdx {
        let ad = self.nodes[a].data;
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data: ad.powf(exponent),
            grad: 0.0,
            children: vec![a],
            local_grads: vec![exponent * ad.powf(exponent - 1.0)],
        });
        idx
    }

    fn log(&mut self, a: ValIdx) -> ValIdx {
        let ad = self.nodes[a].data;
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data: ad.ln(),
            grad: 0.0,
            children: vec![a],
            local_grads: vec![1.0 / ad],
        });
        idx
    }

    fn exp(&mut self, a: ValIdx) -> ValIdx {
        let ad = self.nodes[a].data;
        let e = ad.exp();
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data: e,
            grad: 0.0,
            children: vec![a],
            local_grads: vec![e],
        });
        idx
    }

    fn relu(&mut self, a: ValIdx) -> ValIdx {
        let ad = self.nodes[a].data;
        let idx = self.nodes.len();
        self.nodes.push(Node {
            data: if ad > 0.0 { ad } else { 0.0 },
            grad: 0.0,
            children: vec![a],
            local_grads: vec![if ad > 0.0 { 1.0 } else { 0.0 }],
        });
        idx
    }

    fn neg(&mut self, a: ValIdx) -> ValIdx {
        let minus_one = self.new_val(-1.0);
        self.mul(a, minus_one)
    }

    fn div(&mut self, a: ValIdx, b: ValIdx) -> ValIdx {
        let inv_b = self.pow_const(b, -1.0);
        self.mul(a, inv_b)
    }

    fn add_const(&mut self, a: ValIdx, c: f64) -> ValIdx {
        let cv = self.new_val(c);
        self.add(a, cv)
    }

    fn mul_const(&mut self, a: ValIdx, c: f64) -> ValIdx {
        let cv = self.new_val(c);
        self.mul(a, cv)
    }

    fn backward(&mut self, root: ValIdx) {
        // Topological sort
        let n = self.nodes.len();
        let mut visited = vec![false; n];
        let mut topo = Vec::with_capacity(n);

        fn build_topo(nodes: &[Node], visited: &mut [bool], topo: &mut Vec<usize>, v: usize) {
            if visited[v] {
                return;
            }
            visited[v] = true;
            for &child in &nodes[v].children {
                build_topo(nodes, visited, topo, child);
            }
            topo.push(v);
        }

        build_topo(&self.nodes, &mut visited, &mut topo, root);

        // Zero all grads
        for node in self.nodes.iter_mut() {
            node.grad = 0.0;
        }
        self.nodes[root].grad = 1.0;

        // Backward pass
        for &v in topo.iter().rev() {
            let v_grad = self.nodes[v].grad;
            let n_children = self.nodes[v].children.len();
            for ci in 0..n_children {
                let child = self.nodes[v].children[ci];
                let lg = self.nodes[v].local_grads[ci];
                self.nodes[child].grad += lg * v_grad;
            }
        }
    }
}

// Define the model architecture: a stateless function mapping token sequence and parameters to logits over what comes next.
// Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU

fn linear(tape: &mut Tape, x: &[ValIdx], w: &[Vec<ValIdx>]) -> Vec<ValIdx> {
    let mut out = Vec::with_capacity(w.len());
    for row in w {
        let mut s = tape.mul(row[0], x[0]);
        for i in 1..row.len() {
            let prod = tape.mul(row[i], x[i]);
            s = tape.add(s, prod);
        }
        out.push(s);
    }
    out
}

fn softmax(tape: &mut Tape, logits: &[ValIdx]) -> Vec<ValIdx> {
    let max_val = logits
        .iter()
        .map(|&v| tape.data(v))
        .fold(f64::NEG_INFINITY, f64::max);
    let mut exps = Vec::with_capacity(logits.len());
    for &v in logits {
        let shifted = tape.add_const(v, -max_val);
        let e = tape.exp(shifted);
        exps.push(e);
    }
    // sum
    let mut total = exps[0];
    for i in 1..exps.len() {
        total = tape.add(total, exps[i]);
    }
    let mut probs = Vec::with_capacity(exps.len());
    for &e in &exps {
        let p = tape.div(e, total);
        probs.push(p);
    }
    probs
}

fn rmsnorm(tape: &mut Tape, x: &[ValIdx]) -> Vec<ValIdx> {
    let n = x.len() as f64;
    // ms = sum(xi * xi) / len(x)
    let mut ms = tape.mul(x[0], x[0]);
    for i in 1..x.len() {
        let sq = tape.mul(x[i], x[i]);
        ms = tape.add(ms, sq);
    }
    ms = tape.mul_const(ms, 1.0 / n);
    // scale = (ms + 1e-5) ^ -0.5
    let ms_eps = tape.add_const(ms, 1e-5);
    let scale = tape.pow_const(ms_eps, -0.5);
    // x * scale
    x.iter().map(|&xi| tape.mul(xi, scale)).collect()
}

/// State dict: maps string keys to matrices (Vec<Vec<ValIdx>>)
struct StateDict {
    entries: Vec<(String, Vec<Vec<ValIdx>>)>,
}

impl StateDict {
    fn new() -> Self {
        StateDict {
            entries: Vec::new(),
        }
    }

    fn insert(&mut self, key: &str, val: Vec<Vec<ValIdx>>) {
        self.entries.push((key.to_string(), val));
    }

    fn get(&self, key: &str) -> &Vec<Vec<ValIdx>> {
        for (k, v) in &self.entries {
            if k == key {
                return v;
            }
        }
        panic!("key not found: {}", key);
    }
}

fn gpt(
    tape: &mut Tape,
    sd: &StateDict,
    token_id: usize,
    pos_id: usize,
    keys: &mut Vec<Vec<Vec<ValIdx>>>,
    values: &mut Vec<Vec<Vec<ValIdx>>>,
    n_layer: usize,
    n_head: usize,
    head_dim: usize,
    n_embd: usize,
) -> Vec<ValIdx> {
    let tok_emb = &sd.get("wte")[token_id]; // token embedding
    let pos_emb = &sd.get("wpe")[pos_id]; // position embedding
    let mut x: Vec<ValIdx> = tok_emb // joint token and position embedding
        .iter()
        .zip(pos_emb.iter())
        .map(|(&t, &p)| tape.add(t, p))
        .collect();
    x = rmsnorm(tape, &x);

    for li in 0..n_layer {
        // 1) Multi-head attention block
        let x_residual = x.clone();
        x = rmsnorm(tape, &x);
        let wq_key = format!("layer{}.attn_wq", li);
        let wk_key = format!("layer{}.attn_wk", li);
        let wv_key = format!("layer{}.attn_wv", li);
        let wo_key = format!("layer{}.attn_wo", li);
        let q = linear(tape, &x, sd.get(&wq_key));
        let k = linear(tape, &x, sd.get(&wk_key));
        let v = linear(tape, &x, sd.get(&wv_key));
        keys[li].push(k);
        values[li].push(v);

        let mut x_attn = Vec::with_capacity(n_embd);
        for h in 0..n_head {
            let hs = h * head_dim;
            let q_h: Vec<ValIdx> = (hs..hs + head_dim)
                .map(|j| {
                    // q is the last computed q
                    q[j]
                })
                .collect();

            let n_ctx = keys[li].len();
            let k_h: Vec<Vec<ValIdx>> = (0..n_ctx)
                .map(|t| (hs..hs + head_dim).map(|j| keys[li][t][j]).collect())
                .collect();
            let v_h: Vec<Vec<ValIdx>> = (0..n_ctx)
                .map(|t| (hs..hs + head_dim).map(|j| values[li][t][j]).collect())
                .collect();

            // attn_logits
            let scale = 1.0 / (head_dim as f64).sqrt();
            let mut attn_logits = Vec::with_capacity(n_ctx);
            for t in 0..n_ctx {
                let mut dot = tape.mul(q_h[0], k_h[t][0]);
                for j in 1..head_dim {
                    let prod = tape.mul(q_h[j], k_h[t][j]);
                    dot = tape.add(dot, prod);
                }
                let scaled = tape.mul_const(dot, scale);
                attn_logits.push(scaled);
            }

            let attn_weights = softmax(tape, &attn_logits);

            // head_out[j] = sum_t attn_weights[t] * v_h[t][j]
            for j in 0..head_dim {
                let mut s = tape.mul(attn_weights[0], v_h[0][j]);
                for t in 1..n_ctx {
                    let prod = tape.mul(attn_weights[t], v_h[t][j]);
                    s = tape.add(s, prod);
                }
                x_attn.push(s);
            }
        }

        x = linear(tape, &x_attn, sd.get(&wo_key));
        x = x
            .iter()
            .zip(x_residual.iter())
            .map(|(&a, &b)| tape.add(a, b))
            .collect();

        // 2) MLP block
        let x_residual = x.clone();
        x = rmsnorm(tape, &x);
        let fc1_key = format!("layer{}.mlp_fc1", li);
        let fc2_key = format!("layer{}.mlp_fc2", li);
        x = linear(tape, &x, sd.get(&fc1_key));
        x = x.iter().map(|&xi| tape.relu(xi)).collect();
        x = linear(tape, &x, sd.get(&fc2_key));
        x = x
            .iter()
            .zip(x_residual.iter())
            .map(|(&a, &b)| tape.add(a, b))
            .collect();
    }

    linear(tape, &x, sd.get("lm_head"))
}

fn main() {
    // Let there be an input dataset `docs`: list of documents (e.g. a dataset of names)
    let input_path = "input.txt";
    if !Path::new(input_path).exists() {
        let url = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt";
        Command::new("curl")
            .args(["-sL", "-o", input_path, url])
            .status()
            .expect("failed to download input.txt (curl required)");
    }
    let content = fs::read_to_string(input_path).expect("failed to read input.txt");
    let mut docs: Vec<&str> = content
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();

    let mut rng = Rng::new(42); // Let there be order among chaos
    rng.shuffle(&mut docs);
    println!("num docs: {}", docs.len());

    // Let there be a Tokenizer to translate strings to discrete symbols and back
    let mut uchars: Vec<char> = {
        // unique characters in the dataset become token ids 0..n-1
        let mut set = std::collections::BTreeSet::new();
        for doc in &docs {
            for ch in doc.chars() {
                set.insert(ch);
            }
        }
        set.into_iter().collect()
    };
    uchars.sort();
    let bos = uchars.len(); // token id for the special Beginning of Sequence (BOS) token
    let vocab_size = uchars.len() + 1; // total number of unique tokens, +1 is for BOS
    println!("vocab size: {}", vocab_size);

    fn char_to_id(uchars: &[char], ch: char) -> usize {
        uchars.iter().position(|&c| c == ch).unwrap()
    }

    // Initialize the parameters, to store the knowledge of the model.
    let n_embd: usize = 16; // embedding dimension
    let n_head: usize = 4; // number of attention heads
    let n_layer: usize = 1; // number of layers
    let block_size: usize = 16; // maximum sequence length
    let head_dim = n_embd / n_head; // dimension of each head

    // Persistent tape just for parameters
    let mut param_tape = Tape::new();
    let mut sd = StateDict::new();

    let matrix =
        |tape: &mut Tape, rng: &mut Rng, nout: usize, nin: usize, std: f64| -> Vec<Vec<ValIdx>> {
            (0..nout)
                .map(|_| {
                    (0..nin)
                        .map(|_| tape.new_val(rng.gauss(0.0, std)))
                        .collect()
                })
                .collect()
        };

    sd.insert(
        "wte",
        matrix(&mut param_tape, &mut rng, vocab_size, n_embd, 0.08),
    );
    sd.insert(
        "wpe",
        matrix(&mut param_tape, &mut rng, block_size, n_embd, 0.08),
    );
    sd.insert(
        "lm_head",
        matrix(&mut param_tape, &mut rng, vocab_size, n_embd, 0.08),
    );

    for i in 0..n_layer {
        let k = format!("layer{}.attn_wq", i);
        let v = matrix(&mut param_tape, &mut rng, n_embd, n_embd, 0.08);
        sd.insert(&k, v);
        let k = format!("layer{}.attn_wk", i);
        let v = matrix(&mut param_tape, &mut rng, n_embd, n_embd, 0.08);
        sd.insert(&k, v);
        let k = format!("layer{}.attn_wv", i);
        let v = matrix(&mut param_tape, &mut rng, n_embd, n_embd, 0.08);
        sd.insert(&k, v);
        let k = format!("layer{}.attn_wo", i);
        let v = matrix(&mut param_tape, &mut rng, n_embd, n_embd, 0.08);
        sd.insert(&k, v);
        let k = format!("layer{}.mlp_fc1", i);
        let v = matrix(&mut param_tape, &mut rng, 4 * n_embd, n_embd, 0.08);
        sd.insert(&k, v);
        let k = format!("layer{}.mlp_fc2", i);
        let v = matrix(&mut param_tape, &mut rng, n_embd, 4 * n_embd, 0.08);
        sd.insert(&k, v);
    }

    // Flatten params into a single list
    let param_indices: Vec<ValIdx> = sd
        .entries
        .iter()
        .flat_map(|(_, mat)| mat.iter().flat_map(|row| row.iter().copied()))
        .collect();
    let num_params = param_indices.len();
    println!("num params: {}", num_params);

    // Let there be Adam, the blessed optimizer and its buffers
    let learning_rate: f64 = 0.01;
    let beta1: f64 = 0.85;
    let beta2: f64 = 0.99;
    let eps_adam: f64 = 1e-8;
    let mut m_buf = vec![0.0f64; num_params]; // first moment buffer
    let mut v_buf = vec![0.0f64; num_params]; // second moment buffer

    // Repeat in sequence
    let num_steps = 1000; // number of training steps
    for step in 0..num_steps {
        // Take single document, tokenize it, surround it with BOS special token on both sides
        let doc = docs[step % docs.len()];
        let mut tokens: Vec<usize> = Vec::with_capacity(doc.len() + 2);
        tokens.push(bos);
        for ch in doc.chars() {
            tokens.push(char_to_id(&uchars, ch));
        }
        tokens.push(bos);
        let n = std::cmp::min(block_size, tokens.len() - 1);

        // Create a fresh tape for this step, but copy param data from param_tape
        let mut tape = Tape::new();
        // Re-create param nodes in the new tape with current data values
        let mut step_sd = StateDict::new();
        for (key, mat) in &sd.entries {
            let new_mat: Vec<Vec<ValIdx>> = mat
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|&pidx| tape.new_val(param_tape.data(pidx)))
                        .collect()
                })
                .collect();
            step_sd.insert(key, new_mat);
        }

        // Map from step_sd param indices back to param_tape indices
        let step_param_indices: Vec<ValIdx> = step_sd
            .entries
            .iter()
            .flat_map(|(_, mat)| mat.iter().flat_map(|row| row.iter().copied()))
            .collect();

        // Forward the token sequence through the model, building up the computation graph all the way to the loss.
        let mut keys_cache: Vec<Vec<Vec<ValIdx>>> = (0..n_layer).map(|_| Vec::new()).collect();
        let mut values_cache: Vec<Vec<Vec<ValIdx>>> = (0..n_layer).map(|_| Vec::new()).collect();
        let mut losses = Vec::with_capacity(n);

        for pos_id in 0..n {
            let token_id = tokens[pos_id];
            let target_id = tokens[pos_id + 1];
            let logits = gpt(
                &mut tape,
                &step_sd,
                token_id,
                pos_id,
                &mut keys_cache,
                &mut values_cache,
                n_layer,
                n_head,
                head_dim,
                n_embd,
            );
            let probs = softmax(&mut tape, &logits);
            let log_p = tape.log(probs[target_id]);
            let neg_log_p = tape.neg(log_p);
            losses.push(neg_log_p);
        }

        // final average loss over the document sequence. May yours be low.
        let mut total_loss = losses[0];
        for i in 1..losses.len() {
            total_loss = tape.add(total_loss, losses[i]);
        }
        let loss = tape.mul_const(total_loss, 1.0 / n as f64);

        // Backward the loss, calculating the gradients with respect to all model parameters.
        tape.backward(loss);

        let loss_val = tape.data(loss);

        // Adam optimizer update: update the model parameters based on the corresponding gradients.
        let lr_t = learning_rate * (1.0 - step as f64 / num_steps as f64); // linear learning rate decay
        for (i, (&step_pidx, &orig_pidx)) in step_param_indices
            .iter()
            .zip(param_indices.iter())
            .enumerate()
        {
            let g = tape.grad(step_pidx);
            m_buf[i] = beta1 * m_buf[i] + (1.0 - beta1) * g;
            v_buf[i] = beta2 * v_buf[i] + (1.0 - beta2) * g * g;
            let m_hat = m_buf[i] / (1.0 - beta1.powi((step + 1) as i32));
            let v_hat = v_buf[i] / (1.0 - beta2.powi((step + 1) as i32));
            let new_data = param_tape.data(orig_pidx) - lr_t * m_hat / (v_hat.sqrt() + eps_adam);
            param_tape.set_data(orig_pidx, new_data);
        }

        println!(
            "step {:4} / {:4} | loss {:.4}",
            step + 1,
            num_steps,
            loss_val
        );
    }

    // Inference: may the model babble back to us
    let temperature: f64 = 0.5; // in (0, 1], control the "creativity" of generated text, low to high
    println!("\n--- inference (new, hallucinated names) ---");
    for sample_idx in 0..20 {
        // Build a fresh tape with current params for inference
        let mut tape = Tape::new();
        let mut inf_sd = StateDict::new();
        for (key, mat) in &sd.entries {
            let new_mat: Vec<Vec<ValIdx>> = mat
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|&pidx| tape.new_val(param_tape.data(pidx)))
                        .collect()
                })
                .collect();
            inf_sd.insert(key, new_mat);
        }

        let mut keys_cache: Vec<Vec<Vec<ValIdx>>> = (0..n_layer).map(|_| Vec::new()).collect();
        let mut values_cache: Vec<Vec<Vec<ValIdx>>> = (0..n_layer).map(|_| Vec::new()).collect();
        let mut token_id = bos;
        let mut sample = String::new();

        for pos_id in 0..block_size {
            let logits = gpt(
                &mut tape,
                &inf_sd,
                token_id,
                pos_id,
                &mut keys_cache,
                &mut values_cache,
                n_layer,
                n_head,
                head_dim,
                n_embd,
            );
            // Apply temperature
            let scaled_logits: Vec<ValIdx> = logits
                .iter()
                .map(|&l| tape.mul_const(l, 1.0 / temperature))
                .collect();
            let probs = softmax(&mut tape, &scaled_logits);
            let weights: Vec<f64> = probs.iter().map(|&p| tape.data(p)).collect();
            token_id = rng.choices(&weights);
            if token_id == bos {
                break;
            }
            sample.push(uchars[token_id]);
        }
        println!("sample {:2}: {}", sample_idx + 1, sample);
    }
}
