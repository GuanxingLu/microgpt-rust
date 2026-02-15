///
/// The most atomic way to train and inference a GPT in pure, dependency-free Go.
/// This file is the complete algorithm.
/// Everything else is just efficiency.
///
/// Ported from microgpt.py by @karpathy
///

package main

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"sort"
	"strings"
)

// ---- Simple RNG (xorshift64) seeded deterministically ----
// random.seed, random.choices, random.gauss, random.shuffle

type Rng struct {
	state uint64
}

func NewRng(seed uint64) *Rng {
	return &Rng{state: seed}
}

func (r *Rng) NextU64() uint64 {
	r.state ^= r.state << 13
	r.state ^= r.state >> 7
	r.state ^= r.state << 17
	return r.state
}

func (r *Rng) NextF64() float64 {
	return float64(r.NextU64()) / float64(^uint64(0))
}

// Box-Muller transform for normal distribution
func (r *Rng) Gauss(mean, std float64) float64 {
	u1 := math.Max(r.NextF64(), 1e-30) // avoid log(0)
	u2 := r.NextF64()
	z := math.Sqrt(-2.0*math.Log(u1)) * math.Cos(2.0*math.Pi*u2)
	return mean + std*z
}

// Weighted random choice, returns index
func (r *Rng) Choices(weights []float64) int {
	total := 0.0
	for _, w := range weights {
		total += w
	}
	rv := r.NextF64() * total
	for i, w := range weights {
		rv -= w
		if rv <= 0.0 {
			return i
		}
	}
	return len(weights) - 1
}

// Fisher-Yates shuffle
func (r *Rng) Shuffle(slice []string) {
	n := len(slice)
	for i := n - 1; i >= 1; i-- {
		j := int(r.NextU64() % uint64(i+1))
		slice[i], slice[j] = slice[j], slice[i]
	}
}

// Let there be Autograd, to recursively apply the chain rule through a computation graph

type ValIdx = int

type Node struct {
	data       float64   // scalar value of this node calculated during forward pass
	grad       float64   // derivative of the loss w.r.t. this node, calculated in backward pass
	children   []ValIdx  // children of this node in the computation graph
	localGrads []float64 // local derivative of this node w.r.t. its children
}

type Tape struct {
	nodes []Node
}

func NewTape() *Tape {
	return &Tape{}
}

func (t *Tape) NewVal(data float64) ValIdx {
	idx := len(t.nodes)
	t.nodes = append(t.nodes, Node{
		data:       data,
		grad:       0.0,
		children:   nil,
		localGrads: nil,
	})
	return idx
}

func (t *Tape) Data(idx ValIdx) float64 {
	return t.nodes[idx].data
}

func (t *Tape) SetData(idx ValIdx, val float64) {
	t.nodes[idx].data = val
}

func (t *Tape) Grad(idx ValIdx) float64 {
	return t.nodes[idx].grad
}

func (t *Tape) Add(a, b ValIdx) ValIdx {
	data := t.nodes[a].data + t.nodes[b].data
	idx := len(t.nodes)
	t.nodes = append(t.nodes, Node{
		data:       data,
		grad:       0.0,
		children:   []ValIdx{a, b},
		localGrads: []float64{1.0, 1.0},
	})
	return idx
}

func (t *Tape) Mul(a, b ValIdx) ValIdx {
	ad := t.nodes[a].data
	bd := t.nodes[b].data
	idx := len(t.nodes)
	t.nodes = append(t.nodes, Node{
		data:       ad * bd,
		grad:       0.0,
		children:   []ValIdx{a, b},
		localGrads: []float64{bd, ad},
	})
	return idx
}

func (t *Tape) PowConst(a ValIdx, exponent float64) ValIdx {
	ad := t.nodes[a].data
	idx := len(t.nodes)
	t.nodes = append(t.nodes, Node{
		data:       math.Pow(ad, exponent),
		grad:       0.0,
		children:   []ValIdx{a},
		localGrads: []float64{exponent * math.Pow(ad, exponent-1.0)},
	})
	return idx
}

func (t *Tape) Log(a ValIdx) ValIdx {
	ad := t.nodes[a].data
	idx := len(t.nodes)
	t.nodes = append(t.nodes, Node{
		data:       math.Log(ad),
		grad:       0.0,
		children:   []ValIdx{a},
		localGrads: []float64{1.0 / ad},
	})
	return idx
}

func (t *Tape) Exp(a ValIdx) ValIdx {
	ad := t.nodes[a].data
	e := math.Exp(ad)
	idx := len(t.nodes)
	t.nodes = append(t.nodes, Node{
		data:       e,
		grad:       0.0,
		children:   []ValIdx{a},
		localGrads: []float64{e},
	})
	return idx
}

func (t *Tape) Relu(a ValIdx) ValIdx {
	ad := t.nodes[a].data
	var d, lg float64
	if ad > 0.0 {
		d = ad
		lg = 1.0
	}
	idx := len(t.nodes)
	t.nodes = append(t.nodes, Node{
		data:       d,
		grad:       0.0,
		children:   []ValIdx{a},
		localGrads: []float64{lg},
	})
	return idx
}

func (t *Tape) Neg(a ValIdx) ValIdx {
	minusOne := t.NewVal(-1.0)
	return t.Mul(a, minusOne)
}

func (t *Tape) Div(a, b ValIdx) ValIdx {
	invB := t.PowConst(b, -1.0)
	return t.Mul(a, invB)
}

func (t *Tape) AddConst(a ValIdx, c float64) ValIdx {
	cv := t.NewVal(c)
	return t.Add(a, cv)
}

func (t *Tape) MulConst(a ValIdx, c float64) ValIdx {
	cv := t.NewVal(c)
	return t.Mul(a, cv)
}

func (t *Tape) Backward(root ValIdx) {
	// Topological sort
	n := len(t.nodes)
	visited := make([]bool, n)
	topo := make([]int, 0, n)

	var buildTopo func(v int)
	buildTopo = func(v int) {
		if visited[v] {
			return
		}
		visited[v] = true
		for _, child := range t.nodes[v].children {
			buildTopo(child)
		}
		topo = append(topo, v)
	}
	buildTopo(root)

	// Zero all grads
	for i := range t.nodes {
		t.nodes[i].grad = 0.0
	}
	t.nodes[root].grad = 1.0

	// Backward pass
	for i := len(topo) - 1; i >= 0; i-- {
		v := topo[i]
		vGrad := t.nodes[v].grad
		for ci, child := range t.nodes[v].children {
			lg := t.nodes[v].localGrads[ci]
			t.nodes[child].grad += lg * vGrad
		}
	}
}

// Define the model architecture: a stateless function mapping token sequence and parameters to logits over what comes next.
// Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU

func linear(tape *Tape, x []ValIdx, w [][]ValIdx) []ValIdx {
	out := make([]ValIdx, 0, len(w))
	for _, row := range w {
		s := tape.Mul(row[0], x[0])
		for i := 1; i < len(row); i++ {
			prod := tape.Mul(row[i], x[i])
			s = tape.Add(s, prod)
		}
		out = append(out, s)
	}
	return out
}

func softmax(tape *Tape, logits []ValIdx) []ValIdx {
	maxVal := math.Inf(-1)
	for _, v := range logits {
		d := tape.Data(v)
		if d > maxVal {
			maxVal = d
		}
	}
	exps := make([]ValIdx, 0, len(logits))
	for _, v := range logits {
		shifted := tape.AddConst(v, -maxVal)
		e := tape.Exp(shifted)
		exps = append(exps, e)
	}
	// sum
	total := exps[0]
	for i := 1; i < len(exps); i++ {
		total = tape.Add(total, exps[i])
	}
	probs := make([]ValIdx, 0, len(exps))
	for _, e := range exps {
		p := tape.Div(e, total)
		probs = append(probs, p)
	}
	return probs
}

func rmsnorm(tape *Tape, x []ValIdx) []ValIdx {
	n := float64(len(x))
	// ms = sum(xi * xi) / len(x)
	ms := tape.Mul(x[0], x[0])
	for i := 1; i < len(x); i++ {
		sq := tape.Mul(x[i], x[i])
		ms = tape.Add(ms, sq)
	}
	ms = tape.MulConst(ms, 1.0/n)
	// scale = (ms + 1e-5) ^ -0.5
	msEps := tape.AddConst(ms, 1e-5)
	scale := tape.PowConst(msEps, -0.5)
	// x * scale
	out := make([]ValIdx, len(x))
	for i, xi := range x {
		out[i] = tape.Mul(xi, scale)
	}
	return out
}

// State dict: maps string keys to matrices ([][]ValIdx)

type StateDict struct {
	entries []struct {
		key string
		val [][]ValIdx
	}
}

func NewStateDict() *StateDict {
	return &StateDict{}
}

func (sd *StateDict) Insert(key string, val [][]ValIdx) {
	sd.entries = append(sd.entries, struct {
		key string
		val [][]ValIdx
	}{key, val})
}

func (sd *StateDict) Get(key string) [][]ValIdx {
	for _, e := range sd.entries {
		if e.key == key {
			return e.val
		}
	}
	panic("key not found: " + key)
}

func gpt(
	tape *Tape,
	sd *StateDict,
	tokenID int,
	posID int,
	keys [][][]ValIdx,
	values [][][]ValIdx,
	nLayer int,
	nHead int,
	headDim int,
	nEmbd int,
) []ValIdx {
	tokEmb := sd.Get("wte")[tokenID] // token embedding
	posEmb := sd.Get("wpe")[posID]   // position embedding
	x := make([]ValIdx, nEmbd)       // joint token and position embedding
	for i := 0; i < nEmbd; i++ {
		x[i] = tape.Add(tokEmb[i], posEmb[i])
	}
	x = rmsnorm(tape, x)

	for li := 0; li < nLayer; li++ {
		// 1) Multi-head attention block
		xResidual := make([]ValIdx, len(x))
		copy(xResidual, x)
		x = rmsnorm(tape, x)
		wqKey := fmt.Sprintf("layer%d.attn_wq", li)
		wkKey := fmt.Sprintf("layer%d.attn_wk", li)
		wvKey := fmt.Sprintf("layer%d.attn_wv", li)
		woKey := fmt.Sprintf("layer%d.attn_wo", li)
		q := linear(tape, x, sd.Get(wqKey))
		k := linear(tape, x, sd.Get(wkKey))
		v := linear(tape, x, sd.Get(wvKey))
		keys[li] = append(keys[li], k)
		values[li] = append(values[li], v)

		xAttn := make([]ValIdx, 0, nEmbd)
		for h := 0; h < nHead; h++ {
			hs := h * headDim
			qH := make([]ValIdx, headDim)
			for j := 0; j < headDim; j++ {
				qH[j] = q[hs+j]
			}

			nCtx := len(keys[li])
			kH := make([][]ValIdx, nCtx)
			vH := make([][]ValIdx, nCtx)
			for t := 0; t < nCtx; t++ {
				kH[t] = make([]ValIdx, headDim)
				vH[t] = make([]ValIdx, headDim)
				for j := 0; j < headDim; j++ {
					kH[t][j] = keys[li][t][hs+j]
					vH[t][j] = values[li][t][hs+j]
				}
			}

			// attn_logits
			scaleVal := 1.0 / math.Sqrt(float64(headDim))
			attnLogits := make([]ValIdx, nCtx)
			for t := 0; t < nCtx; t++ {
				dot := tape.Mul(qH[0], kH[t][0])
				for j := 1; j < headDim; j++ {
					prod := tape.Mul(qH[j], kH[t][j])
					dot = tape.Add(dot, prod)
				}
				scaled := tape.MulConst(dot, scaleVal)
				attnLogits[t] = scaled
			}

			attnWeights := softmax(tape, attnLogits)

			// head_out[j] = sum_t attn_weights[t] * v_h[t][j]
			for j := 0; j < headDim; j++ {
				s := tape.Mul(attnWeights[0], vH[0][j])
				for t := 1; t < nCtx; t++ {
					prod := tape.Mul(attnWeights[t], vH[t][j])
					s = tape.Add(s, prod)
				}
				xAttn = append(xAttn, s)
			}
		}

		x = linear(tape, xAttn, sd.Get(woKey))
		for i := 0; i < len(x); i++ {
			x[i] = tape.Add(x[i], xResidual[i])
		}

		// 2) MLP block
		xResidual = make([]ValIdx, len(x))
		copy(xResidual, x)
		x = rmsnorm(tape, x)
		fc1Key := fmt.Sprintf("layer%d.mlp_fc1", li)
		fc2Key := fmt.Sprintf("layer%d.mlp_fc2", li)
		x = linear(tape, x, sd.Get(fc1Key))
		for i := range x {
			x[i] = tape.Relu(x[i])
		}
		x = linear(tape, x, sd.Get(fc2Key))
		for i := 0; i < len(x); i++ {
			x[i] = tape.Add(x[i], xResidual[i])
		}
	}

	return linear(tape, x, sd.Get("lm_head"))
}

func main() {
	// Let there be an input dataset `docs`: list of documents (e.g. a dataset of names)
	inputPath := "input.txt"
	if _, err := os.Stat(inputPath); os.IsNotExist(err) {
		url := "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
		cmd := exec.Command("curl", "-sL", "-o", inputPath, url)
		if err := cmd.Run(); err != nil {
			panic("failed to download input.txt (curl required)")
		}
	}
	contentBytes, err := os.ReadFile(inputPath)
	if err != nil {
		panic("failed to read input.txt")
	}
	content := string(contentBytes)
	lines := strings.Split(strings.TrimSpace(content), "\n")
	var docs []string
	for _, l := range lines {
		l = strings.TrimSpace(l)
		if l != "" {
			docs = append(docs, l)
		}
	}

	rng := NewRng(42) // Let there be order among chaos
	rng.Shuffle(docs)
	fmt.Printf("num docs: %d\n", len(docs))

	// Let there be a Tokenizer to translate strings to discrete symbols and back
	charSet := make(map[rune]bool) // unique characters in the dataset become token ids 0..n-1
	for _, doc := range docs {
		for _, ch := range doc {
			charSet[ch] = true
		}
	}
	var uchars []rune
	for ch := range charSet {
		uchars = append(uchars, ch)
	}
	sort.Slice(uchars, func(i, j int) bool { return uchars[i] < uchars[j] })
	bos := len(uchars)           // token id for the special Beginning of Sequence (BOS) token
	vocabSize := len(uchars) + 1 // total number of unique tokens, +1 is for BOS
	fmt.Printf("vocab size: %d\n", vocabSize)

	charToID := func(ch rune) int {
		for i, c := range uchars {
			if c == ch {
				return i
			}
		}
		panic("char not found")
	}

	// Initialize the parameters, to store the knowledge of the model.
	nEmbd := 16              // embedding dimension
	nHead := 4               // number of attention heads
	nLayer := 1              // number of layers
	blockSize := 16          // maximum sequence length
	headDim := nEmbd / nHead // dimension of each head

	// Persistent tape just for parameters
	paramTape := NewTape()
	sd := NewStateDict()

	matrix := func(tape *Tape, rng *Rng, nout, nin int, std float64) [][]ValIdx {
		mat := make([][]ValIdx, nout)
		for i := 0; i < nout; i++ {
			mat[i] = make([]ValIdx, nin)
			for j := 0; j < nin; j++ {
				mat[i][j] = tape.NewVal(rng.Gauss(0.0, std))
			}
		}
		return mat
	}

	sd.Insert("wte", matrix(paramTape, rng, vocabSize, nEmbd, 0.08))
	sd.Insert("wpe", matrix(paramTape, rng, blockSize, nEmbd, 0.08))
	sd.Insert("lm_head", matrix(paramTape, rng, vocabSize, nEmbd, 0.08))

	for i := 0; i < nLayer; i++ {
		sd.Insert(fmt.Sprintf("layer%d.attn_wq", i), matrix(paramTape, rng, nEmbd, nEmbd, 0.08))
		sd.Insert(fmt.Sprintf("layer%d.attn_wk", i), matrix(paramTape, rng, nEmbd, nEmbd, 0.08))
		sd.Insert(fmt.Sprintf("layer%d.attn_wv", i), matrix(paramTape, rng, nEmbd, nEmbd, 0.08))
		sd.Insert(fmt.Sprintf("layer%d.attn_wo", i), matrix(paramTape, rng, nEmbd, nEmbd, 0.08))
		sd.Insert(fmt.Sprintf("layer%d.mlp_fc1", i), matrix(paramTape, rng, 4*nEmbd, nEmbd, 0.08))
		sd.Insert(fmt.Sprintf("layer%d.mlp_fc2", i), matrix(paramTape, rng, nEmbd, 4*nEmbd, 0.08))
	}

	// Flatten params into a single list
	var paramIndices []ValIdx
	for _, e := range sd.entries {
		for _, row := range e.val {
			paramIndices = append(paramIndices, row...)
		}
	}
	numParams := len(paramIndices)
	fmt.Printf("num params: %d\n", numParams)

	// Let there be Adam, the blessed optimizer and its buffers
	learningRate := 0.01
	beta1 := 0.85
	beta2 := 0.99
	epsAdam := 1e-8
	mBuf := make([]float64, numParams) // first moment buffer
	vBuf := make([]float64, numParams) // second moment buffer

	// Repeat in sequence
	numSteps := 1000 // number of training steps
	for step := 0; step < numSteps; step++ {
		// Take single document, tokenize it, surround it with BOS special token on both sides
		doc := docs[step%len(docs)]
		tokens := make([]int, 0, len(doc)+2)
		tokens = append(tokens, bos)
		for _, ch := range doc {
			tokens = append(tokens, charToID(ch))
		}
		tokens = append(tokens, bos)
		n := blockSize
		if len(tokens)-1 < n {
			n = len(tokens) - 1
		}

		// Create a fresh tape for this step, but copy param data from paramTape
		tape := NewTape()
		// Re-create param nodes in the new tape with current data values
		stepSD := NewStateDict()
		for _, e := range sd.entries {
			newMat := make([][]ValIdx, len(e.val))
			for i, row := range e.val {
				newMat[i] = make([]ValIdx, len(row))
				for j, pidx := range row {
					newMat[i][j] = tape.NewVal(paramTape.Data(pidx))
				}
			}
			stepSD.Insert(e.key, newMat)
		}

		// Map from stepSD param indices back to paramTape indices
		var stepParamIndices []ValIdx
		for _, e := range stepSD.entries {
			for _, row := range e.val {
				stepParamIndices = append(stepParamIndices, row...)
			}
		}

		// Forward the token sequence through the model, building up the computation graph all the way to the loss.
		keysCache := make([][][]ValIdx, nLayer)
		valuesCache := make([][][]ValIdx, nLayer)
		for i := 0; i < nLayer; i++ {
			keysCache[i] = [][]ValIdx{}
			valuesCache[i] = [][]ValIdx{}
		}
		var losses []ValIdx

		for posID := 0; posID < n; posID++ {
			tokenID := tokens[posID]
			targetID := tokens[posID+1]
			logits := gpt(
				tape,
				stepSD,
				tokenID,
				posID,
				keysCache,
				valuesCache,
				nLayer,
				nHead,
				headDim,
				nEmbd,
			)
			probs := softmax(tape, logits)
			logP := tape.Log(probs[targetID])
			negLogP := tape.Neg(logP)
			losses = append(losses, negLogP)
		}

		// final average loss over the document sequence. May yours be low.
		totalLoss := losses[0]
		for i := 1; i < len(losses); i++ {
			totalLoss = tape.Add(totalLoss, losses[i])
		}
		loss := tape.MulConst(totalLoss, 1.0/float64(n))

		// Backward the loss, calculating the gradients with respect to all model parameters.
		tape.Backward(loss)

		lossVal := tape.Data(loss)

		// Adam optimizer update: update the model parameters based on the corresponding gradients.
		lrT := learningRate * (1.0 - float64(step)/float64(numSteps)) // linear learning rate decay
		for i := 0; i < len(stepParamIndices); i++ {
			stepPidx := stepParamIndices[i]
			origPidx := paramIndices[i]
			g := tape.Grad(stepPidx)
			mBuf[i] = beta1*mBuf[i] + (1.0-beta1)*g
			vBuf[i] = beta2*vBuf[i] + (1.0-beta2)*g*g
			mHat := mBuf[i] / (1.0 - math.Pow(beta1, float64(step+1)))
			vHat := vBuf[i] / (1.0 - math.Pow(beta2, float64(step+1)))
			newData := paramTape.Data(origPidx) - lrT*mHat/(math.Sqrt(vHat)+epsAdam)
			paramTape.SetData(origPidx, newData)
		}

		fmt.Printf("step %4d / %4d | loss %.4f\n", step+1, numSteps, lossVal)
	}

	// Inference: may the model babble back to us
	temperature := 0.5 // in (0, 1], control the "creativity" of generated text, low to high
	fmt.Println("\n--- inference (new, hallucinated names) ---")
	for sampleIdx := 0; sampleIdx < 20; sampleIdx++ {
		// Build a fresh tape with current params for inference
		tape := NewTape()
		infSD := NewStateDict()
		for _, e := range sd.entries {
			newMat := make([][]ValIdx, len(e.val))
			for i, row := range e.val {
				newMat[i] = make([]ValIdx, len(row))
				for j, pidx := range row {
					newMat[i][j] = tape.NewVal(paramTape.Data(pidx))
				}
			}
			infSD.Insert(e.key, newMat)
		}

		keysCache := make([][][]ValIdx, nLayer)
		valuesCache := make([][][]ValIdx, nLayer)
		for i := 0; i < nLayer; i++ {
			keysCache[i] = [][]ValIdx{}
			valuesCache[i] = [][]ValIdx{}
		}
		tokenID := bos
		var sample []rune

		for posID := 0; posID < blockSize; posID++ {
			logits := gpt(
				tape,
				infSD,
				tokenID,
				posID,
				keysCache,
				valuesCache,
				nLayer,
				nHead,
				headDim,
				nEmbd,
			)
			// Apply temperature
			scaledLogits := make([]ValIdx, len(logits))
			for i, l := range logits {
				scaledLogits[i] = tape.MulConst(l, 1.0/temperature)
			}
			probs := softmax(tape, scaledLogits)
			weights := make([]float64, len(probs))
			for i, p := range probs {
				weights[i] = tape.Data(p)
			}
			tokenID = rng.Choices(weights)
			if tokenID == bos {
				break
			}
			sample = append(sample, uchars[tokenID])
		}
		fmt.Printf("sample %2d: %s\n", sampleIdx+1, string(sample))
	}
}
