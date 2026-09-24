# Systems & Inference Engineering Notes
> *First-principles breakdowns of GPU computing, Softmax dynamics, Logit mechanics, and architectural bottlenecks.*

---

## 1. Softmax & The Significance of $e^x$ (Exponential)

Standard Softmax equation:
$$P_i = \frac{e^{x_i - m}}{\sum_{j=0}^{D-1} e^{x_j - m}} \quad \text{where } m = \max(\mathbf{x})$$

Why natural exponential $e^x$ and not absolute value $|x|$, square $x^2$, or linear scaling?

### A. Non-Negotiable Positivity ($e^x > 0$)
* Raw outputs from linear layers (logits) range across $(-\infty, +\infty)$ (e.g., $[-10.5, 0.0, 3.2]$).
* Probabilities strictly require:
  1. $P_i \ge 0$
  2. $\sum P_i = 1$
* $e^x > 0$ holds true for every real number $x \in \mathbb{R}$.
* *Why not $|x|$ or $x^2$?* Because $|-5| = 5$ and $(-5)^2 = 25$. They destroy ranking and sign information ($x_1 > x_2 \centernot\implies |x_1| > |x_2|$). $e^x$ preserves monotonicity strictly ($x_1 > x_2 \implies e^{x_1} > e^{x_2}$).

### B. "Soft" + "Max" (Contrast Amplification)
* We want `argmax` (the highest score wins), but raw `argmax` has zero derivatives everywhere (breaks backpropagation).
* $e^x$ is a smooth, differentiable approximation of `argmax`:
  * If $x_1 = 10$ and $x_2 = 8$, the linear difference is just $2$ ($25\%$).
  * $e^{10} \approx 22,026$ vs $e^8 \approx 2,981 \implies$ ratio jumps to **$7.4\times$**!
  * Result: $P(x_1) \approx 88\%$, $P(x_2) \approx 12\%$.
* The winner is exponentially separated, while losers still receive tiny non-zero probability and valid gradients.

### C. The Cancellation Magic in Backpropagation ($\frac{d}{dx} e^x = e^x$)
When Softmax is paired with Cross-Entropy Loss:
$$\text{Loss} = -\log(P_{\text{target}}) = -\log\left(\frac{e^{x_{\text{target}}}}{\sum e^{x_j}}\right) = -x_{\text{target}} + \log\left(\sum e^{x_j}\right)$$

* $\log$ and $e^x$ cancel out cleanly ($\log(e^z) = z$).
* Computing the gradient with respect to logit $x_i$ reduces to:
$$\frac{\partial \text{Loss}}{\partial x_i} = P_i - Y_i \quad (\text{Predicted Probability} - \text{Target})$$
* **GPU Impact:** No expensive roots, divisions, or transcendental derivatives during backward pass. It's just a single vector subtraction on hardware.

---

## 2. What are Logits & Why Do We Keep Them Unnormalized?

Logits are the raw, un-normalized outputs of the final Linear Layer (LM Head):
$$\text{Hidden State } [1 \times d_{\text{model}}] \times \text{LM Head Weights } [d_{\text{model}} \times V] = \text{Logits } [1 \times V]$$

* **The Mapping:** Index $i$ in the logit vector maps directly to Token ID $i$ in the Vocabulary. The value is the unconstrained confidence vote for that token.

### The 4 Inference Superpowers of Logits:

1. **Greedy Sampling Skip:**
   $$\text{argmax}(e^x) = \text{argmax}(x)$$
   When `temperature = 0`, **never run Softmax on the GPU**. Directly run `argmax` over the raw logits. You save $V$ exponentiations, $V$ additions, and $V$ divisions per token!
2. **Temperature Control ($T$):**
   $$\text{Logit}' = \frac{\text{Logit}}{T}$$
   Scaling logits before Softmax controls entropy without altering model weights.
3. **Repetition & Frequency Penalties:**
   $$\text{Logit}_{\text{token}} \leftarrow \text{Logit}_{\text{token}} - \alpha$$
   Directly subtract penalty from previously generated token logits.
4. **Constrained / Grammar Decoding (JSON Mode):**
   If a token violates grammar/schema at step $t$, set its logit directly to $-\infty$:
   $$\text{Logit}_{\text{invalid}} = -\infty \implies e^{-\infty} = 0 \implies P_{\text{invalid}} = 0\%$$

---

## 3. The Architecture Failure: "The Unembedding Paradox"

### The Dimensionality Explosion:
$$d_{\text{model}} = 4096 \xrightarrow{\text{LM Head}} |V| = 128,000 \xrightarrow{\text{Sampling}} 1 \text{ Token}$$

* A 4096-dim vector containing 32 layers of compressed reasoning is expanded into 128,000 floats (massive memory footprint & compute), only to throw away $99.999\%$ of the values to pick **one single integer token ID**.
* Information-theoretically, this is an absurdity inherited from 2017-era architectures.

### The "Static Ocean" Analogy:
* **The Claim:** The search space is treated like an infinite ocean where every point must be scanned.
* **The Reality:** Language has rigid syntactic, grammatical, and semantic locality.
  * If previous token = `"to"`, the next token domain is overwhelmingly restricted to verbs.
  * If context = Python code, $90\%$ of natural language vocabulary is statistically dead.
* **Why Industry Does It Anyway:** 
  * Dense GEMM ($[1 \times 4096] \times [4096 \times 128000]$) is a contiguous memory block that Tensor Cores chew through easily.
  * Researchers defaulted to brute-forcing the static ocean with GPU capital rather than designing structured candidate filtering.

### The True Positive & Calibration Crisis:
* For tokens ranked outside the top $\sim 20$, logits enter the long tail (values like $-18.4, -22.1$).
* In FP16/FP8, these tail values are indistinguishable from quantization noise.
* Computing 127,980 useless logits burns datacenter wattage, stresses memory bandwidth, and injects hallucination risk when temperature $> 0$.

---

## 4. The Systems Engineering Golden Rules

> **1. The Hardware Reality:**
> GPUs love continuous, dense, dumb matrix blocks. GPUs hate irregular branching (`if/else`), non-coalesced memory lookups, and ragged data structures. 
> *Doing unnecessary math densely is often faster than irregularly searching for what math to skip—unless sparsity is structured into hardware-aligned tiles (e.g., 2:4, block sparsity, MoE).*

> **2. The Efficiency Mandate:**
> *"Every single FLOP executed that was not required to determine the final correct token is an engineering failure."*

---

## 5. Architectural Frontiers to Watch & Build

1. **Approximate Nearest Neighbor (ANN) Unembedding:**
   Treat the LM Head not as a dense GEMM, but as a clustered vector search over 128k embeddings using LSH or hierarchical indexing.
2. **Grammar-Aware Early Masking:**
   Pruning candidate vocabulary sets prior to projection rather than post-hoc logit masking.
3. **Hardware-Aligned Sparsity (MoE, GQA, LayerSkip):**
   * **GQA:** Eliminating redundant KV heads (Llama 3 cuts 75% of KV heads).
   * **MoE:** Activating only small sub-networks per token (DeepSeek-V3: 671B total, 37B active; 94% of parameters asleep per step).
   * **Early Exiting:** Bypassing upper layers when token confidence is saturated early.
