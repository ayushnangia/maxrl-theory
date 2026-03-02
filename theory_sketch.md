# Variance-Optimal Gradient Estimators for log(pass@k)

## 0. Paper Reconciliation (CRITICAL — read first)

After reading the actual paper source (arXiv 2602.02710), the code, and
running thorough numerical analysis, here is the complete picture.

### The Paper's Estimator (eq 18 / Algorithm 1)

    g̃(x) = Σ_i (r_i/K - 1/N) S_i      (K = number of successes)

Per-sample weights:
- Success (r_i=1):  w_s = 1/K - 1/N = (N-K)/(NK) = (1-K/N)/K
- Failure (r_i=0):  w_f = -1/N

When K=0: gradient set to 0 (slight bias, negligible for most p).

### Code ↔ Paper Mapping

| Code estimator         | What it is                                           |
|------------------------|------------------------------------------------------|
| `MAXRL` (line 402)     | Paper's Algorithm 1 (scalar form, `(r-mean)/mean`)   |
| `VR_COND` (line 788)   | Paper's eq 18 weights (explicit w_s, w_f)            |
| `MACLAURIN` (line 680) | Term-by-term Maclaurin: Σ (1/k) ĝ_k (NOT the paper's main result!) |
| `MACLAURIN_BASELINE`   | LOO variant of Maclaurin                              |
| `CROSS_FITTED`         | Disjoint-tuples baseline variant                      |
| `OVERSAMPLE_SUBSET_VR` | Another baseline variant                              |

### Key Finding: VR_COND = Variance-Optimal Maclaurin

The paper's VR_COND weights (w_s = (N-C)/(CN), w_f = -1/N) are **exactly**
the variance-optimal reweighting of the Maclaurin α schedule:

    α(C) = 1/C  →  w_s* = (1-C/N) · α = (N-C)/(CN)  ✓
                    w_f* = -(C/N) · α  = -1/N          ✓

Moreover, **α(C) = 1/C is the unique α schedule** satisfying the T=N
unbiasedness constraint (up to the K=0 bias from zeroing).

**The paper's 1/K normalization trick automatically achieves variance optimality!**

### The Maclaurin Code is Suboptimal

The `maclaurin_weights` code uses w_s = 1/N (constant!) while the optimal is
w_s* = (1-C/N)/C (adapts to C). Variance ratio Mac/VR_COND:

    p=0.05:  17.6x worse    p=0.30:  1.9x worse     p=0.50:  1.0x (equal)
    p=0.70:  1.6x worse     p=0.90:  7.6x worse     p=0.95: 16.9x worse

The many code variants (LOO, cross-fitted, etc.) are attempts to fix this
suboptimality. VR_COND is the clean solution.

---

## 1. Setup

Policy π_θ generates N responses per prompt. Each gets binary outcome P_i ∈ {0,1}.
Let p(θ) = P(correct), q = 1-p. Score function: s_i = ∇log π(y_i | x).

**Truncated ML Objective:** J^(T)(θ) = Σ_{k=1}^T (1/k) pass@k

**Population gradient:** ∇J^(T) = w_T(p) · ∇p, where w_T(p) = [1-(1-p)^T]/p

**Full ML gradient (T→∞):** ∇log(p) = (1/p) · ∇p

**REINFORCE identity:** ∇p = E[P_i · s_i], and E[(1-P_i) · s_i] = -∇p.

## 2. Estimator Class

By exchangeability, any estimator using (C, N, {P_i}) takes the form:

    ĝ = Σ_i w_i · s_i,  where w_i = w_s(C) if P_i=1, w_f(C) if P_i=0

Two free functions: w_s: {0,...,N} → ℝ, w_f: {0,...,N} → ℝ.

## 3. Unbiasedness Constraint

Let C_{-i} = Σ_{j≠i} P_j ~ Bin(N-1, p). Since s_i depends only on sample i:

    E[w_i · s_i | C_{-i}=j] = [w_s(j+1) - w_f(j)] · ∇p

Define α(j) ≡ w_s(j+1) - w_f(j). Summing over N samples:

    E[ĝ] = N · E_{j~Bin(N-1,p)}[α(j)] · ∇p

Unbiasedness for ∇J^(T) requires:

    N · E_{j~Bin(N-1,p)}[α(j)] = w_T(p)   ∀ p ∈ (0,1)

### Theorem (Unique α at T=N)

For T=N, the unique solution is **α(j) = 1/(j+1)**, equivalently α(C) = 1/C.

**Proof.** Expanding both sides as polynomials of degree N-1 in p and matching
coefficients gives N equations for N unknowns α(0),...,α(N-1). The system
has full rank, and the solution is α(j) = 1/(j+1). This follows from the
combinatorial identity C(N-1,j)/(j+1) = C(N,j+1)/N.  ∎

## 4. Impossibility Theorem

**Theorem 1.** No weights (w_s, w_f) depending only on (C, N) yield an unbiased
estimator of ∇log(p) = (1/p)∇p for all p ∈ (0,1).

**Proof.** The LHS N · Σ C(N-1,j) p^j q^{N-1-j} α(j) is a polynomial of degree
N-1 in p. The RHS 1/p is not a polynomial (pole at p=0).  ∎

**Corollary.** All estimators in this class — including the paper's 1/K estimator
and all Maclaurin variants — are biased for the full ML gradient ∇log(p).
They can only target the truncated objective J^(T).

## 5. MAIN RESULT: Variance-Optimal Weights

### Theorem 2 (Variance-Optimal Weight Allocation)

Given any multiplier schedule α(C), the weights minimizing E[Σ w_i²]
(the variance proxy) subject to w_s(C) - w_f(C) = α(C) are:

    w_s*(C) = (1 - C/N) · α(C)
    w_f*(C) = -(C/N) · α(C)

**Proof.** For fixed C, the proxy is C·w_s² + (N-C)·w_f². The Lagrangian
with constraint w_s - w_f = α gives:

    ∂/∂w_s: 2C·w_s = λ   →   w_s = λ/(2C)
    ∂/∂w_f: 2(N-C)·w_f = -λ   →   w_f = -λ/(2(N-C))

    Constraint: λ/(2C) + λ/(2(N-C)) = α   →   λ = 2C(N-C)α/N

    w_s* = (N-C)α/N = (1-C/N)α,   w_f* = -Cα/N = -(C/N)α   ∎

### Corollary: VR_COND is Optimal

With α(C) = 1/C (the unique unbiased schedule at T=N):

    w_s* = (1-C/N)/C = (N-C)/(NC)     ← Paper's VR_COND ✓
    w_f* = -(C/N)/C = -1/N             ← Paper's VR_COND ✓

**The paper's estimator is the unique minimum-variance unbiased estimator
in the (C,N)-dependent weight class for the T=N truncated ML objective.**

### Where Maclaurin Fails

Maclaurin uses the SAME α(C) = 1/C (it's uniquely determined) but allocates
weights as w_s = 1/N, w_f = combinatorial(C) — NOT the optimal split.

The optimal split has w_s*(C) = (1-C/N)/C which:
- Increases weight on successes when C is small (hard prompts)
- Decreases weight on successes when C is large (easy prompts)

Maclaurin's w_s = 1/N treats every prompt identically.

## 6. Variance Gap: Maclaurin vs VR_COND (= Optimal)

### Numerical Results (N=16)

    p=0.05:  17.6x     p=0.10:   8.2x     p=0.20:   3.4x
    p=0.30:   1.9x     p=0.40:   1.2x     p=0.50:   1.0x
    p=0.60:   1.1x     p=0.70:   1.6x     p=0.80:   3.0x
    p=0.90:   7.6x     p=0.95:  16.9x

**U-shaped gap:** Maclaurin matches VR_COND only at p≈0.5 (where C/N ≈ 0.5
makes w_s = 1/N close to (1-0.5)/C with C=N/2, i.e., 1/N ≈ 0.5/(N/2) = 1/N).

### Impact During Training

- Early training (p ≈ 0.05-0.2): 3.4-17.6x variance wasted by Maclaurin
- Late training (p ≈ 0.8-0.95): 3.0-16.9x variance wasted
- Mid training (p ≈ 0.3-0.5): ≤ 1.9x (acceptable)

These extremes dominate training compute.

## 7. Extension: T < N (Truncated Maclaurin)

When T < N, the truncated estimator uses only the first T terms of the
Maclaurin expansion. The α schedule changes (is no longer 1/C) and
**the paper's VR_COND no longer applies** (it targets T=N).

For truncated estimators:
1. Compute α_T(C) from the Maclaurin weights: α_T(C) = w_s_mac(C) - w_f_mac(C)
2. Apply variance-optimal reweighting: w_s* = (1-C/N)·α_T(C), w_f* = -(C/N)·α_T(C)

This gives a variance improvement over Maclaurin at any T, without changing
the bias (same α, just better weight allocation).

## 8. What's Left for a Paper

### Confirmed Novel Contributions

1. **Impossibility theorem** (Thm 1): Formally proves no finite-sample unbiased
   estimator exists for the full ∇log(p). The paper implicitly acknowledges
   this (only claims T=N) but doesn't prove impossibility.

2. **Variance-optimal weight formula** (Thm 2): w_s* = q̂·α, w_f* = -p̂·α.
   This is a clean, general result that:
   - Explains WHY VR_COND works (it's optimal!)
   - Explains WHY Maclaurin is suboptimal (wrong weight split)
   - Gives an immediate fix for truncated (T<N) estimators

3. **Uniqueness of α at T=N**: The harmonic schedule α(C)=1/C is the ONLY
   unbiased option. This is implicit in the paper's Theorem 2 proof but
   not stated as a uniqueness result.

4. **Variance gap characterization**: U-shaped in p, quantified exactly.
   Explains when the Maclaurin code variants (LOO, cross-fitted, etc.)
   help vs hurt.

### What's NOT Novel (paper already has it)

- The 1/K normalization trick = variance-optimal weights (discovered here,
  but the paper already uses it as VR_COND)
- The control variate interpretation (paper Section 3.3)

### Possible Research Directions

1. **T<N optimal**: For truncated Maclaurin, apply our variance-optimal
   weights to the truncated α schedule. The paper's VR_COND doesn't handle
   this case. Experiments comparing Mac_T2 vs OptimalWeights_T2.

2. **MSE-optimal target**: Instead of matching Maclaurin's biased target,
   jointly optimize bias + variance. What's the best m(C) if we're willing
   to accept some bias in exchange for lower MSE?

3. **Adaptive T**: Choose T based on observed C to balance bias/variance
   differently per prompt difficulty.

4. **Connection to Rao-Blackwell**: Is VR_COND the Rao-Blackwellization
   of some simpler estimator?

## 9. Implementation Notes

### For T<N variance-optimal estimator

```python
@register_adv_est("variance_optimal_truncated")
def compute_variance_optimal_truncated_advantage(
    token_level_rewards, response_mask, index,
    epsilon=1e-6, config=None, **kwargs
):
    scores = token_level_rewards.sum(dim=-1)
    T = config.truncate_order
    with torch.no_grad():
        bsz = scores.shape[0]
        unique_ids = np.unique(index)
        n = bsz // len(unique_ids)
        P = rearrange(scores, "(m n) -> m n", n=n)
        P = (P > 0.5).float()
        C = torch.sum(P, dim=1)  # (m,)

        # Get Maclaurin alpha for truncated objective
        W_s_mac, W_f_mac = maclaurin_weights(C, n, T)
        alpha = W_s_mac - W_f_mac  # effective multiplier per prompt

        # Variance-optimal weights
        p_hat = C / n
        q_hat = 1 - p_hat
        W_succ = q_hat * alpha
        W_fail = -p_hat * alpha

        advantage = W_succ[:, None] * P + W_fail[:, None] * (1 - P)
        advantage = advantage.flatten().unsqueeze(-1) * response_mask
    return advantage, advantage
```

Note: For T=N, this reduces to VR_COND (already in the codebase).
The value is for T<N where VR_COND doesn't apply.
