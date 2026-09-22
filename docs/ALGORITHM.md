# Formal specification of the adaptive loop

The examination board asked for two things that belong together: a
pseudocode or flowchart of the Active Learning pipeline (Sec. 5 of the
revision plan), and greater mathematical formalisation of the adaptive
mechanism. This document supplies both, in a form that can be pasted
into Chapter 4 with only the label prefixes changed.

It is also the specification that `src/al_loop.py` implements. When the
code and this document disagree, one of them is a bug; the line
references below are the map between them.

---

## 1. Notation

| Symbol | Meaning | Where in code |
|---|---|---|
| $\mathcal{L}_k$ | labelled set after cycle $k$; $\|\mathcal{L}_0\| = L_0$ | `labelled` |
| $\mathcal{U}$ | unlabelled pool, released progressively | `u_pool` |
| $\mathcal{C}_k \subseteq \mathcal{U}$ | candidate batch released at cycle $k$ | `batch_indices` |
| $\mathcal{T}$ | held-out test set, fixed across all runs | `splits["T"]` |
| $\mathcal{M}_k$ | detector after the $k$-th update | `trainer.model` |
| $b$ | annotation budget per cycle | `cfg.budget` |
| $K$ | number of adaptation cycles | `cfg.cycles` |
| $T$ | stochastic forward passes for MC Dropout | `cfg.T` |
| $\alpha$ | retained fraction of the uncertainty threshold | `cfg.alpha` |
| $\beta$ | cluster-size divisor, $K_{\text{cl}} = \min(b, \lfloor\|\mathcal{C}_k\|/\beta\rfloor)$ | `cfg.beta` |
| $\tau_k$ | adaptive uncertainty threshold at cycle $k$ | `apply_uncertainty_prefilter` |
| $\tau_{\text{shift}}$ | shift-detection threshold, calibrated on $\mathcal{L}_0$ | `trigger.tau_shift` |
| $n_{\min}$ | accumulation trigger | `cfg.n_min` |
| $W$ | moving-average window for shift detection | `cfg.shift_window` |

---

## 2. The acquisition function

### 2.1 Epistemic uncertainty by MC Dropout

MC Dropout is an **approximate** Bayesian method: dropout at inference
time draws from a variational distribution $q(\theta)$ that approximates
the true posterior $p(\theta \mid \mathcal{L}_k)$, not from the posterior
itself. Exact alternatives (MCMC, HMC) are prohibitive at the edge, which
is the reason for the approximation and the reason it must be named as
one throughout the text.

For an image $x$ with $T$ stochastic passes producing per-pass detection
confidences $\hat{p}_t(x)$:

$$
\mu(x) = \frac{1}{T}\sum_{t=1}^{T} \hat{p}_t(x), \qquad
\sigma^2(x) = \frac{1}{T}\sum_{t=1}^{T}\bigl(\hat{p}_t(x) - \mu(x)\bigr)^2
$$

### 2.2 BALD

BALD is the mutual information between the prediction and the model
parameters — the part of the uncertainty that more labelled data can
reduce, as opposed to irreducible observation noise:

$$
\mathrm{BALD}(x) \;=\; \underbrace{\mathbb{H}\bigl[\mathbb{E}_{q(\theta)}[\hat{p}]\bigr]}_{\text{total}}
\;-\; \underbrace{\mathbb{E}_{q(\theta)}\bigl[\mathbb{H}[\hat{p}]\bigr]}_{\text{aleatoric}}
$$

with $\mathbb{H}[p] = -p\log p - (1-p)\log(1-p)$ over the image-level
detection confidence. Implemented in
`MCDropoutEstimator._compute_image_metrics`.

### 2.3 Adaptive threshold

Only the most uncertain fraction of the candidate batch is considered for
annotation:

$$
\tau_k = \mathrm{Quantile}_{1-\alpha_k}\bigl(\{\sigma^2(x) : x \in \mathcal{C}_k\}\bigr),
\qquad
\alpha_k = \max\Bigl(\alpha,\ \frac{b}{|\mathcal{C}_k|}\Bigr)
$$

$$
\mathcal{C}_k^{\text{filt}} = \{x \in \mathcal{C}_k : \sigma^2(x) \ge \tau_k\}
$$

The $\alpha_k$ floor is the correction to a defect in the original
implementation. With a fixed $\alpha = 0.1$ and $|\mathcal{C}_k| \approx
350$, the filter retained about 35 candidates against a budget of
$b = 50$; the code then discarded the filter as unusable, so the
documented two-stage selection silently never executed. Making the
retained fraction at least $b/|\mathcal{C}_k|$ keeps the threshold's
meaning while guaranteeing the budget can be funded.

### 2.4 Diversity-aware selection

Cluster the filtered candidates in embedding space and allocate the
budget proportionally to cluster size, taking the highest-BALD member of
each cluster first:

$$
K_{\text{cl}} = \min\Bigl(b,\ \Bigl\lfloor \tfrac{|\mathcal{C}_k^{\text{filt}}|}{\beta} \Bigr\rfloor\Bigr),
\qquad
\{G_1,\dots,G_{K_{\text{cl}}}\} = \text{$k$-means}\bigl(\phi(\mathcal{C}_k^{\text{filt}})\bigr)
$$

$$
b_j = \Bigl\lfloor b \cdot \frac{|G_j|}{|\mathcal{C}_k^{\text{filt}}|} \Bigr\rfloor,
\qquad
\mathcal{S}_k = \bigcup_{j=1}^{K_{\text{cl}}} \operatorname*{arg\,top-}b_j\bigl\{\mathrm{BALD}(x) : x \in G_j\bigr\}
$$

Residual budget from the flooring is assigned greedily to the clusters
with the highest maximum BALD score, so the budget is always exhausted.

---

## 3. The update trigger

The loop does not retrain every cycle. It updates when either condition
fires — accumulation, or detected shift:

$$
\text{update at cycle } k \iff
\underbrace{n_k \ge n_{\min}}_{\text{(A) accumulation}}
\;\;\lor\;\;
\underbrace{\frac{1}{W}\sum_{j=k-W+1}^{k} \bar{\sigma}^2_j > \tau_{\text{shift}}}_{\text{(B) shift detection}}
$$

where $n_k$ is the number of annotations accumulated since the last
update, $\bar{\sigma}^2_j$ is the mean predictive variance of batch $j$,
and $\tau_{\text{shift}} = \mathrm{Percentile}_{90}$ of the variance
distribution measured on $\mathcal{L}_0$ before deployment.

Condition (A) bounds the staleness of the model by the annotation rate.
Condition (B) exists because staleness is not the only failure mode: an
abrupt change in scene statistics can invalidate the model long before
$n_{\min}$ annotations have accumulated, and the rising uncertainty is
the only signal available in-flight, since labels are not.

**The honest limitation.** The mechanism is a triggered update policy, not
closed-loop control in the systems sense: there is no plant model, no
setpoint and no stability guarantee, and uncertainty steers *labelling*
rather than any actuator. The thesis text should describe it as an
adaptive continual-learning pipeline and reserve control-theoretic
vocabulary for work that earns it — which is the board's point in
Sec. 3 of the revision plan.

---

## 4. Algorithm 1

```
Algorithm 1: Uncertainty-guided Active Learning loop for UAV-edge adaptation

Input : initial labelled set L0, unlabelled pool U, test set T,
        budget b, cycles K, passes T_mc, threshold fraction alpha,
        cluster divisor beta, accumulation trigger n_min, window W
Output: adapted model M_K, trajectory of metrics

 1  M_0   <- TRAIN(L0)                             // pre-deployment
 2  tau_shift <- Percentile_90({ sigma^2(x) : x in L0 })   // calibrate on known data
 3  n <- 0 ; history <- []
 4  EVALUATE(M_0, T)                               // cycle 0 reference
 5
 6  for k = 1 .. K do
 7      // -- Stage 1: acquisition -------------------------------------
 8      C_k <- RELEASE(U, k)                       // progressive pool release
 9      if C_k = empty then break
10
11      // -- Stage 2: inference and uncertainty filtering -------------
12      for x in C_k do
13          {p_1..p_T} <- T_mc stochastic passes of M_{k-1} on x
14          sigma2(x)  <- Var_t(p_t)
15          BALD(x)    <- H[mean_t p_t] - mean_t H[p_t]
16      history <- history + [ mean_x sigma2(x) ]
17      alpha_k <- max(alpha, b / |C_k|)
18      tau_k   <- Quantile_{1-alpha_k}({ sigma2(x) })
19      C_filt  <- { x in C_k : sigma2(x) >= tau_k }
20
21      // -- Stage 3: diversity-aware supervision ---------------------
22      K_cl <- min(b, floor(|C_filt| / beta))
23      G_1..G_Kcl <- k-means(phi(C_filt), K_cl)    // phi = penultimate features
24      S_k  <- empty
25      for j = 1 .. K_cl do
26          b_j <- floor(b * |G_j| / |C_filt|)
27          S_k <- S_k + top-b_j of G_j by BALD
28      S_k <- S_k + greedy fill to exactly b samples
29      Y_k <- ORACLE_ANNOTATE(S_k)                 // human in the loop
30      L_k <- L_{k-1} + (S_k, Y_k) ; U <- U \ S_k ; n <- n + |S_k|
31
32      // -- Stage 4: conditional incremental update ------------------
33      shift <- mean(history[-W:]) > tau_shift
34      if n >= n_min or shift or k = K then
35          M_k <- FINE_TUNE(M_{k-1}, L_k, epochs=E, lr=lr_inc)
36          n <- 0
37      else
38          M_k <- M_{k-1}
39
40      EVALUATE(M_k, T)                            // fixed test set, every cycle
41  end for
42
43  return M_K, trajectory
```

### LaTeX version

Paste into Chapter 4 with `\usepackage[ruled,vlined]{algorithm2e}`:

```latex
\begin{algorithm}[ht]
\SetAlgoLined
\KwIn{$\mathcal{L}_0$, $\mathcal{U}$, $\mathcal{T}$, budget $b$, cycles $K$,
      passes $T$, fraction $\alpha$, divisor $\beta$, trigger $n_{\min}$, window $W$}
\KwOut{adapted model $\mathcal{M}_K$ and metric trajectory}
$\mathcal{M}_0 \leftarrow \textsc{Train}(\mathcal{L}_0)$\;
$\tau_{\text{shift}} \leftarrow \mathrm{Percentile}_{90}\{\sigma^2(x): x \in \mathcal{L}_0\}$\;
$n \leftarrow 0$; $\mathcal{H} \leftarrow \emptyset$\;
\For{$k \leftarrow 1$ \KwTo $K$}{
  $\mathcal{C}_k \leftarrow \textsc{Release}(\mathcal{U}, k)$\;
  \lIf{$\mathcal{C}_k = \emptyset$}{\textbf{break}}
  \ForEach{$x \in \mathcal{C}_k$}{
    $\sigma^2(x) \leftarrow \mathrm{Var}_t\,\hat p_t(x)$;\quad
    $\mathrm{BALD}(x) \leftarrow \mathbb{H}[\bar p] - \overline{\mathbb{H}[p]}$\;
  }
  $\mathcal{H} \leftarrow \mathcal{H} \cup \{\overline{\sigma^2}\}$;\quad
  $\alpha_k \leftarrow \max(\alpha, b/|\mathcal{C}_k|)$\;
  $\tau_k \leftarrow \mathrm{Quantile}_{1-\alpha_k}\{\sigma^2(x)\}$;\quad
  $\mathcal{C}^{\text{filt}} \leftarrow \{x : \sigma^2(x) \ge \tau_k\}$\;
  $K_{\text{cl}} \leftarrow \min(b, \lfloor |\mathcal{C}^{\text{filt}}|/\beta \rfloor)$;\quad
  $\{G_j\} \leftarrow \text{$k$-means}(\phi(\mathcal{C}^{\text{filt}}), K_{\text{cl}})$\;
  $\mathcal{S}_k \leftarrow \bigcup_j \operatorname{top-}b_j\{\mathrm{BALD}(x): x \in G_j\}$,\quad
  $b_j = \lfloor b|G_j|/|\mathcal{C}^{\text{filt}}| \rfloor$\;
  $\mathcal{L}_k \leftarrow \mathcal{L}_{k-1} \cup \textsc{Annotate}(\mathcal{S}_k)$;\quad
  $n \leftarrow n + |\mathcal{S}_k|$\;
  \eIf{$n \ge n_{\min}$ \textbf{\emph{or}} $\overline{\mathcal{H}_{-W:}} > \tau_{\text{shift}}$
       \textbf{\emph{or}} $k = K$}{
    $\mathcal{M}_k \leftarrow \textsc{FineTune}(\mathcal{M}_{k-1}, \mathcal{L}_k)$;\quad $n \leftarrow 0$\;
  }{
    $\mathcal{M}_k \leftarrow \mathcal{M}_{k-1}$\;
  }
  $\textsc{Evaluate}(\mathcal{M}_k, \mathcal{T})$\;
}
\caption{Uncertainty-guided Active Learning loop for UAV--edge adaptation}
\label{alg:al_loop}
\end{algorithm}
```

---

## 5. Complexity and the edge-native argument

Per cycle, with $|\mathcal{C}|$ candidates, $d$-dimensional embeddings and
$E$ fine-tuning epochs over $|\mathcal{L}_k|$ images:

| Stage | Cost | Where it runs |
|---|---|---|
| Deterministic inference | $O(|\mathcal{C}|)$ forward passes | on the real-time path |
| MC Dropout scoring | $O(T \cdot |\mathcal{C}|)$ forward passes | off the real-time path |
| Feature extraction | $O(|\mathcal{C}|)$ forward passes, reuses the backbone | off the path |
| $k$-means | $O(I \cdot K_{\text{cl}} \cdot |\mathcal{C}^{\text{filt}}| \cdot d)$ | off the path, negligible |
| Fine-tuning | $O(E \cdot |\mathcal{L}_k|)$ | background, between passes |

Contribution C3 claims the pipeline is *edge-native*. The board asked what
specifically makes it so, and the defensible answer is this separation:
the deployed detector is a single deterministic forward pass whose
latency is independent of $T$, while every expensive component — the $T$
stochastic passes, clustering, and fine-tuning — runs off the real-time
path between acquisition windows. The claim is therefore about **where
the cost lands in the duty cycle**, not about the total cost being small.
Block `E6_operational_profiling` measures both sides separately so the
claim can be checked rather than asserted; anything that cannot be
demonstrated from those measurements should be removed from C3's wording.
