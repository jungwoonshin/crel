# CREL: Centered Residual Energy Loss for Structured Prediction

## Abstract

We present Centered Residual Energy Loss (CREL), an energy-based method for
structured multi-label classification that addresses a fundamental limitation of
prior energy-based approaches: gradient redundancy with per-label supervision.
Starting from the precision matrix decomposition of Gaussian Markov Random
Fields, we derive an energy function that operates exclusively on centered label
residuals and models only off-diagonal pairwise coupling. The resulting energy
provides a training signal that is provably orthogonal to binary cross-entropy,
while admitting an O(Lr) computation via low-rank factorization of the coupling
matrix. We describe the complete method including the centering mechanism,
diagonal correction, cooperative training protocol, and inference-time
refinement.

---

## 1. Introduction

### 1.1 Problem Setting

Multi-label classification (MLC) assigns a binary label vector
$\mathbf{y} \in \{0, 1\}^L$ to an input $\mathbf{x} \in \mathbb{R}^D$. Labels
are typically correlated: the presence of one label shifts the conditional
probability of others. Standard approaches treat each label independently via
binary cross-entropy (BCE), discarding this correlation structure entirely.

Energy-based models (EBMs) offer a principled alternative. An energy function
$E_\theta(\mathbf{x}, \mathbf{y}): \mathbb{R}^D \times \{0,1\}^L \to
\mathbb{R}$ assigns a scalar score to each input-label pair. Ground-truth
configurations should receive higher energy than incorrect ones. At inference,
the predicted label vector is obtained by maximizing E over the label space.

### 1.2 Prior Work: SEAL

The Structured Energy-based model for Abstract Label spaces (SEAL) defines:

$$E_{\text{SEAL}}(\mathbf{x}, \mathbf{y}) = \underbrace{\sum_{i=1}^{L} y_i \cdot \mathbf{b}_i^\top g(\mathbf{x})}_{E_{\text{local}}} + \underbrace{\mathbf{v}^\top \operatorname{softplus}(\mathbf{M}\mathbf{y})}_{E_{\text{global}}}$$

where $g(\mathbf{x}) \in \mathbb{R}^d$ is a learned feature representation,
$\mathbf{b}_i$ are per-label weight vectors, $\mathbf{M} \in \mathbb{R}^{h
\times L}$, and $\mathbf{v} \in \mathbb{R}^h$. The local term provides
per-label, input-conditioned scores; the global term introduces a nonlinear
function over the full label vector.

### 1.3 The Gradient Redundancy Problem

The task network $F_\phi(\mathbf{x})$ is jointly trained with BCE:

$$\mathcal{L}_{\text{BCE}} = -\sum_{i=1}^{L} \left[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

The gradient of SEAL's local energy with respect to $y_i$ is:

$$\frac{\partial E_{\text{local}}}{\partial y_i} = \mathbf{b}_i^\top g(\mathbf{x})$$

This is a per-label, input-conditioned scalar — functionally identical in
structure to the logit that BCE already optimizes. The local term's gradient
carries no information about label $j \neq i$. The global term's gradient,

$$\frac{\partial E_{\text{global}}}{\partial y_i} = \sum_{k=1}^{h} v_k \cdot \sigma'([\mathbf{M}\mathbf{y}]_k) \cdot M_{ki}$$

does depend on the full label vector through $\mathbf{M}\mathbf{y}$, but this
dependency is implicit: there is no explicit parameterization of pairwise
coupling, no guarantee of alignment with the true conditional dependency
structure, and no mechanism preventing collapse to a per-label function.

We formalize this redundancy. Let $\mathbf{g}_E = \nabla_{\hat{\mathbf{y}}}
E(\mathbf{x}, \hat{\mathbf{y}})$ and $\mathbf{g}_B =
\nabla_{\hat{\mathbf{y}}} \mathcal{L}_{\text{BCE}}$. Define:

$$\rho = \frac{\mathbf{g}_E^\top \mathbf{g}_B}{\|\mathbf{g}_E\| \, \|\mathbf{g}_B\|}$$

For SEAL, empirically $\rho \gg 0$: the energy gradient is substantially
collinear with BCE. The energy network expends capacity re-learning marginal
label information rather than contributing complementary structural signal. An
ideal energy function would achieve $\rho \approx 0$.

---

## 2. Method

### 2.1 Motivating Decomposition

We derive the functional form of the energy from a Gaussian Markov Random Field
(GMRF) over the label vector conditioned on input $\mathbf{x}$:

$$p(\mathbf{y} \mid \mathbf{x}) \propto \exp\!\left(-\frac{1}{2}(\mathbf{y} - \boldsymbol{\mu}(\mathbf{x}))^\top \boldsymbol{\Lambda}(\mathbf{x})\,(\mathbf{y} - \boldsymbol{\mu}(\mathbf{x}))\right)$$

where $\boldsymbol{\mu}(\mathbf{x}) = \mathbb{E}[\mathbf{y} \mid \mathbf{x}]$
is the conditional mean and $\boldsymbol{\Lambda}(\mathbf{x}) =
\operatorname{Cov}(\mathbf{y} \mid \mathbf{x})^{-1}$ is the precision matrix.
The log-density is:

$$\log p(\mathbf{y} \mid \mathbf{x}) = -\frac{1}{2}\,\bar{\mathbf{y}}^\top \boldsymbol{\Lambda}\,\bar{\mathbf{y}} + \text{const}$$

where $\bar{\mathbf{y}} = \mathbf{y} - \boldsymbol{\mu}(\mathbf{x})$ is the
centered residual. Decomposing $\boldsymbol{\Lambda}$ into diagonal and
off-diagonal parts:

$$-\frac{1}{2}\,\bar{\mathbf{y}}^\top \boldsymbol{\Lambda}\,\bar{\mathbf{y}} = \underbrace{-\frac{1}{2}\sum_i \Lambda_{ii}\,\bar{y}_i^2}_{\text{per-label variance}} \;+\; \underbrace{-\frac{1}{2}\sum_{i \neq j} \Lambda_{ij}\,\bar{y}_i\,\bar{y}_j}_{\text{cross-label coupling}}$$

The diagonal term depends only on each label's individual deviation from its
marginal — exactly the signal that BCE provides. The off-diagonal term captures
pairwise conditional dependencies — exactly the structural information that BCE
cannot model.

**Design principle.** An energy function for structured prediction should model
only the off-diagonal term, leaving marginal correction to BCE. Realizing this
principle requires three mechanisms:

1. **Centering** to remove the marginal signal from the energy's input.
2. **Off-diagonal parameterization** to model only cross-label coupling.
3. **Low-rank factorization** for computational efficiency.

### 2.2 Centering via Exponential Moving Average

The conditional mean $\boldsymbol{\mu}(\mathbf{x})$ is unknown. We approximate
it by tracking the task network's predictions with a per-sample exponential
moving average (EMA). For training sample $n$ at step $t$:

$$\mu_n^{(t)} = \beta(t)\,\mu_n^{(t-1)} + (1 - \beta(t))\,F_\phi(\mathbf{x}_n)$$

with hard initialization $\mu_n^{(0)} = F_\phi(\mathbf{x}_n)$ on first
encounter. The EMA coefficient is annealed linearly during a warmup phase:

$$\beta(t) = \begin{cases} \beta_{\text{warm}} + \dfrac{t}{T_{\text{warm}}}\left(\beta_{\text{final}} - \beta_{\text{warm}}\right) & t < T_{\text{warm}} \\[6pt] \beta_{\text{final}} & t \geq T_{\text{warm}} \end{cases}$$

with $\beta_{\text{warm}} = 0.9$ and $\beta_{\text{final}} = 0.99$.

**Per-sample tracking.** Different inputs have different marginals. A global
mean would be a poor approximation for any individual sample. Per-sample
tracking yields $\mu_n \approx \mathbb{E}[F_\phi(\mathbf{x}_n)]$ under the
stationarity assumption that $F_\phi$ changes slowly relative to the EMA rate.

**Warmup annealing.** Early in training, $F_\phi$ changes rapidly, so a smaller
$\beta$ (faster adaptation) prevents stale centering targets. After warmup, a
larger $\beta$ provides smooth, low-variance targets.

The centered residual is:

$$\bar{\mathbf{y}} = F_\phi(\mathbf{x}) - \boldsymbol{\mu}(\mathbf{x})$$

This vector isolates the component of variation arising from label correlations
rather than per-label base rates.

### 2.3 Input-Conditioned Low-Rank Coupling

We parameterize the off-diagonal precision structure with a low-rank,
input-conditioned matrix. Define learnable label embeddings $\mathbf{e}_i \in
\mathbb{R}^{d_e}$ for $i = 1, \ldots, L$, and construct:

$$A(\mathbf{x}) = W_2 \cdot \operatorname{ReLU}\!\left(W_1^{\text{label}}\,\mathbf{E} + W_1^{\text{feat}}\,g(\mathbf{x})\right) \in \mathbb{R}^{L \times r}$$

where $\mathbf{E} \in \mathbb{R}^{L \times d_e}$ is the label embedding matrix,
$g(\mathbf{x}) \in \mathbb{R}^{d_x}$ is the feature network output, and
$r \ll L$ is the factorization rank. The implicit coupling matrix is:

$$Q(\mathbf{x}) = A(\mathbf{x})\,A(\mathbf{x})^\top \in \mathbb{R}^{L \times L}$$

Each entry $Q_{ij} = \mathbf{a}_i^\top \mathbf{a}_j$ measures the coupling
between labels $i$ and $j$ in the $r$-dimensional embedding space. The matrix
is positive semi-definite by construction and input-conditioned: label
correlations can vary by input.

**Split projection.** Computing $A$ naively requires concatenating label
embeddings $(L \times d_e)$ with feature vectors $(d_x)$ into a
$(B \times L \times (d_e + d_x))$ tensor. We exploit the additive structure of
the first linear layer to avoid this materialization:

$$\mathbf{h}_{\text{label}} = W_1^{\text{label}}\,\mathbf{E} \in \mathbb{R}^{L \times d_h} \qquad \mathbf{h}_{\text{feat}} = W_1^{\text{feat}}\,g(\mathbf{x}) \in \mathbb{R}^{B \times d_h}$$

$$\mathbf{h} = \operatorname{ReLU}\!\left(\mathbf{h}_{\text{label}} + \mathbf{h}_{\text{feat}}\right) \in \mathbb{R}^{B \times L \times d_h}$$

The label component is computed once and broadcast across the batch; the feature
component is computed once and broadcast across labels. No concatenation tensor
is formed.

**Rank selection.** The rank $r$ is determined by spectral analysis of the
empirical label covariance $\boldsymbol{\Sigma} = \frac{1}{N}\mathbf{Y}^\top
\mathbf{Y} - \boldsymbol{\bar{y}}\,\boldsymbol{\bar{y}}^\top$. We select the
smallest $r$ such that the top-$r$ eigenvalues explain at least 90% of total
variance:

$$r_{\text{eff}} = \min\!\left\{r : \frac{\sum_{k=1}^{r}\lambda_k}{\sum_{k=1}^{L}\lambda_k} \geq 0.9\right\}$$

subject to a practical cap $r \leq \min(L/4,\; 64)$. This ensures the
factorization has sufficient capacity for the dominant correlation modes while
maintaining the $r \ll L$ efficiency guarantee.

### 2.4 The O(Lr) Quadratic Energy

The quadratic form $\bar{\mathbf{y}}^\top Q\,\bar{\mathbf{y}}$ would require
$O(L^2)$ operations if computed directly. The low-rank factorization
$Q = AA^\top$ admits the identity:

$$\bar{\mathbf{y}}^\top (AA^\top)\,\bar{\mathbf{y}} = (A^\top \bar{\mathbf{y}})^\top (A^\top \bar{\mathbf{y}}) = \|A^\top \bar{\mathbf{y}}\|^2$$

Computed in two steps:

**Projection** ($O(Lr)$):

$$\mathbf{z} = A^\top \bar{\mathbf{y}} \in \mathbb{R}^r$$

**Squared norm** ($O(r)$):

$$\|\mathbf{z}\|^2 = \sum_{k=1}^{r} z_k^2$$

Total cost: $O(Lr + r) = O(Lr)$.

### 2.5 Diagonal Correction

The factorization $Q = AA^\top$ includes diagonal entries
$Q_{ii} = \|\mathbf{a}_i\|^2$. The corresponding terms in the quadratic form,

$$\sum_i Q_{ii}\,\bar{y}_i^2 = \sum_i \|\mathbf{a}_i\|^2\,\bar{y}_i^2$$

are per-label self-coupling terms that penalize individual label deviations
independently of other labels, duplicating the role of BCE. We subtract them:

$$E_{\text{quad}}(\mathbf{x}, \mathbf{y}) = -\frac{1}{2L}\left(\|A^\top \bar{\mathbf{y}}\|^2 - \sum_{i=1}^{L}\|\mathbf{a}_i\|^2\,\bar{y}_i^2\right)$$

Equivalently:

$$E_{\text{quad}}(\mathbf{x}, \mathbf{y}) = -\frac{1}{2L}\,\bar{\mathbf{y}}^\top\!\left(AA^\top - \operatorname{diag}(AA^\top)\right)\bar{\mathbf{y}} = -\frac{1}{2L}\sum_{i \neq j}\mathbf{a}_i^\top \mathbf{a}_j\,\bar{y}_i\,\bar{y}_j$$

This models exclusively the off-diagonal pairwise coupling derived in
Section 2.1. The $1/L$ normalization is chosen so that the energy gradient has
$O(1)$ magnitude, comparable to BCE. The negative sign means configurations
where $\bar{\mathbf{y}}$ aligns with the principal subspace of the off-diagonal
coupling have lower (more favorable) energy.

**Non-redundancy guarantee.** The gradient with respect to $\bar{y}_i$:

$$\frac{\partial E_{\text{quad}}}{\partial \bar{y}_i} = -\frac{1}{L}\sum_{j \neq i}\mathbf{a}_i^\top \mathbf{a}_j\,\bar{y}_j$$

The sum contains $L-1$ terms, each $O(1)$ under spectral normalization, giving
$\partial E / \partial \bar{y}_i = O(L/L) = O(1)$. This depends only on labels
$j \neq i$ — there is zero self-contribution. The energy gradient for label $i$
is determined entirely by the values of other labels, weighted by their learned
coupling to label $i$. This is structurally orthogonal to the BCE gradient,
which depends only on label $i$ itself.

### 2.6 Higher-Order Correction

The quadratic form captures pairwise interactions. To model higher-order
dependencies (e.g., three-way label co-occurrences), we add a nonlinear
correction:

$$E_{\text{higher}}(\mathbf{x}, \mathbf{y}) = \mathbf{w}^\top \operatorname{softplus}\!\left(W_h\,(W_y\,\bar{\mathbf{y}} + W_f\,g(\mathbf{x}))\right)$$

This term operates on the centered residual $\bar{\mathbf{y}}$, maintaining
non-redundancy with BCE. It is input-conditioned and projected through a
low-dimensional bottleneck to prevent overfitting.

### 2.7 Complete Energy Function

$$E_{\text{CREL}}(\mathbf{x}, \mathbf{y}) = E_{\text{quad}}(\mathbf{x}, \mathbf{y}) + E_{\text{higher}}(\mathbf{x}, \mathbf{y})$$

All linear layers (except label embeddings) are spectrally normalized to bound
the Lipschitz constant:

$$W_{\text{norm}} = \frac{W}{\sigma_{\max}(W)}$$

where $\sigma_{\max}$ is the largest singular value. This prevents energy scale
degeneracy during contrastive training, where unbounded energy magnitudes
preserve the ranking but destabilize gradients.

---

## 3. Training

### 3.1 Energy Network: InfoNCE

Given input $\mathbf{x}$, ground-truth labels $\mathbf{y}^0$, and $K$ negative
samples $\mathbf{y}^1, \ldots, \mathbf{y}^K$ drawn via independent Bernoulli
sampling from the task network's output:

$$y_i^k \sim \operatorname{Bernoulli}(F_\phi(\mathbf{x})_i) \qquad \text{for each } i, k$$

the energy network is trained with InfoNCE:

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(E(\mathbf{x}, \mathbf{y}^0) / \tau)}{\sum_{k=0}^{K}\exp(E(\mathbf{x}, \mathbf{y}^k) / \tau)}$$

where $\tau$ is the temperature. This is a $(K+1)$-way classification problem:
identify the ground-truth configuration among $K$ negatives. The energy network
learns to assign higher scores to correct label configurations.

An L2 regularization term prevents energy scale collapse:

$$\mathcal{L}_{\text{reg}} = \lambda_{\text{reg}} \cdot \frac{1}{K+1}\sum_{k=0}^{K} E(\mathbf{x}, \mathbf{y}^k)^2$$

### 3.2 Task Network: Cooperative Refinement

Unlike adversarial training (e.g., GANs), CREL uses cooperative training: both
networks optimize in the same direction. The task network is updated via a
gradient-based cooperative target.

**Why not direct energy maximization.** A natural formulation would be
$\mathcal{L}_{\text{coop}} = -E(\mathbf{x}, F_\phi(\mathbf{x}))$, directly
pushing predictions toward high-energy configurations. However, InfoNCE
constrains only the energy's **ranking** (relative order among configurations),
not its **scale** (absolute magnitude). The scale is an artifact of the NCE
training dynamics. Using $-E$ as a task loss couples the task network to this
artifact, creating a feedback loop: the task network chases high-energy regions,
the negatives (sampled from the task network) shift there, the energy network
raises E(ground truth) to compensate, and the task network chases the new scale.
This is the same instability as using raw discriminator logits in GANs.

**Normalized gradient direction loss.** The energy's gradient
$\nabla_{\mathbf{y}} E$ encodes structural information — which direction to
adjust labels to look more ground-truth-like — without scale dependence. We
normalize it to a unit direction vector per sample:

$$\hat{\mathbf{d}} = \frac{\nabla_{\mathbf{y}} E(\mathbf{x}, F_\phi(\mathbf{x});\,\theta_{\text{frozen}})}{\|\nabla_{\mathbf{y}} E(\mathbf{x}, F_\phi(\mathbf{x});\,\theta_{\text{frozen}})\|}$$

The cooperative loss is a dot product between the task network's predictions and
this direction, averaged over labels:

$$\mathcal{L}_{\text{coop}} = -\frac{1}{L}\,F_\phi(\mathbf{x})^\top \hat{\mathbf{d}}$$

The total task loss is simply their sum, with no weighting hyperparameter:

$$\mathcal{L}_{\text{task}} = \mathcal{L}_{\text{BCE}}(F_\phi(\mathbf{x}), \mathbf{y}) + \mathcal{L}_{\text{coop}}$$

**Natural gradient balance.** The $1/L$ averaging over labels in the
cooperative loss matches the reduction used by BCE ($\text{mean}$ over batch and
labels). Since $\hat{\mathbf{d}}$ has unit norm, each component
$\hat{d}_i \sim O(1/\sqrt{L})$. The per-element gradient contributions are
therefore:

$$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial \hat{y}_i} \sim O\!\left(\frac{1}{L}\right) \qquad \frac{\partial \mathcal{L}_{\text{coop}}}{\partial \hat{y}_i} \sim O\!\left(\frac{1}{L\sqrt{L}}\right)$$

The ratio is $1/\sqrt{L}$: the structural correction is naturally
$1/\sqrt{L}$ of the marginal correction. This scaling is principled — for large
label spaces, marginals dominate and correlations contribute a smaller relative
correction; for small label spaces, correlations matter more per label. No
weighting hyperparameter is needed because the balance emerges from the
mathematical form of the losses.

### 3.3 The Necessity of BCE

The centering design creates a clean separation of concerns that makes BCE
indispensable rather than redundant.

The EMA center $\boldsymbol{\mu}(\mathbf{x})$ is a tracking mechanism, not a
training signal — it records the task network's predictions but provides no
gradient. If BCE is removed:

1. No loss term pushes $F_\phi(\mathbf{x})$ toward correct marginal
   probabilities.
2. $\boldsymbol{\mu}(\mathbf{x})$ tracks whatever $F_\phi$ produces.
3. $\bar{\mathbf{y}} = F_\phi(\mathbf{x}) - \boldsymbol{\mu}(\mathbf{x})
   \to \mathbf{0}$ as the EMA converges to predictions.
4. The energy receives near-zero residuals and collapses.

BCE anchors $F_\phi$ to the ground-truth labels, giving the centering
subtraction a meaningful baseline. The resulting division of labor is:

| Component | Signal | Gradient w.r.t. $\hat{y}_i$ |
|-----------|--------|----------------------------|
| BCE | Per-label marginal: $\hat{y}_i \to y_i$ | Depends on $\hat{y}_i$ only |
| Energy | Cross-label structure: $\bar{y}_i \bar{y}_j$ | Depends on $\hat{y}_{j \neq i}$ only |

The two gradients are orthogonal by construction, providing complementary
supervision.

### 3.4 Inference

At test time, the task network's predictions can be optionally refined via $T$
steps of gradient ascent on the energy:

$$\mathbf{y}^{(0)} = F_\phi(\mathbf{x})$$

$$\mathbf{y}^{(t+1)} = \operatorname{clamp}\!\left(\mathbf{y}^{(t)} + \alpha\,\nabla_{\mathbf{y}} E(\mathbf{x}, \mathbf{y}^{(t)}),\; 0,\; 1\right) \qquad t = 0, \ldots, T-1$$

Because the cooperative training already distills structural knowledge into
$F_\phi$, the refinement provides diminishing but nonzero improvements.

---

## 4. Computational Analysis

The central complexity result is the reduction from $O(L^2)$ to $O(Lr)$ for the
quadratic energy computation.

| Operation | CREL | Naive $L \times L$ |
|-----------|------|--------------------|
| Form coupling matrix | Never materialized | $O(L^2 r)$ |
| Quadratic energy (single sample) | $O(Lr)$ | $O(L^2)$ |
| $K$ negative samples | $O(KLr)$ | $O(KL^2)$ |
| Memory for $A$ | $O(BLr)$ | $O(L^2)$ |

An implementation detail further reduces cost: the matrix $A(\mathbf{x})$ and
the per-label norms $\|\mathbf{a}_i\|^2$ are precomputed once per input and
shared across the ground-truth and all $K$ negative evaluations, avoiding
redundant forward passes through the projection network.

---

## 5. Diagnostics

The method admits three quantitative diagnostics that are not available in
prior work.

**Gradient orthogonality.** The cosine similarity $\rho$ between the energy and
BCE gradients (Section 1.3) should be near zero. Values significantly above zero
indicate the energy is wasting capacity on marginal information.

**Precision alignment.** The relative Frobenius error between the learned
off-diagonal coupling and the empirical precision matrix:

$$\delta = \frac{\|\operatorname{offdiag}(AA^\top) - \operatorname{offdiag}(\boldsymbol{\Sigma}^{-1})\|_F}{\|\operatorname{offdiag}(\boldsymbol{\Sigma}^{-1})\|_F}$$

Lower values indicate the learned coupling is capturing the true conditional
dependency structure.

**Gradient signal-to-noise ratio.** The ratio of gradient mean to gradient
standard deviation across batches:

$$\text{SNR} = \frac{\|\mathbb{E}[\nabla_{\hat{\mathbf{y}}} E]\|}{\mathbb{E}[\|\nabla_{\hat{\mathbf{y}}} E - \mathbb{E}[\nabla_{\hat{\mathbf{y}}} E]\|]}$$

Higher values indicate a more consistent structural signal from the energy
network.

---

## 6. Comparison with SEAL

| Property | SEAL | CREL |
|----------|------|------|
| Gradient redundancy $\rho$ | $\gg 0$ | $\approx 0$ by construction |
| Label coupling | Implicit via softplus | Explicit off-diagonal quadratic |
| Theoretical basis | Heuristic architecture | GMRF precision decomposition |
| Input conditioning | Local term only | Full coupling matrix $A(\mathbf{x})$ |
| Computational complexity | $O(Lh)$ | $O(Lr)$ with principled rank |
| Verifiability | None | Gradient orthogonality, precision alignment, SNR |
| Marginal–structure separation | Not enforced | Enforced by centering + diagonal correction |

The central contribution is methodological: rather than designing an energy
architecture heuristically and hoping it learns complementary structure, CREL
derives the correct functional form from the GMRF precision decomposition,
implements it efficiently via low-rank factorization, and guarantees
non-redundancy with per-label supervision through centering and diagonal
correction.
