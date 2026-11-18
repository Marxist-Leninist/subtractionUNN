# Unary arithmetic and learning: research baseline 001

2026-11-18. Bounded first step, no paid compute and no production-model changes.
This is a research protocol and executable semantic baseline, not a learned-model result.

## What the repository currently implements

Audited `ArithmeticUNN.py` on `feature/four-unary-operations` at commit
5da3c3f0546322ea40c1e28f541f8a77d98c8f31. Original `SubUNN.py` remains unchanged.
`UnaryNN` starts with every weight equal to one. Its weighted count is
C(n) = sum(w[0:min(n,M)]) + max(0,n-M)*w[M]. Thus C(n)=n at initialization,
including M=0. The extension applies Python arithmetic to C(a) and C(b).
This is compressed-count arithmetic, not materialized unary-string processing.
Training is inherited subtraction-only. Correct subtraction examples already
have zero error at initialization: they do not demonstrate learned arithmetic.
Non-unit weights change the interpreted operands, so exact arithmetic on the
original input integers is no longer assured. The extension rejects invalid weights.

## Distinguish four representations

1. Natural unary: U(n)=1 repeated n times; U(0) is the empty sequence. Unary is
   a tally encoding, not an ordinary positional base with a meaningful base-1 logarithm.
2. Fixed-width thermometer: T_N(n)=n ones followed by N-n zeros, for 0<=n<=N.
   It has a range bound absent from variable-length unary.
3. Compressed count: a conventional integer storing n, as used by this repository.
   It represents a unary count but does not pay for n physical symbols.
4. Probability/density stream: the fraction of ones represents a value in [0,1].
   This is not the same integer semantics as U(n). Correlation matters to arithmetic.

All encodings require physical storage and operations. Neither unary notation nor
compression supplies unbounded physical precision or cost-free computation.

## Definitions and correctness arguments

For natural a,b:
- Addition concatenates U(a) and U(b); sequence lengths add.
- Saturating subtraction removes min(a,b) symbols from U(a), leaving max(a-b,0).
  It is not signed subtraction and does not form a group: (2 minus 5)+5=5, not 2.
- Multiplication concatenates b copies of U(a), giving ab symbols by induction on b.
- Euclidean division repeatedly removes b symbols, counting removals q. The invariant
  a=bq+r is preserved; r decreases by positive b, so it terminates with 0<=r<b.
  Uniqueness follows because two valid remainders differ by a multiple of b of
  magnitude below b. b=0 is explicitly undefined/rejected.
- Exact rational division is q+r/b, not rounded integer division. A pair of unary
  numerator/positive-denominator strings can encode rationals but is not one unary
  natural. Signed integers similarly need a sign or a difference pair and canonicalization.

These are mathematical arguments, not machine-checked universal proofs.
`formal_checks.py` exhaustively checks all 4,225 operand pairs in [0,64]^2,
including zero division and the Euclidean invariant. All pass. Its materialized
sequences are intentionally bounded; it is a reference, not an optimized kernel.
The result also checks that identical density-1/2 streams give AND density 1/2,
whereas one specifically balanced pair gives 1/4. A general independence guarantee
is not established by that single example.

## Literature and what it actually supports

- [Trask et al., NALU (2018)](https://arxiv.org/abs/1808.00508): learned gating over
  arithmetic primitives is an architectural inductive bias for numerical extrapolation.
  This differs from hard-coding which operation a caller requests. It is not a unary
  representation paper, and reported task success is not a universal arithmetic proof.
- [Madsen and Johansen, NAU/NMU (2020)](https://arxiv.org/html/2001.05016): inspect
  Sections 2 and 4 for constrained/sparse addition and multiplication units, repeated-seed
  evaluation, extrapolation and optimization difficulties. NMU's direct product construction
  is a useful multiplication baseline; it is not a general division module. Their discussion
  of division and gates motivates reporting failures rather than selecting one lucky run.
- [Heim et al., NPU (2020)](https://arxiv.org/abs/2006.01681): candidate learned-power
  baseline for the division research track. Its real-valued domain and numerical
  computation differ from exact integer quotient/remainder; zero and negative-domain
  policies must be specified rather than assumed. Only the abstract was reviewed here.
- [Alaghi and Hayes, stochastic computing survey (2013)](https://homes.cs.washington.edu/~armin/ACM_TECS_2013.pdf):
  density encodings can simplify circuits, but stream length and correlation affect
  precision and latency. AND multiplication requires appropriate stream dependence;
  scaled addition is not unscaled integer addition. Hardware benefits require hardware
  measurements, not timings of Python tuples.

## Next comparisons: preregister before training

Separate two questions: (A) Does a representation/kernel compute specified arithmetic?
(B) Can a model learn the rule, operand selection, or operation from examples?
Use the deterministic implementation as an oracle/baseline, not as a trained competitor.

First learned experiment should be small CPU addition/subtraction with distractor inputs:
a shared-weight sum encoder versus a fixed-width untied unary linear model and a scalar
linear baseline; add an MLP only with a matched budget. Then add NAU/NMU baselines from
pinned official implementations, without calling an approximate reimplementation a replication.
Use the same semantic inputs/labels and expose when an encoder already performs counting.

Proposed protocol: train operands 0..20, validation held-out pairs within that range,
frozen test ranges 21..40 and 41..100; separate zero/equal/reversed-operand cases.
Fix ten seeds, update budget, precision, optimizer, initializations and stopping rule
before evaluating tests. Publish all runs and failures, not the best seed alone.
For fixed-width models set capacity to 100 in advance and disclose unseen coordinates;
for shared encoders test actual longer strings, not only larger compressed integers.
Do not tune against extrapolation test scores.

Report raw exact-match, absolute error, tolerance-based success with stated threshold,
maximum error, seed success rate, memory, training time, inference time, input/output
encoding costs and parameter counts. Rounded-to-integer accuracy is secondary and
must not replace raw errors. Test arithmetic identities separately from loss.
For division report quotient and remainder accuracy and invariant violations separately;
exclude/reject zero divisors explicitly. Rational approximation and exact quotient/remainder
are distinct tasks. Signed tests are separate from natural saturation.

Add noisy-bit and correlated-stream stress tests only as a separate representation study.
One flipped unary bit changes the decoded count by one, but multiplication may amplify
that error by the other operand; no blanket robustness or energy-efficiency claim follows.

## Resource accounting and admission gates

Materialized U(n) takes n symbols; materialized multiplication output alone takes ab
symbols. Compressed Python integers instead have bit-length-dependent arithmetic costs.
This implementation cannot establish a unary hardware speedup by avoiding materialization.
A fixed-width neuron bank cannot accept arbitrary larger inputs without changing its
interface; shared recurrent rules can extend length but still require computation.

No new neural network was trained in baseline 001. No claim of improved intelligence,
learned division, or production integration is made. Promote a learned candidate only
when independently held-out and multi-seed comparisons support its declared scope.

Run: `python research/unary_baseline_001/formal_checks.py` from repository root.
