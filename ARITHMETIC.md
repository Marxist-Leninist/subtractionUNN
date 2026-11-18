# Four-operation unary-count extension

The original `SubUNN.py` is unchanged. Run the new interface with Python3.8+;
there are no extra dependencies.

```sh
python ArithmeticUNN.py add 7 5
# 12
python ArithmeticUNN.py subtract 3 5
# 0 (legacy saturating subtraction)
python ArithmeticUNN.py signed_subtract 3 5
# -2
python ArithmeticUNN.py multiply 7 5
# 35
python ArithmeticUNN.py divide 17 5
# quotient=3 remainder=2 exact=17/5
python -m unittest discover -s tests -v
```

## Exact semantics

Inputs are non-negative Python integers. Floats, booleans, strings and negative
inputs are rejected. Subtraction defaults to max(0,a-b); the separately named
signed operation represents its result as a signed integer. Division returns
Euclidean quotient/remainder, satisfying a=b*q+r and 0<=r<b. Its `exact`
property returns a standard-library Fraction, a rational extension rather
than pretending fractions are ordinary unary integers. Division by zero
raises ZeroDivisionError. Unknown operations fail explicitly.

```python
from ArithmeticUNN import UnaryArithmeticNN
model = UnaryArithmeticNN(max_value=100)
assert model.forward(12, 4, "add") == 16
assert model.forward(17, 5, "divide").remainder == 2
```

The inherited prefix weights and final tail weight encode a weighted count:
sum(weights[:min(n,max_value)]) + max(0,n-max_value)*weights[-1].
The four operations compose those counts. With the default unit weights,
the counts equal the input integers exactly, including beyond max_value.
Changing weights changes the represented quantities: ordinary arithmetic
accuracy on the original inputs is then no longer guaranteed. Malformed,
non-integer or non-positive weights are rejected.

## What this does and does not demonstrate

The original unit-weight model already computes saturating subtraction
correctly before training; its correct-label training updates are no-ops.
This extension retains that interface, adds deterministic arithmetic
composition and validates semantics. It does **not** show that addition,
multiplication or division were learned by a neural network. The inherited
training methods are still subtraction-only, not a four-task learning
algorithm. A future learned experiment needs nontrivial initialization,
separate heldout data, baselines, and reported failures.

Counts are compressed integers, not allocated strings containing n ones.
Consequently this is not a physical unary-gate simulation or a claim of
constant-time arithmetic: cost depends on prefix length and integer bit length.
Very large max_value allocates that many weights, just as in the original.

## Validation

Eight unittest cases include exhaustive pairs0..40, legacy parity,300 random
50-digit operand pairs, quotient/remainder identities, weighted behavior,
zero and out-of-range inputs, invalid weights, CLI failures and proof that
correct-label training leaves unit weights unchanged. Test success establishes
these implementation contracts, not neural generalization.
