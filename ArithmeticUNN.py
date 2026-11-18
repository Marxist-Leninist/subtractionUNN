"""Four operations on weighted unary counts; see ARITHMETIC.md.

Unit weights give exact natural-number arithmetic without training. This is
a deterministic arithmetic extension, not evidence of learned reasoning.
"""
import argparse
from dataclasses import dataclass
from fractions import Fraction
from SubUNN import UnaryNN


def natural(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer (not bool or float)")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True)
class DivisionResult:
    quotient: int
    remainder: int
    divisor: int

    @property
    def exact(self):
        return Fraction(self.quotient * self.divisor + self.remainder, self.divisor)


class UnaryArithmeticNN(UnaryNN):
    """Legacy weighted-count model plus deterministic arithmetic composition.

    Inputs are natural-number counts, not materialized strings of ones.
    Subtraction saturates at zero, matching UnaryNN.forward. signed_subtract
    is explicit and returns a signed integer, not unsigned unary notation.
    Division returns Euclidean quotient/remainder; zero divisors raise.
    Inherited train/train_model retain their original subtraction-only role.
    """
    aliases = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide",
               "addition": "add", "subtraction": "subtract",
               "multiplication": "multiply", "division": "divide"}

    def __init__(self, max_value=100):
        super().__init__(natural(max_value, "max_value"))

    def _count(self, value):
        natural(value, "operand")
        if len(self.weights) != self.max_value + 1:
            raise ValueError("weights must retain max_value + 1 entries")
        if any(isinstance(w, bool) or not isinstance(w, int) or w < 1
               for w in self.weights):
            raise ValueError("weights must be positive integers")
        prefix = min(value, self.max_value)
        return sum(self.weights[:prefix]) + (value - prefix) * self.weights[-1]

    def forward(self, input1, input2, operation="subtract"):
        if not isinstance(operation, str):
            raise TypeError("operation must be a string")
        operation = self.aliases.get(operation, operation)
        if operation not in {"add", "subtract", "signed_subtract", "multiply", "divide"}:
            raise ValueError(f"unknown operation: {operation!r}")
        a, b = self._count(input1), self._count(input2)
        if operation == "add":
            return a + b
        if operation == "subtract":
            return max(0, a - b)
        if operation == "signed_subtract":
            return a - b
        if operation == "multiply":
            return a * b
        if b == 0:
            raise ZeroDivisionError("unary division by zero")
        q, r = divmod(a, b)
        return DivisionResult(q, r, b)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["add", "subtract", "signed_subtract", "multiply", "divide"])
    parser.add_argument("a", type=int)
    parser.add_argument("b", type=int)
    parser.add_argument("--max-value", type=int, default=100)
    args = parser.parse_args(argv)
    try:
        result = UnaryArithmeticNN(args.max_value).forward(args.a, args.b, args.operation)
    except (ValueError, TypeError, ZeroDivisionError) as error:
        parser.error(str(error))
    if isinstance(result, DivisionResult):
        print(f"quotient={result.quotient} remainder={result.remainder} exact={result.exact}")
    else:
        print(result)


if __name__ == "__main__":
    main()
