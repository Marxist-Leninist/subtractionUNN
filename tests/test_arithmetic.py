import contextlib
import io
import random
import unittest
from fractions import Fraction
from ArithmeticUNN import UnaryArithmeticNN, DivisionResult, main
from SubUNN import UnaryNN


class ArithmeticTests(unittest.TestCase):
    def test_exhaustive_small_and_legacy(self):
        model = UnaryArithmeticNN(7)
        old = UnaryNN(7)
        for a in range(41):
            for b in range(41):
                self.assertEqual(model.forward(a, b), old.forward(a, b))
                self.assertEqual(model.forward(a, b, "add"), a + b)
                self.assertEqual(model.forward(a, b, "subtract"), max(0, a-b))
                self.assertEqual(model.forward(a, b, "signed_subtract"), a-b)
                self.assertEqual(model.forward(a, b, "multiply"), a*b)
                if b:
                    result = model.forward(a, b, "divide")
                    self.assertEqual((result.quotient, result.remainder), divmod(a, b))
                    self.assertEqual(result.exact, Fraction(a, b))
                    self.assertEqual(a, b*result.quotient + result.remainder)
                    self.assertTrue(0 <= result.remainder < b)

    def test_large_counts_and_zero_cap(self):
        rng = random.Random(8319)
        for cap in (0, 1, 100):
            model = UnaryArithmeticNN(cap)
            for _ in range(100):
                a, b = rng.randrange(10**50), rng.randrange(1, 10**50)
                self.assertEqual(model.forward(a, b, "+"), a+b)
                self.assertEqual(model.forward(a, b, "*"), a*b)
                result = model.forward(a, b, "/")
                self.assertEqual(result.exact, Fraction(a, b))

    def test_invalid_operands_and_cap(self):
        for bad in (-1, 1.2, "2", True, None):
            with self.assertRaises((TypeError, ValueError)):
                UnaryArithmeticNN(bad)
            for op in ("add", "subtract", "multiply", "divide"):
                with self.assertRaises((TypeError, ValueError)):
                    UnaryArithmeticNN().forward(bad, 2, op)
                with self.assertRaises((TypeError, ValueError)):
                    UnaryArithmeticNN().forward(2, bad, op)

    def test_zero_division(self):
        for a in (0, 1, 10**30):
            with self.assertRaises(ZeroDivisionError):
                UnaryArithmeticNN().forward(a, 0, "divide")

    def test_unknown_operation(self):
        with self.assertRaises(ValueError):
            UnaryArithmeticNN().forward(1, 2, "power")
        with self.assertRaises(TypeError):
            UnaryArithmeticNN().forward(1, 2, None)

    def test_weighted_semantics_explicit(self):
        model = UnaryArithmeticNN(2)
        model.weights[:] = [2, 3, 4]
        self.assertEqual(model.forward(3, 1, "add"), 11)  # counts9 and2
        self.assertEqual(model.forward(3, 1, "multiply"), 18)
        self.assertEqual(model.forward(3, 1, "divide"), DivisionResult(4, 1, 2))
        old = UnaryNN(2); old.weights[:] = model.weights
        self.assertEqual(model.forward(3, 1), old.forward(3, 1))
        for weights in ([1], [1, 0, 1], [1, 1.5, 1], [1, True, 1]):
            model.weights = weights
            with self.assertRaises(ValueError):
                model.forward(1, 1)

    def test_training_not_needed_and_noop_on_correct_labels(self):
        model = UnaryArithmeticNN(8)
        before = model.weights[:]
        for a in range(9):
            for b in range(9):
                model.train(a, b, max(0, a-b))
        self.assertEqual(model.weights, before)

    def test_cli(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(["divide", "17", "5"])
        self.assertEqual(output.getvalue().strip(), "quotient=3 remainder=2 exact=17/5")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["divide", "1", "0"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
