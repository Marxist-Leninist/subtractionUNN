"""Bounded executable checks, not a proof assistant or trained neural model."""
import json
from fractions import Fraction

def encode(n):
    if type(n) is not int or n < 0 or n > 10000:
        raise ValueError('bounded natural required')
    return (1,) * n

def add(a, b):
    return a + b

def sub(a, b):
    return a[min(len(a), len(b)):]

def mul(a, b):
    return tuple(x for _ in b for x in a)

def divide(a, b):
    if not b:
        raise ZeroDivisionError
    q, r = (), a
    while len(r) >= len(b):
        q = q + (1,)
        r = r[len(b):]
    return q, r

def run():
    pairs = 0
    for i in range(65):
        for j in range(65):
            a, b = encode(i), encode(j)
            assert len(add(a, b)) == i+j
            assert len(sub(a, b)) == max(i-j, 0)
            assert len(mul(a, b)) == i*j
            if j:
                q, r = divide(a, b)
                assert i == j*len(q)+len(r) and 0 <= len(r) < j
                assert Fraction(i, j) == len(q)+Fraction(len(r), j)
            else:
                try:
                    divide(a, b)
                except ZeroDivisionError:
                    pass
                else:
                    raise AssertionError('zero division accepted')
            pairs += 1
    # Exact finite correlation example; no Monte Carlo or independence claim.
    p = (1,1,0,0)
    independent_empirical = (1,0,1,0)
    density = lambda x: Fraction(sum(x),len(x))
    overlap = lambda x,y: density(tuple(a & b for a,b in zip(x,y)))
    assert overlap(p,p) == Fraction(1,2)
    assert overlap(p,independent_empirical) == Fraction(1,4)
    # Saturated subtraction is not a group operation.
    assert len(add(sub(encode(2), encode(5)), encode(5))) == 5 != 2
    return {'pairs':pairs,'operations':'pass','zero_divisor':'rejected',
            'same_stream_AND':'1/2','decorrelated_example_AND':'1/4',
            'training_performed':False,'scope':'bounded constructed arithmetic only'}

if __name__ == '__main__':
    print(json.dumps(run(),indent=2))
