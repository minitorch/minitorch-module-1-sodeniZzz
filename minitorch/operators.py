"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """Returns the product x * y."""
    return x * y

def id(x: float) -> float:
    """Returns the input unchanged."""
    return x

def add(x: float, y: float) -> float:
    """Returns the sum x + y."""
    return x + y

def neg(x: float) -> float:
    """Returns the negation -x."""
    return -1.0 * x

def lt(x: float, y: float) -> float:
    """Returns 1.0 if x < y, else 0.0."""
    return 1.0 if x < y else 0.0

def eq(x: float, y: float) -> float:
    """Returns 1.0 if x == y, else 0.0."""
    return 1.0 if x == y else 0.0

def max(x: float, y: float) -> float:
    """Returns the maximum of x and y."""
    return x if x > y else y

def is_close(x: float, y: float) -> float:
    """Returns 1.0 if |x - y| < 1e-2, else 0.0."""
    return 1.0 if abs(x - y) < 1e-2 else 0.0

def sigmoid(x: float) -> float:
    """Sigmoid function.

    Note:
        $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    Args:
        x: Input value.

    Returns:
        The sigmoid of x.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    return math.exp(x) / (1.0 + math.exp(x))

def relu(x: float) -> float:
    """ReLU activation function."""
    return x if x > 0.0 else 0.0

def log(x: float) -> float:
    """Returns the natural logarithm of x."""
    return math.log(x)

def exp(x: float) -> float:
    """Returns the exponential function of x."""
    return math.exp(x)

def inv(x: float) -> float:
    """Returns the reciprocal."""
    return 1.0 / x

def log_back(x: float, d: float) -> float:
    """Returns the derivative of log times a second arg."""
    return d / x

def inv_back(x: float, d: float) -> float:
    """Returns the derivative of reciprocal times a second arg."""
    return -d / (x * x)

def relu_back(x: float, d: float) -> float:
    """Returns the derivative of ReLU times a second arg."""
    return d if x > 0.0 else 0.0

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Applies a given function to each element of an iterable.

    Args:
        fn: Function from one value to one value.

    Returns:
        A function that takes a list, applies `fn` to each element, and returns a
        new list
    """
    def f(ls: Iterable[float]) -> Iterable[float]:
            return [fn(el) for el in ls]
    return f

def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Combines elements from two iterables using a given function.

    Args:
        fn: combine two values

    Returns:
        Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
        applying fn(x, y) on each pair of elements.

    """
    def f(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        return [fn(el1, el2) for (el1, el2) in zip(ls1, ls2)]
    
    return f

def reduce(fn: Callable[[float, float], float], start: float) -> Callable[[Iterable[float]], float]:
    r"""Reduces an iterable to a single value using a given function.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
        Function that takes a list `ls` of elements
        $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`
    """
    def f(ls: Iterable[float]) -> float:
        result = start
        for x in ls:
            result = fn(result, x)
        return result

    return f

def negList(ls: Iterable[float]) -> Iterable[float]:
    "Use `map` and `neg` to negate each element in `ls`"
    return map(neg)(ls)

def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    "Add the elements of `ls1` and `ls2` using `zipWith` and `add`"
    return zipWith(add)(ls1, ls2)

def sum(ls: Iterable[float]) -> float:
    "Sum up a list using `reduce` and `add`."
    return reduce(add, 0.0)(ls)

def prod(ls: Iterable[float]) -> float:
    "Product of a list using `reduce` and `mul`."
    return reduce(mul, 1.0)(ls)
