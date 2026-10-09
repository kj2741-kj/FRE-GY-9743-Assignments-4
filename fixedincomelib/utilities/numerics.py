import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def _bucket_index(self, x: float) -> int:
        # Leftmost knot index j with axis1[j] >= x, clamped to the last index.
        # For axis1=[1,3,5,7]: x in (x_i, x_{i+1}] -> j=i+1 (left-continuous
        # step); x <= axis1[0] -> j=0; x > axis1[-1] -> j clamped to N-1.
        # One rule covers the interior step and both flat wings.
        j = np.searchsorted(self.axis1_, x, side='left')
        return int(min(j, self.length_ - 1))

    def interpolate(self, x: float) -> float:
        return float(self.values_[self._bucket_index(x)])

    def integrate(self, start_x: float, end_x: float) -> float:
        if start_x > end_x:
            return -self.integrate(end_x, start_x)
        if start_x == end_x:
            return 0.0

        # Insert every knot strictly inside (start_x, end_x) as a breakpoint.
        # Between consecutive breakpoints a < b there is no knot in (a, b),
        # so f is constant on (a, b] and equal to f(b) there (left-continuity).
        interior_knots = self.axis1_[(self.axis1_ > start_x) & (self.axis1_ < end_x)]
        breakpoints = np.concatenate(([start_x], interior_knots, [end_x]))

        total = 0.0
        for a, b in zip(breakpoints[:-1], breakpoints[1:]):
            total += self.interpolate(b) * (b - a)
        return total

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        grad = np.zeros(self.length_)
        grad[self._bucket_index(x)] = 1.0
        return grad

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        if start_x > end_x:
            return -self.gradient_of_integrated_value_wrt_ordinate(end_x, start_x)

        grad = np.zeros(self.length_)
        if start_x == end_x:
            return grad

        # Same breakpoint decomposition as integrate(): the integral is
        # linear in values, so accumulate each segment's width into the
        # ordinate index that is active on that segment.
        interior_knots = self.axis1_[(self.axis1_ > start_x) & (self.axis1_ < end_x)]
        breakpoints = np.concatenate(([start_x], interior_knots, [end_x]))

        for a, b in zip(breakpoints[:-1], breakpoints[1:]):
            j = self._bucket_index(b)
            grad[j] += (b - a)
        return grad


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        else:
            raise Exception('Currently only support PCP interpolation')