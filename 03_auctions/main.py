import numpy as np
import sys
from scipy.stats import rv_continuous
from dataclasses import dataclass


@dataclass
class _DistParam:
    # This is just to make scipy.fit happy, ignore it. ;)
    name: str = ""
    integrality: bool = False
    domain: object = (-np.inf, np.inf)


class OrderStatisticNormal(rv_continuous):
    """Order statistic of a normal distribution.

    Attributes
        n: sample size,
        k: the k'th-smallest value.
    """

    def __init__(self, n, k, *args, **kwargs):
        super(OrderStatisticNormal, self).__init__(*args, **kwargs)
        self.n = n
        self.k = k

    def _shape_info(self):
        return [_DistParam("mean"), _DistParam("sd", domain=(0, np.inf))]

    def _pdf(self, x, mean, sd):
        """Evaluates the PDF of the order statistic of a normal distribution"""

        # Implement this if you want to use 'scipy.fit()'.

        return float("nan")

    def _cdf(self, x, mean, sd):
        """Evaluates the CDF of the order statistic of a normal distribution"""

        # (Optional for testing) Helps generate samples with '.rvs()'.

        return float("nan")


def read_input(io):
    """Parses the private value, the number of participating bidders, and
    the list of past winning bids.
    """

    flt = np.vectorize(float)

    str_private, str_bidders = io.readline().split()
    private, bidders = flt(str_private), np.int32(str_bidders)
    history = np.array(flt(io.readlines()))

    return private, bidders, history


if __name__ == "__main__":
    private, bidders, history = read_input(sys.stdin)

    # 1) Fit the distribution.
    # 2) Estimate the optimal bid.

    print(float("nan"))
