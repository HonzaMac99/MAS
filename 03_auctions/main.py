import numpy as np
import sys
import scipy.stats as stats
from scipy.stats import rv_continuous
from dataclasses import dataclass
from scipy.special import comb

testing = True  # False

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

    def _pdf(self, x, mean, std_dev):
        """Evaluates the PDF of the order statistic of a normal distribution"""

        # Implement this if you want to use 'scipy.fit()'.

        # Calculate the PDF of the kth order statistic of a normal distribution
        # if x < mean:
        #     return 0.0
        # else:

        # n!/[(k-1)!(n-k)!] = n * (n-1)!/[(k-1)!(n-k)!] = n * comb(n-1, k-1)
        coeff = self.n * comb(self.n - 1, self.k - 1)   # comb(self.n - 1, self.k - 1) ??
        p1 = (stats.norm.cdf(x, mean, std_dev)) ** (self.k - 1)
        p2 = (1 - stats.norm.cdf(x, mean, std_dev)) ** (self.n - self.k)
        pdf_norm = stats.norm.pdf(x, mean, std_dev)     # 1 / std_dev ??
        pdf = coeff * p1 * p2 * pdf_norm
        return pdf

    def _cdf(self, x, mean, std_dev):
        """Evaluates the CDF of the order statistic of a normal distribution"""

        # (Optional for testing) Helps generate samples with '.rvs()'.

        # Calculate the CDF of the kth order statistic of a normal distribution

        # if x < mean:
        #     return 0.0
        # else:

        cdf = 0
        for j in range(self.k, self.n + 1):
            coeff = comb(self.n, j)
            p1 = stats.norm.cdf(x, mean, std_dev) ** j
            p2 = (1 - stats.norm.cdf(x, mean, std_dev)) ** (self.n - j)
            cdf += coeff * p1 * p2
        return cdf


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
    private_v, bidders, history = read_input(sys.stdin)

    if testing:
        # opt_bid = float("nan")
        opt_bid = np.mean(history)
        # print(opt_bid)

        distr = OrderStatisticNormal(n=len(history), k=2)
        # params = distr.fit(history)
        # print(params)
    else:

        ### 1) Fit the distribution. ###

        distr = OrderStatisticNormal(n=len(history), k=bidders)

        # fit on the second highest bids (assuming my bid is the highest)
        params = distr.fit(history)
        # TODO: make this faster


        ### 2) Estimate the optimal bid. ###

        # Sample from estimated distribution truncated by private value
        lower_bound = private_v
        upper_bound = np.inf  # No upper bound for sampling
        a, b, loc, scale = params
        sampled_values = stats.truncnorm.rvs((lower_bound - loc) / scale,
                                             (upper_bound - loc) / scale, loc=loc, scale=scale, size=1000)

        # compute the optimal bid
        sorted_values = np.sort(sampled_values)
        second_highest_bids = sorted_values[-2::-1][:len(sorted_values) - 1]
        opt_bid = np.mean(second_highest_bids)

    print(opt_bid)
