import math
from typing import Callable

import numpy as np
import torch


def point_mass_augmented_sampler(p: float = 0.0) -> Callable:
    def decorator(fn: Callable) -> Callable:
        def wrapped(batch_size: int, *args, **kwargs) -> torch.Tensor:
            # Sample \alpha = 0 from a point-mass augmented distribution
            return torch.where(
                torch.rand([batch_size]) < p,
                torch.randint(0, 3, [batch_size]).float() / 2,
                fn(batch_size, *args, **kwargs),
            )

        return wrapped

    return decorator


def beta_sampler(
    batch_size: int, alpha: float = 5, beta: float = 2, *args, **kwargs
) -> torch.Tensor:
    # Sample \alpha in the range [0, 1] from a
    # beta distribution with parameters 0.5, 0.5
    return torch.distributions.beta.Beta(alpha, beta).sample([batch_size])


def low_discrepancy_sampler(batch_size: int, *args, **kwargs) -> torch.Tensor:
    # Sample \alpha in the range [0, 1] from a
    # low discrepancy sequence
    return torch.remainder(
        torch.arange(1, batch_size + 1) / batch_size + np.random.rand(),
        1.0,
    )


def uniform_sampler(batch_size: int, *args, **kwargs) -> torch.Tensor:
    # Sample \alpha in the range [0, 1] from a
    # uniform distribution
    return torch.rand([batch_size])


def discrete_uniform_sampler(
    batch_size: int, maximum: int, *args, **kwargs
) -> torch.Tensor:
    return torch.randint(0, maximum, [batch_size]).float() / (maximum - 1)


def zeroless_discrete_uniform_sampler(
    batch_size: int, maximum: int, *args, **kwargs
) -> torch.Tensor:
    probs = torch.ones(maximum) / (maximum - 1)
    probs[0] = 0

    idx = torch.multinomial(probs, batch_size, replacement=True).float()
    return idx / (maximum - 1)


def constant_sampler(batch_size: int, val: float, *args, **kwargs) -> torch.Tensor:
    return val * torch.ones([batch_size])


def beta_binomial_sampler(
    batch_size: int, maximum: int, alpha: float = 5, beta: float = 2, *args, **kwargs
) -> torch.Tensor:
    p = torch.distributions.beta.Beta(alpha, beta).sample([batch_size])
    return torch.distributions.binomial.Binomial(maximum - 1, p).sample() / (maximum - 1)

def range_sampler(
    batch_size: int, maximum: int, *args, **kwargs
) -> torch.Tensor:
    return torch.randint(0, maximum + 1, [batch_size]).float() / maximum


class AlphaScheduler:
    def __init__(
        self,
        max_steps: int,
        inital_partitions: int,
        final_partitions: int,
    ):
        self.max_steps = max_steps
        self.inital_partitions = inital_partitions // 2
        self.final_partitions = final_partitions // 2

    def num_partitions(self, step: int) -> int:
        # Linear schedule
        return max(
            min(
                int(
                    round(
                        self.inital_partitions
                        + (self.final_partitions - self.inital_partitions)
                        * (step / self.max_steps) ** 1.5
                    )
                ),
                self.final_partitions,
            )
            * 2
            + 1,
            2,
        )

    def __call__(self, batch_size: int, step: int, **kwargs) -> torch.Tensor:
        num_partitions = self.num_partitions(step)
        probs = torch.ones(num_partitions) / (num_partitions - 1)
        probs[0] = 0

        idx = torch.multinomial(probs, batch_size, replacement=True)
        alphas = idx.float() / (num_partitions - 1)

        return alphas


class AlphaSampler:
    def __init__(self, sampler: str, maximum: int = 1001, p: float = 0.5):
        self.sampler_kwargs = {}

        if "beta" in sampler:
            alpha, beta = map(float, sampler.split("_")[-1].split(","))
            sampler = "_".join(sampler.split("_")[:-1])
            self.sampler_kwargs = {"alpha": alpha, "beta": beta}

        if "augmented" in sampler:
            base_sampler = sampler.split("augmented_")[-1]

            @point_mass_augmented_sampler(p=p)
            def _fn(batch_size: int, *args, **kwargs) -> torch.Tensor:
                return globals()[base_sampler](batch_size, *args, **kwargs)

            self.sampler_fn = _fn
        else:
            self.sampler_fn = globals()[sampler]

        self.maximum = maximum

    def __call__(self, batch_size: int, *args, **kwargs) -> torch.Tensor:
        if "step" in kwargs:
            # Remove the step argument from the keyword arguments
            _ = kwargs.pop("step")

        kwargs = kwargs | self.sampler_kwargs
        
        return self.sampler_fn(batch_size, maximum=self.maximum, *args, **kwargs)
