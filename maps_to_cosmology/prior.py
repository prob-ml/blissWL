"""Prior distributions over cosmological parameters."""

from abc import ABC, abstractmethod

import jax.numpy as jnp
import numpyro.distributions as dist
from hydra.utils import instantiate
from jax import random
from omegaconf import DictConfig


class Prior(ABC):
    """Base class for prior distributions."""

    @abstractmethod
    def sample(self, key):
        """Sample a single value from the prior."""
        pass

    @abstractmethod
    def sample_batch(self, key, batch_size: int):
        """Sample a batch of values from the prior."""
        pass


class Normal(Prior):
    """Normal (Gaussian) prior distribution."""

    def __init__(self, loc: float, scale: float):
        self.loc = loc
        self.scale = scale
        self._dist = dist.Normal(loc, scale)

    def sample(self, key):
        return self._dist.sample(key)

    def sample_batch(self, key, batch_size: int):
        return self._dist.sample(key, sample_shape=(batch_size,))


class TruncatedNormal(Prior):
    """Truncated normal prior distribution."""

    def __init__(self, loc: float, scale: float, low: float = None, high: float = None):
        self.loc = loc
        self.scale = scale
        self.low = low if low is not None else -jnp.inf
        self.high = high if high is not None else jnp.inf
        base = dist.Normal(loc, scale)
        self._dist = dist.TruncatedDistribution(base, low=self.low, high=self.high)

    def sample(self, key):
        return self._dist.sample(key)

    def sample_batch(self, key, batch_size: int):
        return self._dist.sample(key, sample_shape=(batch_size,))


class LambdaCDMPrior:
    """Prior over LambdaCDM cosmological parameters."""

    def __init__(self, prior_config: DictConfig):
        self.param_names = list(prior_config.keys())
        self.priors = {
            name: instantiate(prior_config[name]) for name in self.param_names
        }

    def sample(self, key):
        """Sample a single set of cosmological parameters."""
        params = {}
        for name in self.param_names:
            key, subkey = random.split(key)
            params[name] = self.priors[name].sample(subkey)
        return params

    def sample_batch(self, key, batch_size: int):
        """Sample a batch of cosmological parameters."""
        params = {}
        for name in self.param_names:
            key, subkey = random.split(key)
            params[name] = self.priors[name].sample_batch(subkey, batch_size)
        return params
