from jax import random, vmap, nn
import jax.numpy as jnp
import sys

sys.path.append("..")

from ilqr_vae_jax.utils import *
from typing import NamedTuple
from jax.random import PRNGKey
from jax import scipy
from ilqr_vae_jax.typs import Dims
from typing import Union, Optional


class GaussianPriorP(NamedTuple):
    log_std_u: jnp.ndarray
    log_std_ic: jnp.ndarray


class StudentPriorP(NamedTuple):
    log_std_u: jnp.ndarray
    log_std_ic: jnp.ndarray
    nu_m_2: float


class Prior(object):
    """
    The prior model class, p(u).
    Each prior implements its methods.
    This shows the functions that will be exposed from the outside : 
        - initialize_params : takes in a random key and initializes the prior-specific parameters
        - log_prior_t : takes in the prior parameters and the current state, and returns the log prior of the state at time t
        - log_prior : takes in the prior parameters and the time series of states, and returns the log prior of the states (a scalar value, summed across all times)
        - entropy : takes in the prior parameters and returns the entropy of the prior
        - save_params : takes in the prior parameters and a saving directory, and saves the prior parameters in the saving directory
    """

    def __init__(self, dims: Dims):
        self.dims = dims

    def initialize_params(self, key: PRNGKey):
        raise NotImplementedError(
            "direct evaluation of this function is not implemented"
        )

    def log_prior_t(self, prior_params, u_t):
        raise NotImplementedError(
            "direct evaluation of this function is not implemented"
        )

    def log_prior(self, prior_params, us):
        raise NotImplementedError(
            "direct evaluation of this function is not implemented"
        )

    def entropy(self, prior_params):
        raise NotImplementedError(
            "direct evaluation of this function is not implemented"
        )

    def save_params(
        self, prior_params: Union[GaussianPriorP, StudentPriorP], saving_dir: str, flag: Optional[int]
    ):
        flag = "" if flag is None else flag
        np.savez(f"{saving_dir}/prior_prms{flag}.npz", **prior_params._asdict())


class Gaussian(Prior):
    def __init__(self, dims: Dims, full_ic:bool = True, prior_logstd : float = 1.0):
        """
        Note, if you aren't using ilqr, you should set full_ic to True
        """
        self.m = dims.m
        self.n = dims.n
        self.ic_size = dims.n if full_ic else dims.m 
        self.prior_logstd = prior_logstd

    def initialize_params(self, key) -> GaussianPriorP:
        log_std_u = jnp.ones(shape=(self.m, 1))
        log_std_ic = jnp.ones(shape=(self.ic_size, 1)) 
        return GaussianPriorP(log_std_u=log_std_u, log_std_ic=log_std_ic)

    def log_prior_t(self, prior_params: GaussianPriorP, u_t: jnp.ndarray):
        log_std = prior_params.log_std_u
        var = (jnp.exp(2 * log_std)).reshape(-1, 1) + 1e-3
        u_t = u_t.reshape(-1, 1)
        return -0.5 * jnp.sum(
            jnp.square(u_t) / var + jnp.log(var) + jnp.log(2 * np.pi)
        )  # scipy.stats.multivariate_normal.logpdf(u_t.squeeze(), mean = jnp.zeros_like(u_t.squeeze()), cov = jnp.diag(jnp.exp(log_std))) #

    def log_prior(self, prior_params: GaussianPriorP, us: jnp.ndarray):
        lps = jnp.sum(vmap(lambda u_t: self.log_prior_t(prior_params, u_t))(us))
        return lps

    def log_prior_ic(self, prior_params: GaussianPriorP, ic: jnp.ndarray):
        log_std = prior_params.log_std_ic  # .squeeze()
        var = jnp.exp(2 * log_std.reshape(-1, self.ic_size)) + 1e-5
        ic = ic.reshape(-1, self.ic_size)
        return -0.5 * jnp.sum(
            jnp.square(ic) / var + jnp.log(var) + jnp.log(2 * np.pi)
        )  # scipy.stats.multivariate_normal.logpdf(ic.squeeze(), mean = jnp.zeros_like(ic.squeeze()), cov = jnp.diag(jnp.exp(log_std))) #

    def entropy(self, prior_params: GaussianPriorP):
        return 0.5 * jnp.sum(2 * prior_params.log_std + jnp.log(2 * jnp.pi))


class Student(Prior):
    def __init__(self, dims: Dims, full_ic = True, nu_init = 8.0):
        self.m = dims.m
        self.n = dims.n
        self.ic_size = dims.n if full_ic else dims.m 
        self.nu_init = nu_init

    def initialize_params(self, key) -> StudentPriorP:
        log_std_u = 2*jnp.ones(shape=(self.m, 1))
        nu_m_2 = self.nu_init - 2
        log_std_ic = jnp.ones(shape=(self.ic_size, 1)) 
        return StudentPriorP(
            log_std_u=log_std_u, nu_m_2=nu_m_2, log_std_ic=log_std_ic
        )

    def log_prior_t(self, prior_params, u_t):
        nu = nn.softplus(prior_params.nu_m_2) + 2
        m = self.m
        log_std_u = prior_params.log_std_u
        var =  nn.softplus(log_std_u) + 1e-3
        sigma = jnp.sqrt(var*(nu - 2) / nu) 
        nu_half = 0.5 * nu
        nu_plus_m_half = 0.5 * (nu + m)
        cst1 = jnp.log(scipy.special.gamma(nu_half)) - jnp.log(scipy.special.gamma(nu_plus_m_half))# gammaln(nu_half) - scipy.special.gammaln(nu_plus_m_half)
        cst2 = jnp.log(jnp.pi * nu) * 0.5 * m
        cst3 = jnp.sum(jnp.log(sigma))
        u_tilde = u_t.reshape(-1, 1) / sigma.reshape(-1, 1)
        return -(
            cst1
            + cst2
            + cst3
            + nu_plus_m_half * jnp.log(1 + jnp.sum(jnp.square(u_tilde)) / nu))

    def log_prior(self, prior_params, us):
        lps = jnp.sum(vmap(lambda u_t: self.log_prior_t(prior_params, u_t))(us))
        return lps

    def log_prior_ic(self, prior_params: GaussianPriorP, ic: jnp.ndarray):
        log_std = prior_params.log_std_ic  # .squeeze()
        var =  nn.softplus(log_std.reshape(-1, self.ic_size)) + 1e-3
        ic = ic.reshape(-1, self.ic_size)
        return -0.5 * jnp.sum(
            jnp.square(ic) / var + jnp.log(var) + jnp.log(2 * np.pi)
        )
