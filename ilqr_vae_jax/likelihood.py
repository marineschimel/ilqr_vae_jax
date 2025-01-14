from jax import random, vmap, scipy
import sys
import jax.numpy as jnp
from jax import Array, nn
# sys.path.append('..')
from ilqr_vae_jax.utils import *
from jax.random import PRNGKey
from typing import NamedTuple, Union, Optional
from jax.lax import stop_gradient
##this file contains different possible choices of likelihood functions

class GaussianLikelihoodP(NamedTuple):
    c: Array
    bias: Array
    log_std: Array
    
    
class PoissonLikelihoodP(NamedTuple):
    c: Array
    bias: Array
    gain: Array
    
class Likelihood(object):
    """
    The likelihood model class, p(o|z). 
    Each likelihood implements its methods.
    
    The methods that are exposed from the outside are:
        - initialize_params : takes in a random key and initializes the likelihood-specific parameters
        - log_likelihood_t : takes in the likelihood parameters, the current state, and the current observation, and returns the log likelihood of the observation given the state at time t
        - log_likelihood : takes in the likelihood parameters, the time series of states, and the time series of observations, and returns the log likelihood of the observations given the states (a scalar value, summed across all times)
        - sample_pre_os : takes in the likelihood parameters, the current state, and a random key, and returns a time series of the predicted mean of the likelihood distribution
        - sample_os : takes in the likelihood parameters, the current state, and a random key, and returns a time sereis of samples from the likelihood distribution
        - get_metric : takes in the likelihood prediction, the time series of observation, and returns a score of models performance
        - save_params : takes in the likelihood parameters and a saving directory, and saves the likelihood parameters in the saving directory
    """
    def __init__(self, dims=None):
        self.dims = dims
        
    def initialize_params(self):
        raise NotImplementedError('direct evaluation of this function is not implemented')
    
    def log_likelihood_t(self, likelihood_params, x_t, data_t):
        raise NotImplementedError('direct evaluation of this function is not implemented')
    
    def log_likelihood(self, likelihood_params, xs, data):
        raise NotImplementedError('direct evaluation of this function is not implemented')
    
    def sample_pre_os(self, likelihood_params, xs, keys):
        raise NotImplementedError('direct evaluation of this function is not implemented')
    
    def sample_os(self, likelihood_params, xs, keys):
        raise NotImplementedError('direct evaluation of this function is not implemented')
    
    def get_metric(self, preds, data):
        raise NotImplementedError('direct evaluation of this function is not implemented')
    
    def save_params(self, lik_params:Union[GaussianLikelihoodP, PoissonLikelihoodP], saving_dir: str,flag: Optional[int]
    ):
        flag = "" if flag is None else flag
        np.savez(f"{saving_dir}/likelihood_prms{flag}.npz", **lik_params._asdict())
        
    

class Gaussian(Likelihood):
    def __init__(self, dims, scale = 1.0, bias_init = None, learnable = 1):
        self.n = dims.n
        self.n_out = dims.n_out
        self.scale = scale
        self.bias_init = bias_init
        self.learnable = learnable
        
    def initialize_params(self, key) -> GaussianLikelihoodP:
        log_std = jnp.log(self.scale)*jnp.ones(shape = (self.n_out,))
        c = random.normal(key, shape = (self.n_out, self.n))/jnp.sqrt(self.n) 
        bias = self.bias_init.squeeze() if self.bias_init is not None else jnp.zeros(shape = (self.n_out,))
        return GaussianLikelihoodP(log_std = log_std, c = c, bias = bias)
    
    def ll_function(self, mu_t, data_t, scale):
        return scipy.stats.multivariate_normal.logpdf(mu_t, data_t, cov = scale)
    
    def log_likelihood_t(self, likelihood_params: GaussianLikelihoodP, x_t: Array, data_t: Array):
        mu_t = likelihood_params.c@x_t.squeeze() + likelihood_params.bias.squeeze()
        log_std = likelihood_params.log_std 
        var = nn.softplus(log_std*self.learnable) + 1e-3 
        data_t, mu_t, var = data_t.reshape(-1,), mu_t.reshape(-1,), var.reshape(-1,)
        val = vmap(self.ll_function)(mu_t, data_t, var)
        return val.sum()
    
    def log_likelihood(self, likelihood_params: GaussianLikelihoodP, xs: Array, data: Array):
        ts, _exts, obs = data
        mask = ~jnp.isnan(obs)
        lls = vmap(self.log_likelihood_t, in_axes = (None, 0, 0))(likelihood_params, xs, jnp.nan_to_num(obs))
        return lls.sum()
    
    def sample_pre_o(self, likelihood_params: GaussianLikelihoodP, x_t: Array):
        mu_t = likelihood_params.c@x_t.squeeze() + likelihood_params.bias.squeeze()
        return mu_t
    
    def sample_o(self, likelihood_params: GaussianLikelihoodP, mu_t: Array, key: PRNGKey):
        log_std = likelihood_params.log_std.squeeze()
        std = jnp.sqrt(nn.softplus(log_std) + 1e-3)
        o_t = mu_t.squeeze() + random.normal(key, shape = (self.n_out,)) * std
        return o_t
    
    def sample_pre_os(self, likelihood_params: GaussianLikelihoodP, xs: Array):
        return vmap(self.sample_pre_o,in_axes = (None,0))(likelihood_params, xs)
    
    def sample_os(self, likelihood_params: GaussianLikelihoodP, sample_pre_os: Array, keys: Array):
        return vmap(self.sample_o,in_axes = (None,0,0))(likelihood_params, sample_pre_os, keys)

    def get_metric(self, preds, data):
        n_neurons = data.shape[-1]
        assert preds.shape[-1] == n_neurons
        data = data.reshape((-1, n_neurons))
        var_per_neuron = data.var(axis = 0)
        preds = preds.reshape((-1, n_neurons))
        r2_per_neuron = 1 - ((data - preds)**2).mean(axis = 0)/var_per_neuron
        assert r2_per_neuron.shape[0] == n_neurons
        return jnp.mean(r2_per_neuron)

    
class Poisson(Likelihood):
    def __init__(self, dims, dt, fixed_id = False):
        self.n = dims.n
        self.n_out = dims.n_out
        self.phi = jnp.exp #jax.nn.softplus
        self.fixed_id = fixed_id
        self.dt = dt
        
        
    def ll_function(self, mu_t, data_t):
        return scipy.stats.poisson.logpmf(data_t, mu_t)
        
    def initialize_params(self, key) -> PoissonLikelihoodP:
        c = random.normal(key, shape = (self.n_out, self.n))/jnp.sqrt(self.n + self.n_out)
        bias = jnp.zeros(shape = (self.n_out,))
        gain = jnp.zeros(shape = (self.n_out,))
        return PoissonLikelihoodP(c = c, bias = bias, gain = gain)
    
    def log_likelihood_t(self, likelihood_params: PoissonLikelihoodP, x_t: Array, data_t: Array):
        mu_t = self.dt*self.phi(likelihood_params.c@x_t + likelihood_params.bias)
        data_t, mu_t = data_t.reshape(-1,1), mu_t.reshape(-1,1)
        val = vmap(self.ll_function)(mu_t, data_t)
        return val.sum() #jnp.sum(val)
    
    def log_likelihood(self, likelihood_params: PoissonLikelihoodP, xs: Array, data: Array):
        ts, _exts, obs = data
        mask = ~jnp.isnan(obs)
        lls = vmap(self.log_likelihood_t, in_axes = (None, 0, 0))(likelihood_params, xs, obs) #jnp.nan_to_num(obs))
        return jnp.sum(lls) #jnp.where(mask, lls, 0.).sum() # jnp.nan_to_num(lls) #
    
    
    def sample_pre_o(self, likelihood_params: PoissonLikelihoodP, x_t: Array):
        mu_t = self.dt*self.phi(likelihood_params.c@x_t + likelihood_params.bias)
        return mu_t.reshape((self.n_out,))
    
    def sample_o(self, likelihood_params: PoissonLikelihoodP, pre_o_t: Array, key: PRNGKey):
        return random.poisson(key, pre_o_t.reshape((self.n_out,)))
    
    def sample_pre_os(self, likelihood_params: PoissonLikelihoodP, xs: Array):
        return vmap(self.sample_pre_o,in_axes = (None,0))(likelihood_params, xs)
    
    def sample_os(self, likelihood_params: PoissonLikelihoodP,  sample_pre_os: Array, keys: Array):
        return vmap(self.sample_o,in_axes = (None,0,0))(likelihood_params,  sample_pre_os, keys)
    
    def get_metric(self, preds, data):
        n_neurons = data.shape[-1]
        assert preds.shape[-1] == n_neurons
        data = data.reshape((-1, n_neurons))
        mean_rate = data.mean(axis = 0) ##average number of spikes per neuron / dt 
        preds = preds.reshape((-1, n_neurons))
        def vmapped_ll(x, y):
            return jax.vmap(jax.scipy.stats.poisson.logpmf)(x,y)
        ll_preds = jax.vmap(vmapped_ll)(data, preds)
        mean_preds = jax.vmap(vmapped_ll)(data, mean_rate[None]*jnp.ones_like(data))
        return ((ll_preds.sum() - mean_preds.sum())/(np.log(2)*data.sum()))
    
    
