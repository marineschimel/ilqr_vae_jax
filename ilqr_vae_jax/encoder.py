##this file contains different possible encoder networks
##the encoder should take in a time series of states
# def birnn_encoder(states, data):
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, Array
from jax.random import PRNGKey
import jax.random as jr
import sys
import ilqr_vae_jax
from typing import Any, Tuple
import chex
import pdb

# sys.path.append("..")
from diffilqrax.diff_ilqr import dilqr
from diffilqrax.typs import System, ModelDims, iLQRParams, ParallelSystem
from diffilqrax.parallel_dilqr import parallel_dilqr

from ilqr_vae_jax.likelihood import Likelihood
from ilqr_vae_jax.prior import Prior
from ilqr_vae_jax.dynamics import Dynamics, GRU, make_dyn_for_scan

# from lib import dynamics
from ilqr_vae_jax.utils import keygen, linear_readout
from ilqr_vae_jax.typs import (
    Dims,
    BiRNNParams,
    ReadoutParams,
    VAEParams,
    CovParams,
    iLQRHParams,
)
from ilqr_vae_jax import defaults

class Encoder(object):
    '''The encoder class is the parent class for all encoder networks.
    The main functions that will be exposed from the outside are:
        - initialize_params : takes in a random key and initializes the encoder-specific parameters
        - get_posterior_mean_and_cov : takes in the encoder parameters, a random key, the time series data, and whether the model is input-driven (inputs_allowed = 1) or whether we onlu infer the initial condition (inputs_allowed = 0). 
        The function then returns the posterior mean and covariance of the initial condition and the control inputs
    '''
    def __init__(self, dims: Dims):
        self.dims = dims

    def initialize_params(self, key: PRNGKey) -> CovParams:
        raise NotImplementedError(
            "direct evaluation of this function is not implemented"
        )

    def get_posterior_mean_and_cov(self, params, x_t):
        raise NotImplementedError(
            "direct evaluation of this function is not implemented"
        )
    
    def ic_posterior_sample(self, keys, m, logstd, dim, noisy = 1):
        def single_time_sample(key, m, log_std):
            s = m.squeeze() + noisy*(jnp.exp(log_std.squeeze()) * jax.random.normal(key, (dim,)))
            return s.squeeze()
        ss = jax.vmap(single_time_sample, in_axes = (0, None, None))(keys, m, logstd)
        return ss.reshape(-1, dim)


class BiRNN(Encoder):
    '''This implements a bidirectional RNN as the encoder. 
    It consists of two RNNs (one taking in the data forward and the other one backward) whose outputs are then concatenated.
    The encoder then outputs the mean and covariance of the initial condition and a time series, that can either be taken as such and given as inputs driving the dynamics, or given to a coupler that then combines it with the actual state (in which case one can interpret the encoder as attempting to perform state estimation).'''
    def __init__(self, dims: Dims, dynamics_module: Dynamics = GRU):
        n_encoder = defaults.n_encoder if dims.n_encoder is None else dims.n_encoder
        m_encoder = dims.m if dims.m_encoder is None else dims.m_encoder
        n_out = dims.n_out
        self.dims = dims
        self.horizon = dims.horizon
        self.birnn_dims = dims._replace(n=n_encoder, m=n_out, m_encoder=m_encoder)
        self.gru_dyn = ilqr_vae_jax.dynamics.GRU(self.birnn_dims)

    def initialize_params(self, key:PRNGKey) -> BiRNNParams:
        _, skeys = keygen(key, 7)
        fwd_rnn = self.gru_dyn.initialize_params(next(skeys))
        bwd_rnn = self.gru_dyn.initialize_params(next(skeys))
        readout_ic = ReadoutParams(
            c=0.01 * jr.normal(next(skeys), (2 * self.dims.n, 2 * self.dims.n_encoder)),
            b=jnp.zeros((2 * self.dims.n)),
        )
        readout = ReadoutParams(
            c=0.0
            * jr.normal(
                next(skeys), (2 * self.dims.m_encoder, 2 * self.dims.n_encoder)
            ),
            b=jnp.zeros((2 * self.dims.m_encoder,)),
        )
        x0_fwd = jr.normal(next(skeys), (self.birnn_dims.n_encoder,))
        x0_bwd = jr.normal(next(skeys), (self.birnn_dims.n_encoder,))
        return BiRNNParams(
            fwd_rnn=fwd_rnn,
            bwd_rnn=bwd_rnn,
            readout=readout,
            x0_bwd=x0_bwd,
            x0_fwd=x0_fwd,
            readout_ic=readout_ic,
        )

    def run_rnn(self, params, x0, us):
        _, xs = lax.scan(
            make_dyn_for_scan(self.gru_dyn.dynamics_t, params),
            x0,
            (us, None, None),
        )
        return xs

    def run_bidirectional_rnn(self, params: BiRNNParams, o_t: Array):
        """Run an RNN encoder backwards and forwards over some time series data.

        Arguments:
          params: a dictionary of bidrectional RNN encoder parameters
          fwd_rnn: function for running forward rnn encoding
          bwd_rnn: function for running backward rnn encoding
          x_t: np array data for RNN input with leading dim being time

        Returns:
          tuple of np array concatenated forward, backward encoding, and
            np array of concatenation of [forward_enc(T), backward_enc(1)]
        """
        fwd_enc_t = self.run_rnn(
            params.fwd_rnn, params.x0_fwd, o_t
        )  ##define parameters
        bwd_enc_t = self.run_rnn(params.bwd_rnn, params.x0_bwd, jnp.flipud(o_t))
        full_enc = jnp.concatenate([fwd_enc_t, bwd_enc_t], axis=1)
        enc_ends = jnp.r_[bwd_enc_t[0], fwd_enc_t[-1]]
        return full_enc, enc_ends

    def get_posterior_mean_and_cov(
        self,
        params: VAEParams,
        key: PRNGKey,
        data: Any,
        inputs_allowed: int = 1,
    ):
        key, skeys = keygen(key, 3)
        ts, ext_us, os = data #might want to incroporate external inputs if any here
        xenc_t, gen_pre_ics = self.run_bidirectional_rnn(params.encoder_params, os)
        ic_gauss_params = linear_readout(params.encoder_params.readout_ic, gen_pre_ics)
        ic_mean, ic_logvar = np.split(ic_gauss_params, 2, axis=0)
        us_mean, us_logvar = np.split(
            jax.vmap(linear_readout, in_axes=(None, 0))(
                params.encoder_params.readout, inputs_allowed * xenc_t
            ),
            2,
            axis=1,
        )
        ic_mean, ic_logvar = jnp.reshape(ic_mean, (self.dims.n,)), jnp.reshape(
            ic_logvar, (self.dims.n,)
        )
        us_mean, us_logvar = jnp.reshape(
            us_mean, (-1, self.birnn_dims.m_encoder)
        ), jnp.reshape(us_logvar, (-1, self.birnn_dims.m_encoder))
        return ic_mean, ic_logvar, us_mean, us_logvar
    
    
class iLQR(Encoder):
    '''The iLQR encoder uses iLQR to infer the initial condition and the control inputs.
    The induced posterior latent trajectories are the same as would be output by an iterative extended kalman filter.'''
    def __init__(
        self,
        likelihood: Likelihood,
        prior: Prior,
        dynamics: Dynamics,
        dims: Dims,
        ilqr_hparams: iLQRHParams,
    ):
        def make_bbeg_mat():
            if self.n_beg == 0:
                return jnp.eye(dims.m)
            else : 
                def bbeg_matrix(k):
                    if self.n_beg == 1:
                        return np.eye(dims.m)
                    elif k >= self.n_beg:
                        return np.zeros((dims.n, dims.m))
                    else :
                        if k == 0:
                            return np.concatenate([jnp.eye(dims.m), np.zeros(((ilqr_hparams.n_beg - k - 1)*dims.m, dims.m))])
                        elif k == self.n_beg - 1:
                            return np.concatenate([jnp.zeros((k*dims.m, dims.m)), np.eye(dims.m)])
                        else :
                            return np.concatenate([jnp.zeros((k*dims.m, dims.m)), np.eye(dims.m), np.zeros(((ilqr_hparams.n_beg - k - 1)*dims.m, dims.m))])
                return jnp.asarray(np.concatenate([bbeg_matrix(k)[None] for k in np.arange(ilqr_hparams.n_beg)]))
        self.n_beg = ilqr_hparams.n_beg
        self.n_out = dims.n_out
        self.n = dims.n
        self.m = dims.m
        self.dims = dims
        self.lik_module = likelihood
        self.dyn_module = dynamics
        self.prior_module = prior
        self.horizon = dims.horizon
        self.dt = ilqr_hparams.dt
        self.sig_s = None
        self.ic_logvar = None
        self.use_linesearch = ilqr_hparams.use_linesearch
        self.bbeg_mat = make_bbeg_mat()

    def initialize_params(self, key) -> CovParams:
        sig_t = -2*jnp.ones((self.horizon, self.m))
        ic_logvar = -jnp.ones((self.m))
        return CovParams(sig_t=sig_t, sig_ic=ic_logvar)
        # return jax.scipy.linalg.block_diag(*[s for s in sig_t])#, sig_s)


    def run_ilqr(self, key, params, dat):
        ts, ext_us, os = dat
        os = (
            jnp.concatenate([jnp.zeros((self.n_beg, self.n_out, 1)), os.reshape(-1, self.n_out, 1)], axis=0)
            if self.n_beg > 0
            else os
        )
        os = jnp.reshape(os, (self.horizon + self.n_beg, self.n_out))
        ext_us =  jnp.zeros((self.horizon + self.n_beg, ext_us.shape[-1]))

        def cost(t, x, u, params):
            o = os[t] ##first n_beg are zeros
            prior_term = jnp.where(t < self.n_beg, self.prior_module.log_prior_ic(params.prior_params, u), self.prior_module.log_prior_t(params.prior_params, u))
            lik_term = jnp.where(t < self.n_beg, 0., self.lik_module.log_likelihood_t(
                params.likelihood_params, x, o
            ))
            return jnp.sum(-lik_term- prior_term)

        def costf(x, theta):
            return -self.lik_module.log_likelihood_t(
                params.likelihood_params, x, os[-1]
            )

        def dynamics(t, x, u, params):
            tt = t - self.n_beg
            dyn_term = jnp.where(t < self.n_beg, x + self.bbeg_mat[t]@u, self.dyn_module.dynamics_t(params.dyn_params, x, u, tt, ext_us[t]))
            return dyn_term 
        
        lin_dyn = None 
        
        lin_cost = None
        
        quad_cost = None
        
        model = System(
            cost,
            costf,
            dynamics,
            ModelDims(
                horizon=self.horizon + self.n_beg, n=self.n, m=self.m, dt=self.dt
            ),
            lin_dyn = lin_dyn, 
            lin_cost = lin_cost, 
            quad_cost = quad_cost
        )
        full_params = iLQRParams(x0=jnp.zeros((self.n,)), theta=params)

        @chex.assert_max_traces(n = 3)
        def checked_dilqr(model, p, us, max_iter, use_linesearch, lin_dyn, lin_cost, quad_cost):
            return dilqr(model, p, us, max_iter = max_iter, use_linesearch=use_linesearch) #, lin_dyn = lin_dyn) #, lin_cost = lin_cost, quad_cost = quad_cost)

        #try : 
        tau_star, costs = checked_dilqr(
            model,
            full_params,
            0.*jax.random.normal(key = key, shape = (self.horizon + self.n_beg, self.m)),
            max_iter = 8,
            use_linesearch=self.use_linesearch,
            lin_dyn = lin_dyn, lin_cost = None, quad_cost = None)
        tau_star = jax.lax.cond(costs[-1] < costs[0], lambda: tau_star, lambda: jnp.zeros_like(tau_star))
        return tau_star
       # except : 
         #   return jnp.zeros_like(jnp.zeros((self.horizon + self.n_beg + 1, self.n + self.m)))   

    def get_posterior_mean_and_cov(
        self, params: VAEParams, key: PRNGKey, data: Tuple[Any], inputs_allowed: int = 1
    ):  
        tau_star = self.run_ilqr(key, params, data)
        ic_mean = tau_star[ self.n_beg, : self.n ]
        ic_logvar = params.encoder_params.sig_ic
        us_mean = tau_star[self.n_beg : -1, self.n :]
        us_logvar = params.encoder_params.sig_t
        return ic_mean, ic_logvar, us_mean, us_logvar

    def ic_posterior_sample(self, keys, m, logstd, dim, noisy = 1):
        def single_time_sample(key, m, log_std):
            log_std = jnp.tile(log_std, (self.n_beg, 1))
            log_std = log_std.reshape((self.dims.n,))
            m = m.reshape((self.dims.n, ))
            s = m + noisy*(jax.nn.softplus(log_std.squeeze() + 1e-3) * jax.random.normal(key, (self.dims.n,)))
            return s.reshape((self.n,))
        ss = jax.vmap(single_time_sample, in_axes = (0, None, None))(keys, m, logstd)
        return ss.reshape(-1, dim)
    


class ParalleliLQR(Encoder):
    '''The iLQR encoder uses iLQR to infer the initial condition and the control inputs.
    The induced posterior latent trajectories are the same as would be output by an iterative extended kalman filter.
    In the parallel version of the iLQR algorithm, if we are running the model on GPU, the linearized dynamics and the feedback gains are computed in parallel instead of being
    computed sequentially. This can be faster than the sequential version if the sequence length is long enough. Note however that 
    the memory can become a bottleneck, such that the parallel version is not always faster than the sequential version, even for very long sequences.
    Case-by-case testing is required to assess which version is better.'''
    def __init__(
        self,
        likelihood: Likelihood,
        prior: Prior,
        dynamics: Dynamics,
        dims: Dims,
        ilqr_hparams: iLQRHParams,
    ):
        def make_bbeg_mat():
            if self.n_beg == 0:
                return jnp.eye(dims.m)
            else : 
                def bbeg_matrix(k):
                    if self.n_beg == 1:
                        return np.eye(dims.m)
                    elif k >= self.n_beg:
                        return np.zeros((dims.n, dims.m))
                    else :
                        if k == 0:
                            return np.concatenate([jnp.eye(dims.m), np.zeros(((ilqr_hparams.n_beg - k - 1)*dims.m, dims.m))])
                        elif k == self.n_beg - 1:
                            return np.concatenate([jnp.zeros((k*dims.m, dims.m)), np.eye(dims.m)])
                        else :
                            return np.concatenate([jnp.zeros((k*dims.m, dims.m)), np.eye(dims.m), np.zeros(((ilqr_hparams.n_beg - k - 1)*dims.m, dims.m))])
                return jnp.asarray(np.concatenate([bbeg_matrix(k)[None] for k in np.arange(ilqr_hparams.n_beg)]))
        self.n_beg = ilqr_hparams.n_beg
        self.n_out = dims.n_out
        self.n = dims.n
        self.m = dims.m
        self.dims = dims
        self.lik_module = likelihood
        self.dyn_module = dynamics
        self.prior_module = prior
        self.horizon = dims.horizon
        self.dt = ilqr_hparams.dt
        self.sig_s = None
        self.ic_logvar = None
        self.use_linesearch = ilqr_hparams.use_linesearch
        self.bbeg_mat = make_bbeg_mat()

    def initialize_params(self, key) -> CovParams:
        sig_t = -jnp.ones((self.horizon, self.m))
        ic_logvar = -jnp.ones((self.m))
        return CovParams(sig_t=sig_t, sig_ic=ic_logvar)

        
    def run_ilqr(self, key, params, dat):
        ts, ext_us, os = dat
        os = (
            jnp.concatenate([jnp.zeros((self.n_beg, self.n_out, 1)), os.reshape(-1, self.n_out, 1)], axis=0)
            if self.n_beg > 0
            else os
        )
        os = jnp.reshape(os, (self.horizon + self.n_beg, self.n_out))
        ext_us =  jnp.zeros((self.horizon + self.n_beg, self.n))
        
        def cost(t, x, u, params):
            o = os[t - self.n_beg]
            prior_term = jnp.where(t < self.n_beg, self.prior_module.log_prior_ic(params.prior_params, u), self.prior_module.log_prior_t(params.prior_params, u))
            lik_term = jnp.where(t < self.n_beg, 0., self.lik_module.log_likelihood_t(
                params.likelihood_params, x, o
            ))
            return jnp.sum(-lik_term- prior_term)

        def costf(x, theta):
            return -self.lik_module.log_likelihood_t(
                params.likelihood_params, x, os[-1]
            )

        def dynamics(t, x, u, params):
            dyn_term = jnp.where(t < self.n_beg, x + self.bbeg_mat[t]@u, self.dyn_module.dynamics_t(params.dyn_params, x, u, t, ext_us[t]))
            return dyn_term 
        
        model = System(
            cost,
            costf,
            dynamics,
            ModelDims(
                horizon=self.horizon + self.n_beg, n=self.n, m=self.m, dt=self.dt
            )
        )
        full_params = iLQRParams(x0=jnp.zeros((self.n,)), theta=params)
        def parallel_dynamics(model, params, us, a_term):
            x0 = jnp.zeros((self.n,))
            return self.dyn_module.run_dynamics(params.theta.dyn_params, x0, us, None, ext_us)
            
        def parallel_dynamics_feedback(model, params, us, a_term, Ks, prev_Xs):
            x0 = jnp.zeros((self.n,)) 
            return self.dyn_module.run_dynamics_feedback(params.theta.dyn_params, x0, us, a_term, Ks, prev_Xs)
        parallel_model = ParallelSystem(model, parallel_dynamics, parallel_dynamics_feedback)

        @chex.assert_max_traces(n = 2)
        def pdilqr(parallel_model, p, us, use_linesearch, max_iter):
            return parallel_dilqr(parallel_model, p, us, use_linesearch=use_linesearch, max_iter = max_iter)

        tau_star = pdilqr(
            parallel_model,
            full_params,
            0.*jax.random.normal(key = key, shape = (self.horizon + self.n_beg, self.m)), #us_init : could be an argument
            use_linesearch=self.use_linesearch,
            max_iter = 8
        )
        return tau_star

    def get_posterior_mean_and_cov(
        self, params: VAEParams, key: PRNGKey, data: Tuple[Any], inputs_allowed: int = 1
    ):
        tau_star = self.run_ilqr(key, params, data)
        ic_mean = tau_star[ self.n_beg, : self.n ]
        ic_logvar = jnp.zeros(self.n) #params.encoder_params.sig_ic
        us_mean = tau_star[self.n_beg : -1, self.n :] # (T, self.m)
        us_logvar = jnp.zeros_like(us_mean) #params.encoder_params.sig_t
        return ic_mean, ic_logvar, us_mean, us_logvar

    def ic_posterior_sample(self, keys, m, logstd, dim, noisy = 1):
        def single_time_sample(key, m, log_std):
            #log_std = jnp.tile(log_std, (self.n_beg, 1))
            log_std = log_std.reshape((self.dims.n,))
            m = m.reshape((self.dims.n, ))
            s = m + noisy*(jax.nn.softplus(log_std.squeeze() + 1e-3) * jax.random.normal(key, (self.dims.n,)))
            return s.reshape((self.n,))
        ss = jax.vmap(single_time_sample, in_axes = (0, None, None))(keys, m, logstd)
        return ss.reshape(-1, dim)

