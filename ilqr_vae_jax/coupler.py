import jax.numpy as jnp
from jax import lax
import jax, sys
sys.path.append('..')
from ilqr_vae_jax import dynamics
from ilqr_vae_jax.encoder import *
from ilqr_vae_jax.utils import *
from ilqr_vae_jax.typs import *
from jax.random import PRNGKey
import jax, time, chex, itertools
import pdb


class Coupler(object):
  '''In LFADS, there is coupling between the encoder and the decoder via a controller
  In the case of iLQR, the controller is implicitly defined by iLQR so there is no need for an extra controller
  this coupler module is used to define how to couple the output of the encoder with the dynamics'''
  def __init__(self, dims: Dims):
    self.dims = dims

  def initialize_params(self, key):
    raise NotImplementedError('direct evaluation of this function is not implemented')

  def run_coupled_dyn(self, params, x0, us):
    '''this is a function that will take in as inputs the current controller state c_{t-1}, the output of the encoder at time t u_t, and the state at time t-1 x, 
    and return the corresponding control c_t + next state x_{t+1}
    in the case of iLQR of a basic biRNN encoder, this will return (u_t, nx) where we obtain nx by running the dynamics forward
    in the case of the GRU controller, this will return (nc, nx) where we obtain nc by running the GRU forward given the hidden state c_{t-1} and the concatenateion of the encoder output and the state at time t-1
    and nx by running the dynamics forward one step given the control'''
    raise NotImplementedError('direct evaluation of this function is not implemented')


class GRUController(Coupler):
    def __init__(self, dims, causal = False):
        self.m = dims.m 
        self.n = dims.n
        self.n_controller = dims.n_controller
        self.dims = dims 
        self.m_controller = dims.m_encoder + dims.n
        self.gru_dyn = dynamics.GRU(dims._replace(n = dims.n_controller, m = dims.m_encoder + dims.n, n_out = dims.m_encoder, m_encoder = dims.m))
        self.noisy_coupling = 1
        self.causal = causal

    def initialize_params(self, key):
        m : int = 2*self.m #if not self.causal else self.m
        return GRUControllerParams(readout =  ReadoutParams(c = 0.0*jax.random.normal(key, shape = (m, self.n_controller)), b = jnp.zeros(m)), 
                                  gru_params=self.gru_dyn.initialize_params(key), c0 = jnp.zeros(shape = (self.dims.n_controller,)))

    def run_coupled_dyn(self, params: VAEParams,  x0, us, ext_inpts, dyn_module, keys):
      dyn = dyn_module.dynamics_t
      def coupled_dynamics(x,u):
        (u, _), key, (t, ext_u) = u #u is the encoder output (and a key)
        (x, x_c) = x #x is composed of the state at time t-1 and the controller state at time t-1
        conc_u = jnp.concatenate([x, u], axis = 0)
        nx_c = self.gru_dyn.dynamics_t(params.coupling_params.gru_params, x_c, conc_u, t, ext_u)    
        m_c, logstd_c = np.split(linear_readout(params.coupling_params.readout, nx_c), 2, axis=0)
        stoch_c = m_c + jax.random.normal(key, (self.m,))*(jax.nn.softplus(logstd_c))
        nx_x = dyn(params.dyn_params, x, stoch_c, t, ext_u)
        return (nx_x, nx_c), (nx_x, stoch_c, m_c, logstd_c)
      _, xcs = lax.scan(coupled_dynamics, (x0, params.coupling_params.c0), (us, keys, ext_inpts))
      return xcs #want to return xs, us, us_mean, us_std
    
    
class IDCoupling(Coupler):
    def __init__(self, dims=None):
        self.dims = dims
        self.noisy_coupling = 0 
        self.m = self.dims.m

    def initialize_params(self, key):
        return None

    def run_coupled_dyn(self, params: VAEParams, x0, us, ext_inpts, dyn_module, keys):
        dyn = dyn_module.dynamics_t
        def coupled_dynamics(x, u):
            (u_mean, u_logstd), key, (t, ext_u) = u
            u_mean, u_logstd = u_mean.reshape((self.m,)), u_logstd.reshape((self.m,))
            u = u_mean + (jax.nn.softplus(u_logstd) * random.normal(key, (self.m,)))
            nx = dyn(params.dyn_params, x, u, t, ext_u)   
            return nx, (x, u, u_mean, u_logstd)
        _, xs = lax.scan(coupled_dynamics, x0, (us, keys, ext_inpts))
        xs, us, us_mean, us_logstd = xs
        return (xs, us, us_mean, us_logstd)

class ParallelIDCoupling(Coupler):
    '''This is the same as the IDCoupling but for the parallel model setting. It uses vmap for the sampling instead of doing it inside the scan'''
    def __init__(self, dims):
        self.dims = dims
        self.m = self.dims.m
        self.horizon = self.dims.horizon
        
    def initialize_params(self, key):
        return None
      
    def run_coupled_dyn(self, params: VAEParams, x0, us, ext_inpts, dyn_module, keys):
        key = keys[0]
        us_mean, us_logstd = us
        (ts, exts) = ext_inpts
        us = us_mean + (jax.nn.softplus(us_logstd) * random.normal(key, (self.horizon, self.m)))
        xs = dyn_module.run_dynamics(params.dyn_params, x0, us, ts, exts.xs)
        return (xs[1:], us, us_mean, us_logstd)
