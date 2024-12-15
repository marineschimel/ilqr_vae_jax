"""This file contains different functions/modules that can be called for the dynamics
"""

from typing import Callable, Any, Union, List, Tuple
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import jax.random as jr
import jax, sys
from functools import partial
from jax.lax import scan, associative_scan
from jax.scipy.linalg import block_diag
from ilqr_vae_jax.utils import bmm

import pdb

sys.path.append("..")
from ilqr_vae_jax.typs import *

@jax.vmap
def dynamic_operator(elem1, elem2):
    """Associative operator for forward linear dynamics

    Args:
        elem1 (Tuple[Array, Array]): Previous effective state dynamic and effective bias
        elem2 (Tuple[Array, Array]): Next effective state dynamic and effective bias

    Returns:
        Tuple[Array, Array]: Updated state and control
    """
    F1, c1 = elem1
    F2, c2 = elem2
    F = F2 @ F1
    c = F2 @ c1 + c2
    return F, c

def keygen(key, nkeys):
    """Generate randomness that JAX can use by splitting the JAX keys."""
    keys = jr.split(key, nkeys + 1)
    return keys[0], (k for k in keys[1:])


def make_dyn_for_scan(dyn, params):
    """Scan requires f(h, x) -> h, h, in this application.
    Args:
      rnn : f with sig (params, h, x) -> h
      params: params in f() sig.

    Returns:
      f adapted for scan
    """

    def rnn_for_scan(x, inpts):
        (u, t, ext) = inpts
        nxt_x = dyn(params, x, u, t, ext)
        return nxt_x, nxt_x

    return rnn_for_scan


class Dynamics(object):
    def __init__(self, dims=None):
        self.dims = dims

    def initialize_params(self, key):
        raise NotImplementedError(
            "direct evaluation of this function is not implemented"
        )

    def dynamics_t(self, params, x_t, u_t, t, ext_t):
        raise NotImplementedError(
            "direct evaluation of this function is not implemented"
        )
        
    def run_dynamics(self, params, x0 : jnp.ndarray, xs : jnp.ndarray, us : jnp.ndarray, ts : Optional[jnp.ndarray], exts) -> Union[jnp.ndarray, Tuple[jnp.ndarray, int]]:
        raise NotImplementedError(
            "direct evaluation of this function is not implemented"
        )

    def save_params(
        self,
        dyn_params: Union[VanillaParams, MGUParams, GRUParams],
        saving_dir: str,
        flag: Optional[int]
    ):
        flag = "" if flag is None else flag
        np.savez(f"{saving_dir}/dyn_prms{flag}.npz", **dyn_params._asdict())


# flax pytree
class VRNN(Dynamics):
    def __init__(self, phi: Callable[[jnp.ndarray], jnp.ndarray], dims: Dims, rad=0.7):
        self.dims = dims
        self.phi = phi
        self.rad = rad
        self.n = dims.n
        self.m = dims.m

    def initialize_params(self, key) -> VanillaParams:
        a = self.rad * jr.normal(key, (self.n, self.n)) / jnp.sqrt(self.n)
        b = jr.normal(key, (self.n, self.m))
        bias = jnp.zeros((self.n))
        return VanillaParams(a=a, b=b, bias=bias)

    def dynamics_t(
        self,
        params: VanillaParams,
        x_t: jnp.ndarray,
        u_t: jnp.ndarray,
        t: int,
        ext_t: Any,
    ) -> jnp.ndarray:
        b = params.b / jnp.linalg.norm(params.b, axis = 0)
        x_t = np.reshape(x_t, (-1,))
        u_t = np.reshape(u_t, (-1,))
        nx = (params.a @ self.phi(x_t)) + (b @ u_t) + params.bias
        return nx
    
    def run_dynamics(
        self,
        params: VanillaParams,
        x0: jnp.ndarray,
        us: jnp.ndarray,
        ts: int,
        ext_ts: Any,
    ):
        _, xs = lax.scan(
            make_dyn_for_scan(self.dynamics_t, params), x0, (us, ts, ext_ts)
        )
        return xs

class MGU(Dynamics):
    def __init__(
        self, dims=None, phi: Callable = jax.nn.sigmoid, rho: Callable = jnp.tanh, rad = 0.1,
    ):
        """Minimal Gated Unit

        Args:
            dims (Dims, optional): scale of network. Defaults to None.
            phi (Callable, optional): nonlinear fn 1. Defaults to jax.nn.sigmoid.
            rho (Callable, optional): nonlinear fn applied to forget gate. Defaults to jnp.tanh.
        """
        self.dims = dims
        self.phi = phi
        self.rho = rho
        self.m = self.dims.m
        self.n = self.dims.n
        self.rad = rad

    def initialize_params(self, key) -> MGUParams:
        key, subkey = jr.split(key)
        wrx = self.rad/jnp.sqrt(self.n)*jax.random.normal(key,(self.dims.n, self.dims.n))
        wru = self.rad/jnp.sqrt(self.n)*jnp.ones((self.dims.n, self.dims.m))
        wc = self.rad/jnp.sqrt(self.n)*jax.random.normal(subkey,(self.dims.n, self.dims.n))
        bc = jnp.zeros((self.dims.n,))
        brx = jnp.zeros((self.dims.n,))
        bru = jnp.zeros((self.dims.n,))
        return MGUParams(wru=wru, wrx=wrx, wc=wc, bc=bc, brx=brx, bru=bru)

    def dynamics_t(
        self, params: MGUParams, x_t: jnp.ndarray, u_t: jnp.ndarray, t: int, ext_t: Any
    ) -> jnp.ndarray:
        x_t, u_t = x_t.squeeze(), u_t.squeeze()
        bfg = 0.5
        hpred = x_t
        wru = params.wru / jnp.linalg.norm(params.wru, axis = 0)
        f = self.phi(params.brx + jnp.dot(params.wrx, hpred))
        h_hat = self.rho(params.bru + jnp.dot(params.wc, (hpred * f))).squeeze() + jnp.dot(
            wru, u_t
        ).squeeze()
        res = ((1.0 - f) * hpred) + (f * h_hat)
        return res.squeeze()

    
    def run_dynamics(self, params, x0, us, ts, exts):
        _, xs = lax.scan(make_dyn_for_scan(self.dynamics_t, params), x0, (us, ts, exts))
        return xs


class GRU(Dynamics):
    def __init__(
        self,
        dims: Dims = None,
        phi: Callable = jax.nn.sigmoid,
        rho: Callable = jnp.tanh,
        rad: float = 0.8,
    ):
        self.dims = dims
        self.phi = phi
        self.rho = rho
        self.rad = rad
        self.m_ext = self.dims.m

    #alpha R + (1 alpha)*R
    def initialize_params(self, key) -> GRUParams:
        key, skeys = keygen(key, 7)
        wzx = self.rad * jr.normal(next(skeys), (self.dims.n, self.dims.n))/jnp.sqrt(self.dims.n)
        wzu = self.rad * jr.normal(next(skeys), (self.dims.n, self.dims.m))/jnp.sqrt(self.dims.n)
        wrx = self.rad * jr.normal(next(skeys), (self.dims.n, self.dims.n))/jnp.sqrt(self.dims.n)
        wru = self.rad * jr.normal(next(skeys), (self.dims.n, self.dims.m))/jnp.sqrt(self.dims.n)
        whx = self.rad * jr.normal(next(skeys), (self.dims.n, self.dims.n))/jnp.sqrt(self.dims.n)
        whu = self.rad * jr.normal(next(skeys), (self.dims.n, self.dims.m))/jnp.sqrt(self.dims.n)
        bz = jnp.zeros((self.dims.n,1))
        br = jnp.zeros((self.dims.n,1))
        bh = jnp.zeros((self.dims.n,1))
        return GRUParams(
                wru=wru, wrx=wrx, wzx=wzx, wzu=wzu, whx=whx, whu=whu, br=br, bz=bz, bh=bh, 
                wzu_ext = jnp.zeros((self.dims.n, self.m_ext)), wru_ext = jnp.zeros((self.dims.n, self.m_ext)), whu_ext = 
                jnp.zeros((self.dims.n, self.m_ext))
            )

    def dynamics_t(
        self, params: GRUParams, x_t: jnp.ndarray, u_t: jnp.ndarray, t: int, ext_t: Any
    ) -> jnp.ndarray:
        x_t, u_t = x_t.reshape(-1,1), u_t.reshape(-1,1)
        wzu = params.wzu / jnp.linalg.norm(1e-5 + params.wzu, axis = 0)
        wru = params.wru / jnp.linalg.norm(1e-5 + params.wru, axis = 0)
        whu = params.whu / jnp.linalg.norm(1e-5 + params.whu, axis = 0)
        z_t = self.phi(params.bz + jnp.dot(params.wzx, x_t) + jnp.dot(wzu, u_t))
        r_t = self.phi(params.br + jnp.dot(params.wrx, x_t) + jnp.dot(wru, u_t))
        hh_t = self.rho(params.bh + jnp.dot(params.whx, r_t * x_t) + jnp.dot(whu, u_t))
        nx_t = (1.0 - z_t) * x_t + z_t * hh_t
        return nx_t.squeeze()

    def run_dynamics(self, params, x0, us, ts, exts):
        _, xs = lax.scan(make_dyn_for_scan(self.dynamics_t, params), x0, (us, ts, exts))
        return xs  # jnp.concatenate([x0[None,...], xs[:-1]])


class GRU_Ext(Dynamics):
    def __init__(
        self,
        dims: Dims = None,
        phi: Callable = jax.nn.sigmoid,
        rho: Callable = jnp.tanh,
        rad: float = 0.1,
        ext_inputs: bool = False, 
        m_ext: Optional[int] = None
    ):
        self.dims = dims
        self.phi = phi
        self.rho = rho
        self.rad = rad
        self.ext_inputs = ext_inputs
        self.m_ext = self.dims.n if m_ext is None else m_ext

    def initialize_params(self, key) -> GRUParams:
        key, skeys = keygen(key, 7)
        wzx = self.rad * jr.normal(next(skeys), (self.dims.n, self.dims.n))/jnp.sqrt(self.dims.n)
        wzu = self.rad * jr.normal(next(skeys), (self.dims.n, self.dims.m))/jnp.sqrt(self.dims.n)
        wrx = self.rad * jr.normal(next(skeys), (self.dims.n, self.dims.n))/jnp.sqrt(self.dims.n)
        wru = self.rad * jr.normal(next(skeys), (self.dims.n, self.dims.m))/jnp.sqrt(self.dims.n)
        whx = self.rad * jr.normal(next(skeys), (self.dims.n, self.dims.n))/jnp.sqrt(self.dims.n)
        whu = self.rad * jr.normal(next(skeys), (self.dims.n, self.dims.m))/jnp.sqrt(self.dims.n)
        bz = jnp.zeros((self.dims.n,1))
        br = jnp.zeros((self.dims.n,1))
        bh = jnp.zeros((self.dims.n,1))
        if self.ext_inputs : 
            return GRUParams(
                wru=wru, wrx=wrx, wzx=wzx, wzu=wzu, whx=whx, whu=whu, br=br, bz=bz, bh=bh, 
                wzu_ext = jnp.zeros((self.dims.n, self.m_ext)), wru_ext = jnp.zeros((self.dims.n, self.m_ext)), whu_ext = 
                jnp.zeros((self.dims.n, self.m_ext))
            )

        else : 
            return GRUParams(
                wru=wru, wrx=wrx, wzx=wzx, wzu=wzu, whx=whx, whu=whu, br=br, bz=bz, bh=bh
            )

    def dynamics_t(
        self, params: GRUParams, x_t: jnp.ndarray, u_t: jnp.ndarray, t: int, ext_t: Any
    ) -> jnp.ndarray:
        x_t, u_t, ext_t = x_t.reshape(-1,1), u_t.reshape(-1,1), ext_t.reshape(-1,1)
        wzu = params.wzu / jnp.linalg.norm(1e-5 + params.wzu, axis = 0)
        wru = params.wru / jnp.linalg.norm(1e-5 + params.wru, axis = 0)
        whu = params.whu / jnp.linalg.norm(1e-5 + params.whu, axis = 0)
        z_t = self.phi(params.bz + jnp.dot(params.wzx, x_t) + jnp.dot(wzu, u_t) + jnp.dot(params.wzu_ext, ext_t))
        r_t = self.phi(params.br + jnp.dot(params.wrx, x_t) + jnp.dot(wru, u_t) + jnp.dot(params.wru_ext, ext_t))
        hh_t = self.rho(params.bh + jnp.dot(params.whx, r_t * x_t) + jnp.dot(whu, u_t) + jnp.dot(params.whu_ext, ext_t))
        nx_t = (1.0 - z_t) * x_t + z_t * hh_t
        return nx_t.squeeze()

    def run_dynamics(self, params, x0, us, ts, exts):
        _, xs = lax.scan(make_dyn_for_scan(self.dynamics_t, params), x0, (us, ts, exts))
        return xs
    



class VRNN_Ext(Dynamics):  
    def __init__(self, phi: Callable[[jnp.ndarray], jnp.ndarray], dt: float, dims: Dims, rad=0.9, m_ext=None):
        self.dims = dims
        self.phi = phi
        self.rad = rad
        self.n = dims.n
        self.m = dims.m
        self.m_ext = self.m if m_ext is None else m_ext
        self.max_eig = 0.95
        self.dt = dt

    def initialize_params(self, key) -> VanillaExtParams:
        keys = jr.split(key, 3)
        a = self.rad * jr.normal(keys[0], (self.n, self.n)) / jnp.sqrt(self.n)
        #init_max_eig = jnp.max(jnp.abs(jnp.linalg.eigvals(a)))
        #a = a / init_max_eig * self.max_eig
        b = jr.normal(keys[1], (self.n, self.m))
        bias = jnp.zeros((self.n))
        b_ext = 0.1*jr.normal(keys[2], (self.n, self.m_ext))
        return VanillaExtParams(a=a, b=b, bias=bias, b_ext=b_ext)

    def dynamics_t(
        self,
        params: VanillaExtParams,
        x_t: jnp.ndarray,
        u_t: jnp.ndarray,
        t: int,
        ext_t: jnp.ndarray,
    ) -> jnp.ndarray:
        b = params.b / jnp.linalg.norm(params.b, axis = 0) ##renormalize outside dyn_t
        b_ext = params.b_ext 
        x_t = np.reshape(x_t, (-1,))
        u_t = np.reshape(u_t, (-1,))
        ext_t = np.reshape(ext_t, (-1,))
        nx = (params.a @ self.phi(x_t)) + (b @ u_t) + b_ext @ ext_t + params.bias
        return nx #_t + dx*self.dt
    
    
    def run_dynamics(
        self,
        params: VanillaExtParams,
        x0: jnp.ndarray,
        us: jnp.ndarray,
        ts: int,
        ext_ts: jnp.ndarray,
    ):
        _, xs = lax.scan(
            make_dyn_for_scan(self.dynamics_t, params), x0, (us, ts, ext_ts)
        )
        return xs
    
    def save_params(self, dyn_params: VanillaParams, saving_dir: str, flag: Optional[int]):
        w_eig, _ = np.linalg.eig(dyn_params.a)
        np.savetxt(f"{saving_dir}/eig{flag}", w_eig)
        np.savez(f"{saving_dir}/dyn_prms{flag}.npz", w_eig=w_eig, **dyn_params._asdict())
    


class ParallelLinear(Dynamics): ##no external input here 
    def __init__(
        self,
        dims=None,
        dt: float = 0.05,
    ):
        self.hparams = dims
        self.n = dims.n
        self.m = dims.m

    def initialize_params(self, key):
        a = 0.2*jr.normal(key, (self.n, self.n)) / jnp.sqrt(self.n)
        b = jr.normal(key, (self.n, self.m))
        bias = jnp.zeros((self.n,))
        return VanillaParams(a=a, b=b, bias=bias)

    def dynamics_t(
        self,
        params: VanillaParams,
        x_t: jnp.ndarray,
        u_t: jnp.ndarray,
        t: int,
        ext_t: jnp.ndarray,
    ) -> jnp.ndarray:
        b = params.b / jnp.linalg.norm(params.b, axis = 0)
        x_t = np.reshape(x_t, (-1,))
        u_t = np.reshape(u_t, (-1,))
        ext_t = np.reshape(ext_t, (-1,))
        a = params.a
        nx = (a @ x_t) + (b @ u_t) + params.bias 
        return nx
    
    
    def build_fwd_elements(self, params, x0, us):
        #x0 = jnp.zeros((self.n,))
        initial_element = (jnp.zeros_like(jnp.diag(x0)), x0)
        b = params.b/ jnp.linalg.norm(params.b, axis = 0)
        b = jnp.tile(b, (us.shape[0],1,1))
        bias = params.bias
        a = params.a 
        bias = jnp.tile(bias, (us.shape[0],1))
        @partial(jax.vmap, in_axes=(0, 0,0,0))
        def _generic(a,b,bias, u):
            return a, b @ u + bias

        generic_elements = _generic(a,b, bias,us)
        return tuple(
            jnp.concatenate([jnp.expand_dims(first_e, 0), gen_es])
            for first_e, gen_es in zip(initial_element, generic_elements)
        )

    def run_dynamics(self, params, x0, us, ts, exts):
        params = params._replace(a = jnp.tile(params.a, (us.shape[0],1,1)))
        dyn_elements = self.build_fwd_elements(params, x0, us)
        c_as, c_bs = associative_scan(dynamic_operator, dyn_elements)
        return c_bs  
    
    def run_dynamics_feedback(self, params, x0, us, a_term, Ks):
        params_fb = params._replace(a = params.a - jax.vmap(jnp.matmul, in_axes = (None,0))(params.b, Ks))
        dyn_elements = self.build_fwd_elements(params_fb, x0, us)
        c_as, c_bs = associative_scan(dynamic_operator, dyn_elements)
        return c_bs  

    def save_params(self, dyn_params: VanillaParams, saving_dir: str, flag: Optional[int]):
        a_eig, _ = np.linalg.eig(dyn_params.a)
        np.savez(f"{saving_dir}/dyn_prms_{flag}.npz", w_eig=a_eig, **dyn_params._asdict())
    
