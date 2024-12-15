from jax import numpy as jnp
from typing import NamedTuple, Optional, Any, List
import optax



class BiRNNParams(NamedTuple):
  fwd_rnn: NamedTuple
  bwd_rnn: NamedTuple
  readout: NamedTuple
  readout_ic: NamedTuple
  x0_fwd: jnp.ndarray
  x0_bwd: jnp.ndarray
  
  
class VAEParams(NamedTuple):
    dyn_params: NamedTuple
    prior_params: NamedTuple
    likelihood_params: NamedTuple
    encoder_params: NamedTuple
    coupling_params: NamedTuple

class VanillaParams(NamedTuple):
    a: jnp.ndarray
    b: jnp.ndarray
    bias: jnp.ndarray
    
 
class VanillaExtParams(NamedTuple):
    a: jnp.ndarray
    b: jnp.ndarray
    bias: jnp.ndarray
    b_ext: jnp.ndarray
   
    
class MGUParams(NamedTuple):
    wrx: jnp.ndarray
    wru: jnp.ndarray
    brx: jnp.ndarray
    bru: jnp.ndarray
    bc: jnp.ndarray
    wc: jnp.ndarray
    wru_ext: Optional[jnp.ndarray] = None
    
    
class GRUParams(NamedTuple):
    wru: jnp.ndarray
    wrx: jnp.ndarray
    wzx: jnp.ndarray
    wzu: jnp.ndarray
    whx: jnp.ndarray
    whu: jnp.ndarray
    br: jnp.ndarray
    bz: jnp.ndarray
    bh: jnp.ndarray
    wru_ext: Optional[jnp.ndarray] = None
    wzu_ext: Optional[jnp.ndarray] = None
    whu_ext: Optional[jnp.ndarray] = None
    
    
class ReadoutParams(NamedTuple):
    c: jnp.ndarray
    b: jnp.ndarray
    
class EncoderParams(NamedTuple):
  fwd_rnn: GRUParams
  bwd_rnn: GRUParams
  readout: ReadoutParams
  
class VAEHParams(NamedTuple):
    print_every: int
    save_every: int
    num_iterations: int
    num_samples: int
    lr: float
    batch_size: int
    num_steps: int

    
class CovParams(NamedTuple):
  sig_t: jnp.ndarray
  sig_ic: jnp.ndarray
  

  
class GRUControllerParams(NamedTuple):
  gru_params: GRUParams
  c0: jnp.ndarray
  readout: ReadoutParams
  

class Dims(NamedTuple):
  n: int
  m: int
  n_out: int
  horizon: int
  m_encoder: Optional[int] = None  #size of the output of the encoder 
  m_controller: Optional[int] = None  #size of the output of the controller
  n_controller: Optional[int] = None #size of the state of the controller
  n_encoder: Optional[int] = None #
  
class iLQRHParams(NamedTuple):
  n_beg: int
  dt: float
  use_linesearch: bool
  
  
class S5Layer(NamedTuple):
    Lambda_re: jnp.ndarray
    Lambda_im: jnp.ndarray
    B_bar: jnp.ndarray
    C_tilde: jnp.ndarray
    D: jnp.ndarray
    eff_Ks: jnp.ndarray
    
class TrainingHParams(NamedTuple):
  optimizer: optax.GradientTransformation
  num_epochs: int
  batch_size: int
  clip_grads: bool
  regularizer: Any
  lr_scheduler: Any
  kl_warmup_end: int
  num_samples: int
  total_num_datapoints: int

  
