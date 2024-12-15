
import h5py
from jax import random, vmap
import jax.numpy as jnp
import numpy as np # original numpy
import sys
from ilqr_vae_jax import typs


def bmm(arr1, arr2):
    """Batch matrix multiplication"""
    return vmap(jnp.matmul)(arr1, arr2)
  
  
def keygen(key, nkeys):
  """Generate randomness that JAX can use by splitting the JAX keys.

  Args:
    key : the random.PRNGKey for JAX
    nkeys : how many keys in key generator

  Returns:
    2-tuple (new key for further generators, key generator)
  """
  keys = random.split(key, nkeys+1)
  return keys[0], (k for k in keys[1:])


def linear_readout(params, x):
  """Implement y = w x + b

  Arguments:
    params: a dictionary of params
    x: np array of input

  Returns:
    np array of output
  """
  return jnp.dot(params.c, x) + params.b


def initialize_readout_params(key, n, n_out, scale = 0.5):
  return typs.ReadoutParams(c = scale*random.normal(key, (n_out, n)), b = jnp.zeros((n_out, 1)))


def coordinated_dropout_fun(key, obs, dropout_rate):
  """Generate a mask for coordinated dropout.

  Arguments:
    key: a JAX PRNG key
    p: the probability of dropout
    n: the number of elements in the mask

  """
  keep_rate = 1 - dropout_rate
  mask = random.bernoulli(key, keep_rate, obs.shape)
  nan_mask = jnp.where(jnp.isclose(mask,1), jnp.nan, 1.0) #nan out the values we see to only have gradients for others
  return mask*obs, obs*nan_mask

def simple_dropout_fun(key, obs, dropout_rate):
  """Generate a mask for dropout.

  Arguments:
    key: a JAX PRNG key
    p: the probability of dropout
    obs: the inputs data
  """
  keep_rate = 1 - dropout_rate
  mask = random.bernoulli(key, keep_rate, obs.shape)
  return mask*obs, obs


def id_fun(key, obs, dropout_rate):
  return obs, obs

def expand_dims_to_match(a, b):
    while np.ndim(a) < np.ndim(b):
        a = a[:,np.newaxis]
    return a
  
def clip_single_example(grad, max_norm):
      #norm = jnp.sqrt(jnp.sum([jnp.sum(g**2) for g in jax.tree_leaves(grad)]))
      val = grad #jnp.max(grad.reshape(grad.shape[0],-1), axis = -1)#/jnp.sqrt(grad.reshape(grad.shape[0],-1).shape[-1])
      #val = jnp.where(grad < max_norm, grad, max_norm)
      norm = jnp.linalg.norm(grad.reshape(grad.shape[0],-1), axis = -1)
      norm = expand_dims_to_match(norm, grad)
      grad = jnp.nan_to_num(grad)
      return jnp.mean(grad * jnp.where(norm < max_norm, 1,0), axis = 0)

# dropout that also returns the binary mask
# def dropout(x, keep_prob, noise_shape=None, seed=None, name=None,
#             binary_tensor=None):  # pylint: disable=invalid-name
#     """Computes dropout.
#     With probability `keep_prob`, outputs the input element scaled up by
#     `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
#     sum is unchanged.
#     By default, each element is kept or dropped independently.  If `noise_shape`
#     is specified, it must be
#     [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
#     to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
#     will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
#     and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
#     kept independently and each row and column will be kept or not kept together.
#     Args:
#       x: A floating point tensor.
#       keep_prob: A scalar `Tensor` with the same type as x. The probability
#         that each element is kept.
#       noise_shape: A 1-D `Tensor` of type `int32`, representing the
#         shape for randomly generated keep/drop flags.
#       seed: A Python integer. Used to create random seeds. See
#         @{tf.set_random_seed}
#         for behavior.
#       name: A name for this operation (optional).
#     Returns:
#       A Tensor of the same shape of `x`.
#     Raises:
#       ValueError: If `keep_prob` is not in `(0, 1]` or if `x` is not a floating
#         point tensor.
#     """
#     with ops.name_scope(name, "dropout", [x]) as name:
#         x = ops.convert_to_tensor(x, name="x")
#         if not x.dtype.is_floating:
#             raise ValueError("x has to be a floating point tensor since it's going to"
#                              " be scaled. Got a %s tensor instead." % x.dtype)

#         # Only apply random dropout if mask is not provided

#         if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
#             raise ValueError("keep_prob must be a scalar tensor or a float in the "
#                              "range (0, 1], got %g" % keep_prob)

#         keep_prob = keep_prob if binary_tensor is None else 1 - keep_prob

#         keep_prob = ops.convert_to_tensor(keep_prob,
#                                           dtype=x.dtype,
#                                           name="keep_prob")

#         keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

#         if binary_tensor is None:
#             # Do nothing if we know keep_prob == 1
#             if tensor_util.constant_value(keep_prob) == 1:
#                 return x, None

#             noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
#             # uniform [keep_prob, 1.0 + keep_prob)
#             random_tensor = keep_prob
#             random_tensor += random_ops.random_uniform(noise_shape,
#                                                        seed=seed,
#                                                        dtype=x.dtype)
#             # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
#             binary_tensor = math_ops.floor(random_tensor)
#         else:
#             # check if binary_tensor is a tensor with right shape
#             binary_tensor = math_ops.cast(binary_tensor, dtype=x.dtype)
#             # pass

#         ret = math_ops.div(x, keep_prob) * binary_tensor
#         # if context.in_graph_mode():
#         ret.set_shape(x.get_shape())
#         return ret, binary_tensor



from typing import NamedTuple, Optional, Tuple

import chex
import jax.lax
import jax.numpy as jnp
import optax


class CustomOptimizerState(NamedTuple):
    opt_state: optax.OptState
    grad_norms: chex.Array
    ignored_grads_count: chex.Array = jnp.array(
        0, dtype=int
    )  # Keep track of how many gradients have been ignored.
    total_steps: chex.Array = jnp.array(0, dtype=int)  # Total number of optimizer steps.


def dynamic_update_ignore_and_grad_norm_clip(
    optimizer: optax.GradientTransformation,
    window_length: int = 100,
    factor_clip_norm: float = 5.0,
    factor_allowable_norm: float = 20.0,
) -> optax.GradientTransformation:
    """Wraps a gradient transform to dynamically clip the gradient norm, and ignore very large gradients.
    More specifically:
    1. Keep track of the last `window_length` gradient norms.
    2. Calculate the median gradient within the window norm. Call this `grad_median_norm`.
    2. If the current gradient is larger than `factor_allowable_norm * grad_median_norm`,
        then no gradient step occurs.
    3. Otherwise the gradient is clipped to a maximum norm of `factor_clip_norm * grad_median_norm`.
    """

    def init(params: chex.ArrayTree) -> CustomOptimizerState:
        opt_state = optimizer.init(params)
        grad_norms = jnp.ones(window_length) * float("nan")
        # After initialisation, for first third of window length take every gradient step.
        grad_norms = grad_norms.at[0 : int(window_length * 2 / 3)].set(1e30)

        return CustomOptimizerState(opt_state=opt_state, grad_norms=grad_norms)

    def update(
        grad: chex.ArrayTree, opt_state: CustomOptimizerState, params: chex.ArrayTree
    ) -> Tuple[chex.ArrayTree, CustomOptimizerState]:

        grad_norm = optax.global_norm(grad)
        grad_median_norm = jnp.nanmedian(opt_state.grad_norms)
        skip_update = (grad_norm > grad_median_norm * factor_allowable_norm) | (
            ~jnp.isfinite(grad_norm)
        )

        # Dynamic global norm clipping.
        global_norm_clip = optax.clip_by_global_norm(grad_median_norm * factor_clip_norm)
        global_norm_clip_state = global_norm_clip.init(params)
        grad = global_norm_clip.update(grad, global_norm_clip_state)[0]
        # Ensure gradients are still finite after normalization.
        grad = jax.tree_util.tree_map(
            lambda p: jnp.where(jnp.isfinite(p), p, jnp.zeros_like(p)), grad
        )

        updates, new_opt_state = optimizer.update(grad, opt_state.opt_state, params=params)

        # Update rolling window of gradient norms
        grad_norms = opt_state.grad_norms.at[:-1].set(opt_state.grad_norms[1:])
        grad_norms = grad_norms.at[-1].set(grad_norm)

        # If grad norm is too big then ignore update.
        updates, new_opt_state, ignored_grad_count = jax.lax.cond(
            skip_update,
            lambda: (
                jax.tree_map(jnp.zeros_like, updates),
                opt_state.opt_state,
                opt_state.ignored_grads_count + 1,
            ),
            lambda: (updates, new_opt_state, opt_state.ignored_grads_count),
        )

        state = CustomOptimizerState(
            opt_state=new_opt_state,
            ignored_grads_count=ignored_grad_count,
            grad_norms=grad_norms,
            total_steps=opt_state.total_steps + 1,
        )
        return updates, state

    return optax.GradientTransformation(init=init, update=update)


class OptimizerConfig(NamedTuple):
    """Optimizer configuration.

    If `dynamic_grad_ignore_and_clip` is True, then `max_global_norm` and `max_param_grad` have no effect.
    """

    peak_lr: float
    init_lr: Optional[float] = None
    optimizer_name: str = "adam"
    use_schedule: bool = False
    n_iter_total: Optional[int] = None
    n_iter_warmup: Optional[int] = None
    n_iter_decay: Optional[int] = None
    end_lr: Optional[float] = None
    max_global_norm: Optional[float] = None
    max_param_grad: Optional[float] = None
    dynamic_grad_ignore_and_clip: bool = False
    dynamic_grad_ignore_factor: float = 20.0
    dynamic_grad_norm_factor: float = 2.0
    dynamic_grad_norm_window: int = 100


def get_optimizer(optimizer_config: OptimizerConfig):
    """Create optimizer. Also returns the learning rate function,
    which is useful for logging the learning rate throughout training.
    """
    if optimizer_config.use_schedule:
        if optimizer_config.n_iter_decay is None:
            # Only get to end_value on final step.
            optimizer_config = optimizer_config._replace(n_iter_decay=optimizer_config.n_iter_total)
        elif optimizer_config.n_iter_decay > optimizer_config.n_iter_total:
            print(
                f"Warmup then cosine schedule of "
                f"{optimizer_config.n_iter_decay} will not "
                f"finish within the number of total training iter "
                f"{optimizer_config.n_iter_total}."
            )

        warmup_then_cosine = optax.warmup_cosine_decay_schedule(
            init_value=float(optimizer_config.init_lr),
            peak_value=float(optimizer_config.peak_lr),
            end_value=float(optimizer_config.end_lr),
            warmup_steps=optimizer_config.n_iter_warmup,
            decay_steps=optimizer_config.n_iter_decay,
        )
        lr = optax.join_schedules(
            schedules=[warmup_then_cosine, optax.constant_schedule(float(optimizer_config.end_lr))],
            boundaries=[optimizer_config.n_iter_decay],
        )
    else:
        lr = float(optimizer_config.peak_lr)

    main_grad_transform = getattr(optax, optimizer_config.optimizer_name)(lr)  # e.g. adam.

    if optimizer_config.dynamic_grad_ignore_and_clip:
        optimizer = dynamic_update_ignore_and_grad_norm_clip(
            optimizer=main_grad_transform,
            window_length=optimizer_config.dynamic_grad_norm_window,
            factor_clip_norm=optimizer_config.dynamic_grad_norm_factor,
            factor_allowable_norm=optimizer_config.dynamic_grad_ignore_factor,
        )
    else:
        grad_transforms = [optax.zero_nans()]
        if optimizer_config.max_param_grad:
            clipping_fn = optax.clip(float(optimizer_config.max_param_grad))
            grad_transforms.append(clipping_fn)
        if optimizer_config.max_global_norm:
            clipping_fn = optax.clip_by_global_norm(float(optimizer_config.max_global_norm))
            grad_transforms.append(clipping_fn)
        grad_transforms.append(main_grad_transform)
        optimizer = optax.chain(*grad_transforms)
    return optimizer, lr
