from jax import numpy as jnp
from typing import NamedTuple, Optional, Any
from ilqr_vae_jax import dynamics
from ilqr_vae_jax.encoder import *
from ilqr_vae_jax.utils import *
from ilqr_vae_jax.typs import *
import optax

optimizer = optax.adam
batch_size = 32
num_samples = 10
lr_scheduler = optax.cosine_decay_schedule(init_value = 1e-3, decay_steps = 10000)
default_training_hparams = typs.TrainingHParams(num_samples = num_samples, kl_warmup_end = 1000, optimizer = optimizer, num_epochs = 10, batch_size = batch_size, clip_grads = True, regularizer = lambda params : 0, lr_scheduler = lr_scheduler, total_num_datapoints = None)
## optional dimension defaults
n_encoder = 64
n_controller = 64


class Dataloader():
    """This is a custom dataloader for the library. Note that this can be mofidied for specific applications,
    but the save_test_data and sample_test_data are required in the vae module"""
    def __init__(self, dims, data, batch_size, test_data):
        os, us, ext_us = data
        self.dims = dims
        self.Ts = jnp.concatenate([jnp.arange(np.shape(os)[1])[None] for _ in range(len(os))], axis = 0)
        self.ext_us = ext_us
        self.ys = os
        mean_ys = jnp.mean(self.ys.reshape((-1, self.ys.shape[-1])), axis = 0)
        std_ys = jnp.std(self.ys.reshape((-1, self.ys.shape[-1])), axis = 0)
        self.ys = (self.ys - mean_ys[None])/std_ys[None]
        self.total_samples = len(self.ys)
        self.batch_size = batch_size
        self.key = jax.random.PRNGKey(0)
        self.all_test_data = test_data
        
    def update_train_dataset(self, new_data):
        os, us, ext_us = new_data
        self.ys = jnp.concatenate([self.ys, os], axis = 0)
        mean_ys = jnp.mean(self.ys.reshape((-1, self.ys.shape[-1])), axis = 0)
        std_ys = jnp.std(self.ys.reshape((-1, self.ys.shape[-1])), axis = 0)
        self.ys = (self.ys - mean_ys[None])/std_ys[None]
        self.ext_us = jnp.concatenate([self.ext_us, ext_us], axis = 0)
        self.Ts = jnp.concatenate([jnp.arange(np.shape(self.ys)[1])[None] for _ in range(len(self.ys))], axis = 0)


    def save_test_data(self, saving_dir):
        pass
        
    def update_test_dataset(self, new_test_data):
        self.all_test_data = new_test_data
        
    def sample_test_data(self):
        os, us, ext_us = self.all_test_data
        return jnp.concatenate([jnp.arange(self.dims.horizon)[None,...] for _ in np.arange(len(os))], axis = 0), ext_us, os
    def __len__(self):
            return (self.total_samples + self.batch_size - 1) // self.batch_size
    def __iter__(self):
        key, subkey = jax.random.split(self.key)
        self.key = subkey
        idces = jax.random.permutation(self.key, jnp.arange(self.total_samples))
        self.ys = self.ys[idces]
        self.ext_us = self.ext_us[idces]
        for i in range(len(self)):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.total_samples)
            y, ext_us = self.ys[start:end], self.ext_us[start:end]
            yield self.Ts[start:end], ext_us, y, jnp.arange(start, end)