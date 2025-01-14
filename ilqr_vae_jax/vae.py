import jax, time, chex, itertools, os
from jax import random
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from jax.random import PRNGKey
import optax, sys
from typing import Any, NamedTuple
# sys.path.append('..')
from ilqr_vae_jax.encoder import *
from ilqr_vae_jax import utils
from ilqr_vae_jax.utils import *
from ilqr_vae_jax.defaults import default_training_hparams
from functools import partial
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import functools
import pdb
import pickle
from jax.experimental import mesh_utils
##posterior sample, entropy, KL, elbo


class VAE:
    """This is the main class for the VAE. 
    It is responsible for training the model and evaluating the ELBO.
    Inputs : 
    - dynamics : the dynamics model (e.g vanilla RNN, GRU...)
    - prior : the prior model (e.g Gaussian or Student prior for the latent input)
    - likelihood : the likelihood model (e.g Poisson for spikes data or Gaussian for real valued data)
    - encoder : the encoder model (e.g a biRNN for LFADS, or iLQR)
    - coupler : the coupler model (this is just for LFADS, it couples the encoder and the dynamics, in a way that mirrors the structure of a Kalman filter)
    - dataloader : the dataloader object that will be used to sample data
    - dims : the dimensions of the data / model
    Functions : 
    - ELBO computation. The trianing objective is the negative ELBO + regularizer (which we are trying to minimize). 
    - Training loop"""
    def __init__(self, dynamics: Any, prior: Any, likelihood: Any, encoder: Any, coupler: Any, dataloader: Any, dims: Dims, inputs_allowed: int = 1, saving_dir = 'results', training_hparams = default_training_hparams, shmap = False, checkpoint = None, dropout = None, dropout_rate = None, params = None):
        self.shmap = shmap
        self.dynamics = dynamics
        self.prior = prior
        self.likelihood = likelihood
        self.encoder = encoder
        self.coupler = coupler
        self.dims = dims
        self.dataloader = dataloader
        self.init_seed = 4
        self.dims = dims
        self.inputs_allowed = inputs_allowed
        self.saving_dir = saving_dir
        self.shmap = shmap #this option is useful if we have a multi-CPU machine and we are using the iLQR algorithm. In this case, we can parallelize using shmap, which distributes the ELBO computation (and thus iLQR computation) across CPUs and can lead to a big speed increase.
        self.training_hparams = training_hparams
        self.save_every = 1000
        self.checkpoint = checkpoint
        self.dropout = dropout
        self.dropout_rate = 0.1 if dropout_rate is None else dropout_rate 
        self.params = params
        self.metrics = []
        self.max_grad_norm = 100.0
        self.warm_start = True
        self.flag = None
        self.verbose = True 
        self.losses = []

    def clip_single_example(self, grads):
        return utils.clip_single_example(grads, self.max_grad_norm) if self.training_hparams.clip_grads else jax.tree_map(lambda g : jnp.mean(g, axis = 0), grads)

    def initialize_params(self, key):
        keys = random.split(key, 5)
        prior_params = self.prior.initialize_params(keys[0])
        likelihood_params = self.likelihood.initialize_params(keys[1])
        encoder_params = self.encoder.initialize_params(keys[2])
        dynamics_params = self.dynamics.initialize_params(keys[3])
        coupling_params = self.coupler.initialize_params(key = keys[4])
        return VAEParams(coupling_params = coupling_params, encoder_params = encoder_params,dyn_params = dynamics_params, prior_params = prior_params, likelihood_params = likelihood_params)  

    def posterior_sample(self, params: VAEParams, ic_mean, ic_logstd, us_mean, us_logstd, ts, exts, key):
        # run data through the encoder
        # sample from the posterior
        key, subkey = random.split(key)
        subkey, subsubkey = random.split(subkey)
        num_samples = self.training_hparams.num_samples
        keys = random.split(key, (num_samples,1) )
        horizon = us_mean.shape[0]
        subkeys = random.split(subkey, (num_samples,horizon))
        subsubkeys = random.split(subsubkey, (num_samples,horizon))
        ic_samples = jax.vmap(self.encoder.ic_posterior_sample, in_axes = (0, None, None, None))(keys, ic_mean[None,...], ic_logstd[None,...], self.dims.n)
        ic_samples = ic_samples.reshape(num_samples, self.dims.n)
        x_samples, c_samples, cs_mean, cs_logstd = jax.vmap(self.coupler.run_coupled_dyn, in_axes = (None, 0, None, None, None, 0))(params, ic_samples, (us_mean, us_logstd), (ts, exts), self.dynamics, subkeys)
        pre_o_samples = jax.vmap(self.likelihood.sample_pre_os, in_axes = (None, 0))(params.likelihood_params, x_samples) ##so this should also include x0 in theory
        o_samples = jax.vmap(self.likelihood.sample_os, in_axes = (None, 0, 0))(params.likelihood_params, pre_o_samples, subsubkeys)
        return ic_samples, c_samples, x_samples, o_samples, pre_o_samples, cs_logstd, cs_mean

    def get_kl_warmup_fun(self):
        kl_warmup_start = 0 
        kl_warmup_end = self.training_hparams.kl_warmup_end
        kl_min = 0
        kl_max = 1
        def kl_warmup(batch_idx):
            progress_frac = ((batch_idx - kl_warmup_start) /
                      (kl_warmup_end - kl_warmup_start))
            kl_warmup = np.where(batch_idx < kl_warmup_start, kl_min,
                          (kl_max - kl_min) * progress_frac + kl_min)
            return np.where(batch_idx > kl_warmup_end, kl_max, kl_warmup)
        return kl_warmup

    def kl_divergence(self, params):
        raise NotImplementedError('direct evaluation of this function is not implemented')

    def entropy(self, gaussian_cov, dimension):
        ld = jnp.sum(jnp.log(gaussian_cov))
        return 0.5*(dimension * (1 + jnp.log(2*jnp.pi)) + ld) 

    def neg_elbo(self, params, data_enc, data_dec, key, kl_warmup):
        """This function computes the ELBO for one data sequence.
        ELBO = \mathbb{E}_q(u) [\log p(y|x(u)) + KL(q(u)||p(u))]
            = \mathbb{E}_q(u) [\log p(y|x(u)) + \log p(u) - \log q(u)] 
            = \mathbb{E}_q(u) [log likelihood + log prior + entropy] 
        In practice, we also include a function \beta(iteration) that is weighing the KL term. 
        This function computes the loss for a single training example : we then vmap it over a batch in batch_elbo."""
        ts, exts, _ = data_enc
        _, _, obs_dec = data_dec
        data_dec = (ts, exts, obs_dec)
        key, subkey = random.split(key)
        ic_mean, ic_logstd, us_mean, us_logstd = self.encoder.get_posterior_mean_and_cov(params, key, data_enc, self.inputs_allowed)
        ic_samples, c_samples, x_samples, o_samples, pre_o_samples, cs_logstd, cs_mean = self.posterior_sample(params, ic_mean, ic_logstd, us_mean, us_logstd, ts, exts, subkey)
        cs_cov = jax.vmap(lambda x : jax.nn.softplus(x)**2 + 1e-3)(cs_logstd)
        ic_cov = jax.nn.softplus(ic_logstd)**2 + 1e-3
        us_entropy = self.inputs_allowed*jnp.sum(jax.vmap(self.entropy, in_axes = (0, None))(cs_cov, self.dims.m))
        us_entropy += self.inputs_allowed*0.5*jnp.sum((cs_mean[None] - c_samples)**2/cs_cov)
        ic_entropy = self.entropy(ic_cov, self.dims.n)
        entropy =  (ic_entropy + us_entropy)
        lp_ic = jnp.sum(jax.vmap(self.prior.log_prior_ic, in_axes = (None, 0))(params.prior_params, ic_samples))
        lp_us = jnp.sum(jax.vmap(self.prior.log_prior, in_axes = (None, 0))(params.prior_params, c_samples))
        ll = jnp.sum(jax.vmap(self.likelihood.log_likelihood, in_axes = (None, 0, None))(params.likelihood_params, x_samples, data_dec)) #not o because we learn readout!!
        return -(ll + kl_warmup*(lp_us + lp_ic + entropy))/(self.training_hparams.num_samples*self.dims.horizon*self.dims.n_out), (ll, entropy,  (lp_us + lp_ic), (jnp.mean(c_samples, axis = 0), jnp.mean(x_samples, axis = 0), jnp.mean(pre_o_samples, axis = 0)))

    def get_sample(self, params, data, key):
        data_enc, data_dec = data
        ts, exts, obs_enc = data_enc
        _, _, obs_dec = data_dec
        T_sample = obs_enc.shape[0]
        data_dec = (ts, exts, obs_dec)
        key, subkey = random.split(key)
        ic_mean, ic_logstd, us_mean, us_logstd = self.encoder.get_posterior_mean_and_cov(params, key, data_enc[:self.dims.horizon], self.inputs_allowed)
        ic_samples, c_samples, x_samples, o_samples, pre_o_samples, cs_logstd, cs_mean = self.posterior_sample(params, ic_mean, ic_logstd, us_mean, us_logstd, ts, exts, subkey)
        lp_ic = jnp.sum(jax.vmap(self.prior.log_prior_ic, in_axes = (None, 0))(params.prior_params, ic_samples))
        cs_cov = jax.vmap(lambda x : jax.nn.softplus(x)**2 + 1e-3)(cs_logstd)
        ic_cov = jax.nn.softplus(ic_logstd)**2 + 1e-3 
        lp_us = jnp.sum(jax.vmap(self.prior.log_prior, in_axes = (None, 0))(params.prior_params, c_samples))
        ll = jnp.sum(jax.vmap(self.likelihood.log_likelihood, in_axes = (None, 0, None))(params.likelihood_params, x_samples, data_dec))
        us_entropy = self.inputs_allowed*jnp.sum(jax.vmap(self.entropy, in_axes = (0, None))(cs_cov, self.dims.m))
        ic_entropy = self.entropy(ic_cov, self.dims.n)
        entropy =  (ic_entropy + us_entropy)
        preds = np.mean(pre_o_samples, axis = 0)
        true = obs_dec
        var_true = np.var(true, axis = 0)
        r2 = np.mean(1 - np.mean((preds - true)**2, axis = 0)/var_true[None,...])
        return jnp.mean(c_samples, axis = 0), jnp.mean(x_samples, axis = 0), jnp.mean(pre_o_samples, axis = 0), (ll, entropy,  (lp_us + lp_ic))

    def batch_neg_elbo(self, params, batch_data, keys, kl_warmup):
        data_batch_enc, data_batch_dec = batch_data
        losses, (log_likelihood, entropy, log_prior, samples) = jax.vmap(self.neg_elbo, in_axes = (None, 0, 0, 0, None))(params, data_batch_enc, data_batch_dec, keys, kl_warmup)
        return jnp.mean(losses) + self.training_hparams.regularizer(params), (jnp.ones(1)*(jnp.mean(losses) + self.training_hparams.regularizer(params)), (jnp.ones(1)*jnp.mean(log_likelihood), jnp.ones(1)*jnp.mean(entropy), jnp.ones(1)*jnp.mean(log_prior), samples))

    @partial(jax.jit, static_argnums=(0,3,))
    def train_step(self, params, opt_state, solver, batch_data, key, kl_warmup):
        keys = jax.random.split(key, self.training_hparams.batch_size)
        data_batch_enc, data_batch_dec = batch_data
        def per_example_clip(grads):
            return jax.tree_map(self.clip_single_example, grads)
        all_losses, grads = jax.value_and_grad(self.batch_neg_elbo, has_aux=True)(params, batch_data, keys, kl_warmup)
        loss, losses = all_losses
        _, losses = all_losses
        loss, (ll_term, entropy_term, lp_term, samples) = losses 
        updates, opt_state = solver.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, grads), opt_state,  jnp.mean(jnp.asarray(loss)), jnp.mean(jnp.asarray(ll_term)), jnp.mean(jnp.asarray(entropy_term)), jnp.mean(jnp.asarray(lp_term)), samples

    def val_grad_elbo(self, keys, data_batch, params, kl_warmup, data_axis_name):
        all_losses, grads = jax.value_and_grad(self.batch_neg_elbo, has_aux=True)(params, data_batch, keys, kl_warmup)
        grads =  jax.lax.pmean(grads, axis_name =data_axis_name) 
        _, loss_terms = all_losses
        return loss_terms, grads

    @partial(jax.jit, static_argnums=(0,3,))
    def train_step_shmap(self, params, opt_state, solver, data_batch, key, kl_warmup):
        """This function is still in development : the idea is to use shmap to parallelize over devices (e.g CPUs) in cases
    where data parallelization is more efficient that GPU parallelization (e.g with the naive iLQR implementation).
    This will only be called to train the model if the shmap option is on.
    Right now the parallelization doesn't work exactly as intended so this function will not return exactly the same output as train_step."""
        device_array = np.array(jax.devices())
        data_axis_name = "data"
        keys = jax.random.split(key, self.training_hparams.batch_size) #(n_devices, batch_size // n_devices))
        mesh = Mesh(device_array,  (data_axis_name,))
        s_fn = shard_map(
        functools.partial(self.val_grad_elbo, params=params, kl_warmup=kl_warmup, data_axis_name = data_axis_name),
        mesh,
        in_specs=(P(data_axis_name), P(data_axis_name)),
        out_specs=(P(data_axis_name), P()),
        check_rep=False,
    )
        losses, grads =  s_fn(keys, data_batch)
        loss, (ll_term, entropy_term, lp_term, samples) = losses
        updates, opt_state = solver.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, grads), opt_state, jnp.mean(jnp.asarray(loss)), jnp.mean(jnp.asarray(ll_term)), jnp.mean(jnp.asarray(entropy_term)), jnp.mean(jnp.asarray(lp_term)), samples

    @partial(jax.jit, static_argnums=(0,))
    def get_samples(self, params, keys, data_batch):
        samples = jax.vmap(self.get_sample, in_axes = (None, 0, 0))(params, data_batch, keys)
        s1, s2, pre_o_samples,  (ll, entropy,  lp) = samples
        _, data_dec = data_batch
        _, _, obs_dec = data_dec
        true = obs_dec.reshape(-1, self.dims.n_out)
        preds = pre_o_samples.reshape(-1,self.dims.n_out)
        var_true = np.var(true, axis = 0)
        test_ll = self.likelihood.get_metric(preds, true)
        return s1, s2, pre_o_samples, (test_ll, ll, entropy, lp)


    def train(self):
        """This is the training loop : it does not have an explicit output but updates parameters at every iteration."""
        if self.shmap :
            train_fn = self.train_step_shmap
            data_axis_name = "data"
            device_array = np.array(jax.devices())
            mesh = Mesh(device_array,  (data_axis_name,))
            sample_fn = shard_map(
        self.get_samples,
        mesh,
        in_specs=(P(), P(data_axis_name), P(data_axis_name)),
        out_specs=(P()),
        check_rep=False,
    )
            print(f'WARNING : shmap is on so we need to ensure that the number of active devices is {self.training_hparams.batch_size}')
        else :
            train_fn = self.train_step
            sample_fn = self.get_samples
        masking_fn = utils.id_fun if self.dropout is None else utils.coordinated_dropout_fun if self.dropout == "coordinated" else utils.simple_dropout_fun
        saving_dir = self.saving_dir
        self.dataloader.save_test_data(f"{saving_dir}") # saves out the os
        test_data = self.dataloader.sample_test_data()
        test_data_batches = test_data, test_data
        num_test_datapoints = test_data[0].shape[0]
        total_loss, total_ll, total_entropy, total_lp = 0.0, 0.0, 0.0, 0.0
        key = jax.random.PRNGKey(self.init_seed)
        key, subkey = jax.random.split(key)
        params = self.initialize_params(subkey) if self.params is None else self.params
        dataloader = self.dataloader
        lr_scheduler = self.training_hparams.lr_scheduler
        solver = self.training_hparams.optimizer(
            learning_rate=lr_scheduler
        )  
        opt_state = solver.init(params)
        all_losses, norm_grads = [], []
        start = time.time()
        # initialize hidden states
        for epoch in range(self.training_hparams.num_epochs):
            num_batches = 0
            for ts, exts, obs, idxs in dataloader:
                num_batches += 1
                key, subkey = jax.random.split(key)
                subkey, subsubkey = jax.random.split(subkey)
                # (ts, exts, obs) = dataloader.sample_train_data(subkey, batch_size = self.training_hparams.batch_size) #something like that
                obs_enc, obs_dec = masking_fn(
                    subkey, obs, self.dropout_rate
                )  # this will incorporate dropout in the observations if it's on
                data_batches = (ts, exts, obs_enc), (ts, exts, obs_dec)
                kl_warmup = self.get_kl_warmup_fun()(
                    epoch * self.training_hparams.total_num_datapoints + num_batches
                )
                (
                    (params, grads),
                    opt_state,
                    loss,
                    ll_term,
                    entropy_term,
                    lp_term,
                    samples,
                ) = train_fn(
                    params, opt_state, solver, data_batches, subsubkey, kl_warmup
                )
                ##mean across samples : here samples should be a tuple of us, xs, ys samples, of size B x T x N (each batch element is already mean across samples)
            
                max_norm_grad = jnp.max(
                    jnp.asarray([jnp.linalg.norm(g) for g in jax.tree_leaves(grads)])
                )
                total_loss += loss
                all_losses.append(loss)
                norm_grads.append(max_norm_grad)
                total_ll += ll_term
                total_entropy += entropy_term
                total_lp += lp_term
            delta_time = time.time() - start
            if self.verbose:
                print(
                    f"epoch {epoch} | time = {time.time() - start}| loss: {total_loss / num_batches} ~ ll: {total_ll/ num_batches}. entropy: {total_entropy / num_batches}. log_prior: {total_lp/ num_batches}"
                )
            total_loss, total_ll, total_entropy, total_lp = 0.0, 0.0, 0.0, 0.0
            keys = jax.random.split(key, num_test_datapoints)
            time_pre_sample = time.time()
            u_samples, x_samples, pre_o_samples, (test_lls, lls, entropies, lps) = (
                sample_fn(params, keys, test_data_batches)
            )
            test_ll = self.likelihood.get_metric(
                pre_o_samples, test_data_batches[-1][-1]
            )
            flag = (
                self.flag
                if self.flag is not None
                else f"{int(epoch)}" if self.checkpoint is not None and epoch % self.checkpoint == 0 else ""
            )
            self.prior.save_params(params.prior_params, self.saving_dir, flag=flag)
            self.dynamics.save_params(params.dyn_params, self.saving_dir, flag=flag)
            self.likelihood.save_params(
                params.likelihood_params, self.saving_dir, flag=flag
            )
            np.save(f"{self.saving_dir}/inputs{flag}", u_samples)
            np.save(f"{self.saving_dir}/latents{flag}", x_samples)
            np.save(f"{self.saving_dir}/predictions{flag}", pre_o_samples)
            pickle.dump(params, open(f"{self.saving_dir}/params_{flag}.pkl", "wb"))
            np.savetxt(f"{self.saving_dir}/losses{flag}", all_losses)
            np.savetxt(
                f"{self.saving_dir}/eff_losses{flag}",
                jnp.where(
                    jnp.asarray(norm_grads) < self.max_grad_norm,
                    jnp.asarray(all_losses),
                    0,
                ),
            )
            self.params = params
            self.metrics.append(
                [test_ll, np.mean(lls), np.mean(entropies), np.mean(lps), delta_time]
            )
            self.losses = all_losses
            np.savetxt(f"{self.saving_dir}/metrics{flag}", self.metrics)
 