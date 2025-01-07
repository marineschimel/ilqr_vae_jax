# ilqr_vae_jax


This repository contains code to fit input-driven, potentially nonlinear dynamical systems, to time series.
It does so by fitting sequential variational auto-encoder, with a generative model that is an input-driven dynamical system. 
We implement several options for the prior over inputs, the dynamics, and the likelihood function.
We also implement two options for the encoder in the VAE : one implements the  [ilQR-VAE paper](https://openreview.net/forum?id=wRODLDHaAiW) and uses the iLQR algorithm to perform inference over the inputs, while the other one uses a biRNN, as in the [LFADS paper](https://www.nature.com/articles/s41592-018-0109-9).


It contains an implementation of the in Jax, as well as the building blocks to fit an LFADS

## Features

- **VAE module**: 
- **Custom Dynamics, Prior and Likelihood Models**: Easily extendable to different dynamics and likelihood models.
- **iLQR encoder module**: iLQR algorithm for trajectory optimization.
- **LFADS encoder/controller**: We


## Installation

To install the required dependencies, run:

```sh
pip install matplotlib scipy jax numpy joblib optax imageio pillow ipython


Requirements : 
- 
