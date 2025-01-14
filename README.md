# ilqr_vae_jax


This repository contains code to fit input-driven, potentially nonlinear dynamical systems, to time series.
It does so by fitting sequential variational auto-encoder, with a generative model that is an input-driven dynamical system. 
We implement several options for the prior over inputs, the dynamics, and the likelihood function.
We also implement two options for the encoder in the VAE : one implements the  [ilQR-VAE paper](https://openreview.net/forum?id=wRODLDHaAiW) and uses the iLQR algorithm to perform inference over the inputs, while the other one uses a biRNN, as in the [LFADS paper](https://www.nature.com/articles/s41592-018-0109-9).



## Features

- **VAE module**: The main VAE module is implemented in the vae.py file. 
- **Custom Dynamics, Prior and Likelihood Models**: Different choices for these modules can be found in the dynamics, prior, and likelihood files. 
- **Custom encoder module**: Two potential encoders are implemented -- an iLQR-based encoder and a biRNN-based encoder. 


## Installation

```bash
git clone git@github.com:marineschimel/ilqr_vae_jax.git
cd ilqr_vae_jax
python3 -m venv .venv
source ./.venv/bin/activate
pip install -e .
pip install -r requirements.txt
```
