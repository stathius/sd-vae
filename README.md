# Implementatuib for the paper "Disentangled generative models for robust dynamical system prediction"

This github repository contain code for the paper titles "Disentangled generative models for robust dynamical system prediction" that was presented in ICML 2023.

In the `main` branch there is code to replicate the phase-space experimen. The branch `rssm` have the code for the video pendulum experiments. Both branches contain code to generate data, train and evaluate the models.

# Dataset

We provide the necessary code to generated the datased used in the experiments of the paper. There are 3 jupyter notebooks in the `data_generation` one for each dynamical systems i.e. pendulum, Lotka-Volterra and 3-Body system. More datasets can be created for experimentation by changing the ODE parameters of each system in the code.

# Training 

All the models are implemented in `pytorch-1.09`, using `pytorch-lightning`. There are 3 training scripts. `train_LSTM.py` can be used for training LSTMs, `train_MLP.py` is both for MLP and SD-MLP models and `train_VAE.py` for VAE & SD-VAE models. Supervision of the latent space can be enabled and tuned using the `--use_supervision`, `--sup_loss_type` and `--sup_multiplier` parameters.

As an example the following command trains a SD-VAE model on the pendulum dataset using latent space supervision:

`train_VAE.py --rec_loss_type L1 --scheduler_min_lr 1e-08 --weight_decay 0 --monitor val/rec/0010 --model_dropout_pct 0.0 --max_epochs 2000 --samples_per_batch_train 1 --samples_per_batch_val 1 --samples_per_batch_test 10 --gpus 0 --num_workers 4 --use_wandb True --progress_bar_refresh_rate 100 --dataset pendulum-2 --coordinates phase_space --dataset_dt 0.01 --use_random_start True --noise_std 0.05 --model_input_size 50 --model_output_size 10 --project pendulum-2_n_vae_sup_tanh --model vae --model_hidden_size 400 200 --model_latent_size 16 --model_nonlinearity leaky --kld_scaling_type beta_fixed --beta 1e-06 --use_supervision True --model_use_extra_factors False --partition_latents False --sup_loss_type linear --sup_multiplier 0.1 --gradient_clip_val 0 --learning_rate 0.0001 --use_layer_norm False --batch_size 16 --scheduler_patience 60 --scheduler_factor 0.4 --early_stopping_patience 200 --seed 1`