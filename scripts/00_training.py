# %%
## Standard libraries
import os
import math
import numpy as np
from typing import Any
from functools import partial


# %%
## tqdm for progress bars
from tqdm.auto import tqdm

# %%
## To run JAX on TPU in Google Colab, uncomment the two lines below
# import jax.tools.colab_tpu
# jax.tools.colab_tpu.setup_tpu()

## JAX
import jax
import jax.numpy as jnp
from jax import random
from flax.core import freeze, unfreeze, FrozenDict
from einops import rearrange, parse_shape
from clu import metrics
from ml_collections import ConfigDict
import wandb

# Seeding for random operations
main_rng = random.PRNGKey(42)

## Flax (NN in JAX)
import flax
from flax import linen as nn
from flax import struct
from flax.training import train_state, checkpoints
from flax.training import orbax_utils
import orbax.checkpoint
import optax

import tensorflow as tf
from keras.datasets import mnist

# Path to the folder where the datasets are/should be downloaded (e.g. MNIST)
# DATASET_PATH = "../../data"
# Path to the folder where the pretrained models are saved
# CHECKPOINT_PATH = "../../saved_models/tutorial12_jax"

print("Device:", jax.devices()[0])

config = ConfigDict({
    "EPOCHS": 20,
    "SEED": 42,
    "LEARNING_RATE": 3e-4,
    "C_IN": 1,
    "C_HIDDEN": 64, 
})

wandb.init(project="PixelCNN",
           name="Gated",
           config=dict(config),
           mode="online",
           job_type="training",
           )
config = wandb.config

# %% [markdown]
# ## Dataset
# 
# > We'll be using MNIST in the range $[0,255]$ (won't put them in $[0,1]$ as is usually done).

# %%
(X_train, Y_train), (X_val, Y_val) = mnist.load_data()

dst_train = tf.data.Dataset.from_tensor_slices((X_train[...,None]))
dst_val = tf.data.Dataset.from_tensor_slices((X_val[...,None]))

dst_train_rdy = dst_train.shuffle(buffer_size=500, seed=42, reshuffle_each_iteration=True)\
                         .batch(256)\
                         .prefetch(1)
dst_val_rdy = dst_val.batch(512)\
                       .prefetch(1)

# %% [markdown]
# ## Masked Convolution
# 
# They define a module to do the masked convolution, but here's already a mask attribute in Flax's `Conv`. Anyway, we'll mimic as of now:

# %%
class MaskedConvolution(nn.Module):
    features: int
    mask: jnp.array
    dilation: int = 1

    @nn.compact
    def __call__(self,
                 inputs,
                 **kwargs,
                 ):
        if len(self.mask.shape) == 2:
            mask_ext = self.mask[...,None,None]
            mask_ext = jnp.tile(mask_ext, reps=(1,1,inputs.shape[-1], self.features))
        else:
            mask_ext = self.mask
        
        outputs = nn.Conv(features=self.features, kernel_size=self.mask.shape[:2], kernel_dilation=self.dilation, mask=mask_ext)(inputs)
        return outputs

# %% [markdown]
# ### Vertical & Horizontal Convolution

class VerticalStackConvolution(nn.Module):
    features: int
    kernel_size: int
    mask_center: bool = False
    dilation: int = 1

    def setup(self):
        mask = jnp.ones(shape=(self.kernel_size, self.kernel_size), dtype=jnp.float32)
        mask = mask.at[self.kernel_size//2+1:].set(0)
        if self.mask_center:
            mask = mask.at[self.kernel_size//2].set(0)
        self.conv = MaskedConvolution(features=self.features, mask=mask, dilation=self.dilation)

    def __call__(self, inputs):
        return self.conv(inputs)

# %%
class HorizontalStackConvolution(nn.Module):
    features: int
    kernel_size: int
    mask_center: bool = False
    dilation: int = 1

    def setup(self):
        ## First dim is 1 because we only look for pixels in the same row
        mask = jnp.ones(shape=(1, self.kernel_size), dtype=jnp.float32)
        mask = mask.at[0,self.kernel_size//2+1:].set(0)
        if self.mask_center:
            mask = mask.at[0,self.kernel_size//2].set(0)
        self.conv = MaskedConvolution(features=self.features, mask=mask, dilation=self.dilation)

    def __call__(self, inputs):
        return self.conv(inputs)

class GatedMaskedConv(nn.Module):
    dilation: int = 1

    @nn.compact
    def __call__(self, v_stack, h_stack):
        c_in = v_stack.shape[-1]

        conv_vert = VerticalStackConvolution(features=2*c_in,
                                             kernel_size=3,
                                             mask_center=False,
                                             dilation=self.dilation)
        conv_horiz = HorizontalStackConvolution(features=2*c_in,
                                                kernel_size=3,
                                                mask_center=False,
                                                dilation=self.dilation)
        conv_vert_to_horiz = nn.Conv(features=2*c_in,
                                     kernel_size=(1,1))
        conv_horiz_1x1 = nn.Conv(features=c_in,
                                 kernel_size=(1,1))

        ## Vertical stack
        v_stack_feat = conv_vert(v_stack)
        # v_val, v_gate = v_stack_feat.split(2, axis=-1)
        v_val, v_gate = jnp.split(v_stack_feat, 2, axis=-1)
        v_stack_out = nn.tanh(v_val) * nn.sigmoid(v_gate)

        ## Horizontal stack
        h_stack_feat = conv_horiz(h_stack)
        h_stack_feat = h_stack_feat + conv_vert_to_horiz(v_stack_feat)
        # h_val, h_gate = h_stack_feat.split(2, axis=-1)
        h_val, h_gate = jnp.split(h_stack_feat, 2, axis=-1)
        h_stack_out = nn.tanh(h_val) * nn.sigmoid(h_gate)
        h_stack_out = conv_horiz_1x1(h_stack_out)
        h_stack_out = h_stack_out + h_stack

        return v_stack_out, h_stack_out

class PixelCNN(nn.Module):
    c_in: int
    c_hidden: int

    def setup(self):
        ## Initial convolutions skipping the center pixel
        self.conv_vstack = VerticalStackConvolution(self.c_hidden, kernel_size=3, mask_center=True)
        self.conv_hstack = HorizontalStackConvolution(self.c_hidden, kernel_size=3, mask_center=True)

        ## Gated convolution blocks. We use dilation instead of downsampling
        self.conv_layers = [
            GatedMaskedConv(),
            GatedMaskedConv(dilation=2),
            GatedMaskedConv(),
            GatedMaskedConv(dilation=4),
            GatedMaskedConv(),
            GatedMaskedConv(dilation=2),
            GatedMaskedConv(),
        ]

        ## Output classification convolution. We have to predict a probability for each possible pixel value.
        self.conv_out = nn.Conv(self.c_in*256, kernel_size=(1,1))

    def pred_logits(self, inputs):
        ## 1. Scale input from [0,255] to [-1,1]
        inputs = (inputs.astype(jnp.float32) / 255.0)*2 - 1

        ## 2. Initial convolutions
        v_stack = self.conv_vstack(inputs)
        h_stack = self.conv_hstack(inputs)

        ## 3. Gated convolutions (This can probably be a `scan`)
        for layer in self.conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)

        ## 4. 1x1 classification convolution
        outputs = self.conv_out(nn.elu(h_stack))

        ## 5. Put into proper shape
        outputs = rearrange(outputs, "b h w (c_in cls) -> b h w c_in cls", **parse_shape(inputs, "b h w c_in"))

        return outputs

    def __call__(self, inputs):
        """Forward pass with bpd calculation."""
        logits = self.pred_logits(inputs)
        labels = inputs.astype(jnp.int32)
        nll = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        bpd = nll.mean() + np.log2(np.exp(1))
        return bpd

    def sample(self, img_shape, rng, img=None):
        ## 1. Create empty image
        if img is None:
            img = jnp.zeros(shape=img_shape, dtype=jnp.int32) - 1

        ## 2. Generation loop
        get_logits = jax.jit(lambda inp: self.pred_logits(inp))
        for h in tqdm(range(img_shape[1]), leave=False):
            for w in tqdm(range(img_shape[2])):
                for c in tqdm(range(img_shape[3])):
                    ## 2.1. Skip if not to be filled (-1)
                    if (img[:,h,w,c] != -1).all().item():
                        continue
                    logits = get_logits(img)
                    logits = logits[:,h,w,c,:]
                    rng, pix_rng = random.split(rng)
                    img = img.at[:,h,w,c].set(random.categorical(pix_rng, logits, axis=-1))
        return img

# %%
model = PixelCNN(c_in=1, c_hidden=64)
inp = jnp.zeros(shape=(1,28,28,1))
variables = model.init(random.PRNGKey(42), inp)

# %%
@struct.dataclass
class Metrics(metrics.Collection):
    """Collection of metrics to be tracked during training."""
    loss: metrics.Average.from_output("loss")

# %%
class TrainState(train_state.TrainState):
    metrics: Metrics
    state: FrozenDict

# %%
def create_train_state(module, key, tx, input_shape):
    """Creates the initial `TrainState`."""
    variables = module.init(key, jnp.ones(input_shape))
    state, params = variables.pop('params')
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        state=state,
        tx=tx,
        metrics=Metrics.empty()
    )

# %%
@partial(jax.jit, static_argnums=2)
def train_step(state, batch, return_grads=False):
    """Train for a single step."""
    img = batch
    def loss_fn(params):
        ## Forward pass through the model
        loss = state.apply_fn({"params": params, **state.state}, img)
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics_updates = state.metrics.single_from_model_output(loss=loss)
    metrics = state.metrics.merge(metrics_updates)
    state = state.replace(metrics=metrics)
    if return_grads: return state, grads
    else: return state

# %%
@jax.jit
def compute_metrics(*, state, batch):
    img = batch
    def loss_fn(params):
        ## Forward pass through the model
        loss = state.apply_fn({"params": params, **state.state}, img)
        return loss
    
    metrics_updates = state.metrics.single_from_model_output(loss=loss_fn(state.params))
    metrics = state.metrics.merge(metrics_updates)
    state = state.replace(metrics=metrics)
    return state

# %%

# %%
state = create_train_state(PixelCNN(c_in=config.C_IN, c_hidden=config.C_HIDDEN), random.PRNGKey(config.SEED), optax.adam(config.LEARNING_RATE), input_shape=(1,28,28,1))

# %%
param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
print(param_count)
wandb.run.summary["param_count"] = param_count

# %%
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(state)

# %%
metrics_history = {
    "train_loss": [],
    "val_loss": [],
}

batch = next(iter(dst_train_rdy.as_numpy_iterator()))

s1, grads = train_step(state, batch, return_grads=True)

s1 = compute_metrics(state=state, batch=batch)

for epoch in range(config.EPOCHS):
    ## Training
    for batch in dst_train_rdy.as_numpy_iterator():
        state, grads = train_step(state, batch, return_grads=True)
        # wandb.log({f"{k}_grad": wandb.Histogram(v) for k, v in flatten_params(grads).items()}, commit=False)
        # state = compute_metrics(state=state, batch=batch)
        # break

    ## Log the metrics
    for name, value in state.metrics.compute().items():
        metrics_history[f"train_{name}"].append(value)
    
    ## Empty the metrics
    state = state.replace(metrics=state.metrics.empty())

    ## Evaluation
    for batch in dst_val_rdy.as_numpy_iterator():
        state = compute_metrics(state=state, batch=batch)
        # break
    for name, value in state.metrics.compute().items():
        metrics_history[f"val_{name}"].append(value)
    state = state.replace(metrics=state.metrics.empty())
    
    ## Obtain activations of last validation batch
    # _, extra = forward_intermediates(state, batch[0])
    
    ## Checkpointing
    if metrics_history["val_loss"][-1] <= min(metrics_history["val_loss"]):
        orbax_checkpointer.save(os.path.join(wandb.run.dir, "model-best"), state, save_args=save_args, force=True) # force=True means allow overwritting.
        # orbax_checkpointer.save("model-best", state, save_args=save_args, force=True) # force=True means allow overwritting.
    # orbax_checkpointer.save(os.path.join(wandb.run.dir, f"model-{epoch+1}"), state, save_args=save_args, force=False) # force=True means allow overwritting.

    # wandb.log({f"{k}": wandb.Histogram(v) for k, v in flatten_params(state.params).items()}, commit=False)
    # wandb.log({f"{k}": wandb.Histogram(v) for k, v in flatten_params(extra["intermediates"]).items()}, commit=False)
    wandb.log({"epoch": epoch+1, **{name:values[-1] for name, values in metrics_history.items()}})
    print(f'Epoch {epoch} -> [Train] Loss: {metrics_history["train_loss"][-1]} [Val] Loss: {metrics_history["val_loss"][-1]}')
    # break


orbax_checkpointer.save(os.path.join(wandb.run.dir, "model-final"), state, save_args=save_args, force=True) # force=True means allow overwritting.
wandb.finish()
