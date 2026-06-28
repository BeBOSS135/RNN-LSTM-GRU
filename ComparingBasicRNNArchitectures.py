# Google Drive
#from google.colab import drive
#drive.mount('/content/drive')
#DATA_DIR = '/content/drive/MyDrive/'

import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision, backend as K
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay

#DATA_DIR = './'
os.makedirs("figures", exist_ok=True)

# TF32 and Mixed Precision to Maximize Speed
# Save VRAM Eventhoug it is Not Needed
tf.config.experimental.enable_tensor_float_32_execution(True)
mixed_precision.set_global_policy('mixed_float16')

# Reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

RUN_LR_FINDER  = False # True to Find Optimal Learning Rate Range

# CNN to Reduce Signal Noise and Extract Features
# 32 Filters for Optimal CUda Core Memory Alignmnet
CNN_FILTERS  = 32
CNN_KERNEL   = 3
CNN_POOL     = 2

# RNN
# 128 Units for Efficient Tensor Core Matrix Multiplication
# Dropout to Avoid Overfitting
UNITS_L1     = 128
UNITS_L2     = 64
DROPOUT      = 0.3

# Weighted Penalty for Class Imbalance
ACTIVE_WEIGHT = 5.0
HUBER_DELTA   = 0.5

# Warmup for Stable Initial Wweight Updates
# 128 Batch Size to Optimize GPU Memory (through testing)
# Parallel Batch Execution with steps_per_execution=16 to Reduce Overhead
EPOCHS         = 100
PATIENCE_ES    = 5
MIN_EPOCHS     = 30   
MIN_DELTA_ES   = 1e-4  
                
WARMUP_FRAC    = 0.10 
INIT_LR        = 1e-3 # 1.5e-6 from LR Finder NOT Worth the Extra Training Time, Didnt Improve Performance in the 100 Epochs Training Window
MIN_LR         = 1e-6 # 1e-7
CLIP_NORM      = 1.0
BATCH_SIZE     = 128
EVAL_BATCH     = 512    
STEPS_PER_EXEC = 16
NOISE_INIT     = 0.02   

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

start_time = time.time()

# Data Loading and Normalisation Factors
print("=" * 42)
print("1. Loading data & Exploratory Plots")
print("=" * 42)

max_agg = float(np.loadtxt(os.path.join(DATA_DIR, 'CoffeeMachinemaxAgg.txt')))
max_app = float(np.loadtxt(os.path.join(DATA_DIR, 'CoffeeMachinemaxApp.txt')))
print(f"  max_agg = {max_agg:.2f} W   max_app = {max_app:.2f} W")

#Loading Straight to float32 to Avoid Casting Overhead in the GPU Pipeline
X_raw = pd.read_csv(os.path.join(DATA_DIR, 'Input_Data.txt'),  header=None).to_numpy(np.float32)
y_raw = pd.read_csv(os.path.join(DATA_DIR, 'Output_Data.txt'), header=None).to_numpy(np.float32)

n_samples, time_steps = X_raw.shape
MINUTES  = np.arange(time_steps)

print(f"  Dataset : {n_samples} sequences × {time_steps} timesteps")

# Signal Behavior Analysis
# Class Imbalance check
print("\n2. Exploratory plots")

active_idx   = np.where(y_raw.max(axis=1) > 0)[0]
inactive_idx = np.where(y_raw.max(axis=1) == 0)[0]

rng = np.random.default_rng(SEED)

chosen = (list(rng.choice(active_idx,   size=2, replace=False)) +
          list(rng.choice(inactive_idx, size=1, replace=False)))
rng.shuffle(chosen)

for i, idx in enumerate(chosen, 1):
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Aggregate (W)', color='steelblue')
    ax1.plot(MINUTES, X_raw[idx], color='steelblue', lw=1.5, label='Aggregate')
    ax1.tick_params(axis='y', labelcolor='steelblue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Coffee Machine (W)', color='tomato')
    ax2.plot(MINUTES, y_raw[idx], color='tomato', lw=1.5, ls='--', label='Coffee Machine')
    ax2.tick_params(axis='y', labelcolor='tomato')

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc='upper right')
    status = 'ON' if y_raw[idx].max() > 0 else 'OFF'
    plt.title(f'Exploratory Plot {i} | Sample {idx} | Device: {status}')
    plt.tight_layout()
    plt.savefig(f'figures/Exploratory_Plot_{i}.png')
    plt.show()
    plt.close(fig)

# Normalisation and Splitting
print("\n3. Normalising & Splitting")
# Inputs may be floored at 0 (aggregate power is non-negative).
X_norm  = np.clip(X_raw, 0.0, None)
y_norm  = y_raw
dX_norm = np.diff(X_norm, axis=1, prepend=X_norm[:, :1]) 
features = 2  # aggregate + derivative
# Constants for Loss Function Calculation
ACTIVE_THRESHOLD = tf.constant(0.0046, dtype=tf.float32)
_ACTIVE_WEIGHT   = tf.constant(ACTIVE_WEIGHT,   dtype=tf.float32)
_HUBER_DELTA     = tf.constant(HUBER_DELTA,     dtype=tf.float32)

# Stratified to Avoid Class Imbalances
def stratified_split(indices, train_r, val_r, r):
    idx  = r.permutation(indices)
    n_tr = int(len(idx) * train_r)
    n_va = int(len(idx) * val_r)
    return idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]

a_tr, a_va, a_te = stratified_split(active_idx,   TRAIN_RATIO, VAL_RATIO, rng)
i_tr, i_va, i_te = stratified_split(inactive_idx, TRAIN_RATIO, VAL_RATIO, rng)

train_idx = np.concatenate([a_tr, i_tr]); rng.shuffle(train_idx)
val_idx   = np.concatenate([a_va, i_va]); rng.shuffle(val_idx)
test_idx  = np.concatenate([a_te, i_te]); rng.shuffle(test_idx)

def extract_split_X(idx):
    return np.stack([X_norm[idx], dX_norm[idx]], axis=-1)  

def extract_split_y(arr, idx):
    return arr[idx, :, np.newaxis]                         

X_train = extract_split_X(train_idx)
X_val   = extract_split_X(val_idx)
X_test  = extract_split_X(test_idx)
y_train = extract_split_y(y_norm, train_idx)
y_val   = extract_split_y(y_norm, val_idx)
y_test  = extract_split_y(y_norm, test_idx)

# Performance Optimization
# Perfetch to Allow CPU to Prepare Next Batches While GPU Processes Current Ones
def make_dataset(X, y, training=False, batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    if training:
        ds = ds.shuffle(len(X), seed=SEED, reshuffle_each_iteration=True)

    # Drop Remainder Only During Training - Uniform Batch Size Required for steps_per_execution and XLA
    return ds.batch(batch_size, drop_remainder=training).prefetch(tf.data.AUTOTUNE)

# Custom Weighted Huber Loss
# Create Mask for Active States to Handle Class Imbalance
# Huber Loss for Robustness Against Signal Outliers
def weighted_huber(y_true, y_pred):
    active_mask  = tf.cast(y_true > ACTIVE_THRESHOLD, tf.float32)
    weights      = 1.0 + (_ACTIVE_WEIGHT - 1.0) * active_mask
    err          = y_true - y_pred
    abs_err      = tf.abs(err)
    huber_elem   = tf.where(
        abs_err <= _HUBER_DELTA,
        0.5 * tf.square(err),
        _HUBER_DELTA * (abs_err - 0.5 * _HUBER_DELTA)
    )
    # Scale Penalty by Signal Magnitude - High Wattage Events Punished More for Underprediction
    underpredict_penalty = tf.cast(err > 0, tf.float32) * active_mask * abs_err * (1.0 + 2.0 * y_true)
    inactive_mask        = 1.0 - active_mask
    sparsity_penalty     = 0.6 * inactive_mask * tf.abs(y_pred)
    return tf.reduce_mean(weights * huber_elem + underpredict_penalty + sparsity_penalty)

# Cosine Decay LR With Linear Warmup
# Linear Warmup to Stabilize Gradients
# Cosine Decay for Optimal Convergence
LR_DECAY_EPOCHS = 40
steps_per_epoch = len(train_idx) // BATCH_SIZE
total_steps     = LR_DECAY_EPOCHS * steps_per_epoch
warmup_steps    = max(1, int(total_steps * WARMUP_FRAC))
decay_steps     = total_steps - warmup_steps

class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, warmup_steps, decay_steps, min_lr):
        self.peak_lr      = float(peak_lr)
        self.warmup_steps = int(warmup_steps)
        self.decay_steps  = int(decay_steps)
        self.min_lr       = float(min_lr)
        self._cosine      = CosineDecay(peak_lr, decay_steps, alpha=min_lr / peak_lr)
        self._warmup_steps_f = tf.constant(float(warmup_steps), dtype=tf.float32)

    def __call__(self, step):
        step      = tf.cast(step, tf.float32)
        warmup_lr = self.peak_lr * (step / self._warmup_steps_f)
        cosine_lr = self._cosine(tf.maximum(step - self._warmup_steps_f, 0.0))
        return tf.cond(step < self._warmup_steps_f, lambda: warmup_lr, lambda: cosine_lr)

def make_lr_schedule(): return WarmupCosineDecay(INIT_LR, warmup_steps, decay_steps, MIN_LR)

# Scheduled Noise Layer to Improve Generalization
# Inject Gaussian Noise Directly on GPU for Speed
noise_std_var = tf.Variable(NOISE_INIT, dtype=tf.float32, trainable=False)

class DynamicNoiseLayer(layers.Layer):
    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(tf.shape(inputs), stddev=noise_std_var, dtype=tf.float32)
            noise = tf.cast(noise, inputs.dtype)
            noisy = inputs + noise
            agg = tf.maximum(noisy[..., 0:1], tf.cast(0.0, inputs.dtype))
            return tf.concat([agg, noisy[..., 1:]], axis=-1)
        return inputs

#Decrease Noise Linearly Over Epochs to Allow Fine Tuning in Later Stages
class NoiseDecayCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        new_noise = NOISE_INIT * (1.0 - (epoch / EPOCHS))
        noise_std_var.assign(max(0.0, new_noise))

# start_from_epoch Only Works with Newer Keras and TF Versions
class EarlyStoppingMinEpochs(EarlyStopping):
    def __init__(self, start_epoch=0, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_epoch:
            return  
        super().on_epoch_end(epoch, logs)

# Learning Rate Finder Callback to Map Ideal Learning Rate Range
class LRFinder(Callback):
    def __init__(self, min_lr=1e-6, max_lr=1e-1, steps=100):
        super().__init__()
        self.min_lr, self.max_lr, self.steps = min_lr, max_lr, steps
        self.factor = (max_lr / min_lr) ** (1 / steps)
        self.lrs, self.losses = [], []
        self.best_loss = 1e9

    def on_train_begin(self, logs=None):
        K.set_value(self.model.optimizer.lr, self.min_lr)

    def on_train_batch_end(self, batch, logs=None):
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)
        loss = logs['loss']
        self.losses.append(loss)
        if loss > self.best_loss * 4 or tf.math.is_nan(loss):
            self.model.stop_training = True
        if loss < self.best_loss:
            self.best_loss = loss
        K.set_value(self.model.optimizer.lr, lr * self.factor)

# Model Factory
# float 32 to Ensure Numerical Stability in Mixed Precision Training
def _build_model(rnn_class, name, lr_schedule):
    rnn_kw = {'kernel_initializer': 'glorot_uniform', 'recurrent_initializer': 'orthogonal'}
    inp = layers.Input(shape=(time_steps, features), name='input', dtype='float32')

    x = DynamicNoiseLayer(name='scheduled_gpu_noise')(inp)

    # Feature Extraction with CNN 2 Layers
    cnn_out = layers.Conv1D(CNN_FILTERS, CNN_KERNEL, padding='same', activation='relu', kernel_initializer='he_uniform', name='conv1')(x)
    x = layers.Conv1D(CNN_FILTERS*2, CNN_KERNEL, padding='same', activation='relu', kernel_initializer='he_uniform', name='conv2')(cnn_out)
    x = layers.MaxPooling1D(CNN_POOL, name='pool1')(x)

    # Bidirectional RNN for Temporal Pattern Recognition
    x = layers.Bidirectional(rnn_class(UNITS_L1, return_sequences=True, name=f'{name}_l1', **rnn_kw))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(DROPOUT)(x)

    x = layers.Bidirectional(rnn_class(UNITS_L2, return_sequences=True, name=f'{name}_l2', **rnn_kw))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(DROPOUT)(x)

    # MultiHeadAttention Degraded Performance
    #x = layers.MultiHeadAttention(num_heads=4, key_dim=32, dtype='float32')(x, x)
    #x = layers.LayerNormalization()(x)

    # Dimensionality Restoration
    x = layers.UpSampling1D(CNN_POOL, name='upsample')(x)
    up_len = x.shape[1]
    if up_len > time_steps: x = layers.Cropping1D((0, up_len - time_steps), name='crop')(x)
    elif up_len < time_steps: x = layers.ZeroPadding1D((0, time_steps - up_len), name='pad')(x)

    # Residual/Skip Connection to Preserve Unfiltered CNN Features
    skip = layers.Conv1D(UNITS_L2 * 2, 1, padding='same', name='skip_proj')(cnn_out)
    x    = layers.Add(name='residual')([x, skip])
    x   = layers.TimeDistributed(layers.Dense(32, activation='relu'))(x)
    # Softplus Over ReLU: Smooth at Zero, Never Exactly Zero, Unbounded Above
    out = layers.TimeDistributed(layers.Dense(1, activation='softplus', dtype='float32'), name='output')(x)

    model = models.Model(inp, out, name=f'{name}_model')
    
    # XLA Only for RNN - LSTM and GRU Have Dedicated cuDNN Kernels That Outperform XLA
    use_xla = (name == 'RNN')
    model.compile(optimizer=Adam(learning_rate=lr_schedule, clipnorm=CLIP_NORM),
                  loss=weighted_huber, metrics=['mae'],
                  steps_per_execution=STEPS_PER_EXEC, jit_compile=use_xla)
    return model

def create_rnn(lr):  return _build_model(layers.SimpleRNN, 'RNN',  lr)
def create_lstm(lr): return _build_model(layers.LSTM,      'LSTM', lr)
def create_gru(lr):  return _build_model(layers.GRU,       'GRU',  lr)

# Training Models
print("\n" + "=" * 42)
print("4. Training models")
print("=" * 42)

factories = {'RNN': create_rnn, 'LSTM': create_lstm, 'GRU': create_gru}
COLORS = {'RNN': 'dodgerblue', 'LSTM': 'darkorange', 'GRU': 'forestgreen', 'Ensemble': 'mediumpurple'}
histories   = {}
predictions_test = {}
predictions_train = {}

if RUN_LR_FINDER:
    print("\nRUNNING LR FINDER (RNN Model)")
    K.clear_session()
    ds_train_lr = make_dataset(X_train, y_train, training=True)
    model_lr = create_rnn(1e-6) 
    lr_finder = LRFinder(steps=steps_per_epoch * 2)
    model_lr.fit(ds_train_lr, epochs=2, callbacks=[lr_finder], verbose=0)
    
    plt.figure(figsize=(8, 4))
    plt.plot(lr_finder.lrs, lr_finder.losses)
    plt.xscale('log'); plt.xlabel('Learning Rate'); plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.tight_layout()
    plt.savefig('figures/LR_Finder_Curve.png')
    plt.show()
    print("LR Finder plot saved. Exit ")
    exit()

for name, factory in factories.items():
    print(f"\n{'─' * 60}\n  Model : {name}\n{'─' * 60}")
    
    # Clear TF Graph to Prevent Memory Leaks Between Models
    K.clear_session()
    
    # Instantiate Datasets Inside Loop for Safe Session Clearing
    ds_train      = make_dataset(X_train, y_train, training=True)
    ds_val        = make_dataset(X_val,   y_val)
    ds_test       = make_dataset(X_test,  y_test, batch_size=EVAL_BATCH)
    ds_train_eval = make_dataset(X_train, y_train, training=False, batch_size=EVAL_BATCH)
    
    lr_schedule = make_lr_schedule()
    model = factory(lr_schedule)

    cbs = [
        EarlyStoppingMinEpochs(start_epoch=MIN_EPOCHS, monitor='val_loss', patience=PATIENCE_ES,
                               min_delta=MIN_DELTA_ES, restore_best_weights=True, verbose=1),
        NoiseDecayCallback()
    ]

    t0   = time.time()
    hist = model.fit(ds_train, epochs=EPOCHS, validation_data=ds_val, verbose=2, callbacks=cbs)
    elapsed = (time.time() - t0) / 60
    print(f"\n  {name}  train={hist.history['loss'][-1]:.5f}  val={hist.history['val_loss'][-1]:.5f}  time={elapsed:.2f} min")

    histories[name]           = hist
    predictions_test[name]    = model.predict(ds_test,       verbose=0)
    predictions_train[name]   = model.predict(ds_train_eval, verbose=0)
    del model

# Visualization of Training Progress
fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=False)
for name in factories:
    h = histories[name].history
    axes[0].plot(h['loss'],     color=COLORS[name], lw=1.5, label=name)
    axes[1].plot(h['val_loss'], color=COLORS[name], lw=1.5, label=name)
axes[0].set_title('Training Loss (all models)');   axes[0].set_xlabel('Epoch')
axes[1].set_title('Validation Loss (all models)'); axes[1].set_xlabel('Epoch')
for ax in axes:
    ax.set_ylabel('Weighted Huber Loss'); ax.legend()
plt.suptitle('Loss Curves – Combined Comparison', fontsize=13)
plt.tight_layout()
plt.savefig('figures/Loss_Curves.png')
plt.show()
plt.close()

# Performance Benchmarking
def vectorised_errors(y_true_3d, y_pred_3d, scale):
    yt  = y_true_3d.squeeze() * scale
    yp  = y_pred_3d.squeeze() * scale
    err = yt - yp
    return (np.sqrt(np.mean(err**2, axis=1)), np.mean(np.abs(err), axis=1), np.max(np.abs(err), axis=1))

all_errors_test  = {}
all_errors_train = {}

for name in factories:
    rmse_te, mae_te, maxerr_te = vectorised_errors(y_test, predictions_test[name], max_app)
    all_errors_test[name]  = {'RMSE': rmse_te, 'MAE': mae_te, 'MaxErr': maxerr_te}

    rmse_tr, mae_tr, maxerr_tr = vectorised_errors(y_train, predictions_train[name], max_app)
    all_errors_train[name] = {'RMSE': rmse_tr, 'MAE': mae_tr, 'MaxErr': maxerr_tr}

# Zero Prediction if Aggregate Below 5% of Max - Appliance Cannot Be On
AGG_GATE = 0.05
agg_test_ch  = X_test[:, :, 0:1]
agg_train_ch = X_train[:, :, 0:1]

# Suppress Predictions in First 3 Timesteps Unless Startup Ramp Detected
ONSET_GUARD  = 3
# Minimum Derivative to Count as an Appliance Turn-On Event
DX_ONSET_MIN = 0.05
agg_dx_test  = X_test[:, :, 1:2]  
agg_dx_train = X_train[:, :, 1:2]

def apply_postproc_gates(pred, agg_ch, dx_ch):
    pred = np.where(agg_ch < AGG_GATE, 0.0, pred)
    pred[:, :ONSET_GUARD, :] = np.where(
        dx_ch[:, :ONSET_GUARD, :] < DX_ONSET_MIN, 0.0, pred[:, :ONSET_GUARD, :])
    return pred

for name in factories:
    predictions_test[name]  = apply_postproc_gates(predictions_test[name],  agg_test_ch,  agg_dx_test)
    predictions_train[name] = apply_postproc_gates(predictions_train[name], agg_train_ch, agg_dx_train)

# Ensemble Weights Based on Validation Loss
best_val_losses = {name: min(histories[name].history['val_loss']) for name in factories}
inv_losses = {name: 1.0 / loss for name, loss in best_val_losses.items()}
sum_inv = sum(inv_losses.values())
weights = {name: inv / sum_inv for name, inv in inv_losses.items()}

# Ensemble Test
pred_ensemble_te = sum(predictions_test[n] * weights[n] for n in factories)
pred_ensemble_te = apply_postproc_gates(pred_ensemble_te, agg_test_ch, agg_dx_test)
ens_rmse_te, ens_mae_te, ens_maxerr_te = vectorised_errors(y_test, pred_ensemble_te, max_app)
all_errors_test['Ensemble'] = {'RMSE': ens_rmse_te, 'MAE': ens_mae_te, 'MaxErr': ens_maxerr_te}
predictions_test['Ensemble'] = pred_ensemble_te

# Ensemble Train
pred_ensemble_tr = sum(predictions_train[n] * weights[n] for n in factories)
pred_ensemble_tr = apply_postproc_gates(pred_ensemble_tr, agg_train_ch, agg_dx_train)
ens_rmse_tr, ens_mae_tr, ens_maxerr_tr = vectorised_errors(y_train, pred_ensemble_tr, max_app)
all_errors_train['Ensemble'] = {'RMSE': ens_rmse_tr, 'MAE': ens_mae_tr, 'MaxErr': ens_maxerr_tr}

all_model_names = list(factories.keys()) + ['Ensemble']

# Bar Charts
x = np.arange(len(all_model_names))
bar_specs = [
    ('MAE',    'Mean Absolute Error (MAE) - Test',     '#4c72b0'),
    ('RMSE',   'Root Mean Square Error (RMSE) - Test', '#dd8452'),
    ('MaxErr', 'Maximum Error (MaxErr) - Test',        '#55a868'),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (metric, title, color) in zip(axes, bar_specs):
    means = [all_errors_test[m][metric].mean() for m in all_model_names]
    stds  = [all_errors_test[m][metric].std()  for m in all_model_names]
    # Prevents Negative Values
    yerr  = [np.minimum(means, stds), stds]
    ax.bar(x, means, width=0.5, yerr=yerr, capsize=4, color=color)
    ax.set_title(title)
    ax.set_xticks(x); ax.set_xticklabels(all_model_names, rotation=15)
    ax.set_ylim(bottom=0)
axes[0].set_ylabel('Watts')

plt.tight_layout()
plt.savefig('figures/Performance_BarCharts_Separated.png')
plt.show()
plt.close()

# Per Sequence Error Plots
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
n_seq = len(y_test)
seq_x = np.arange(1, n_seq + 1)

for name in all_model_names:
    axes[0].plot(seq_x, all_errors_test[name]['MAE'], label=name, alpha=0.7, lw=1.2)
    axes[1].plot(seq_x, all_errors_test[name]['RMSE'], label=name, alpha=0.7, lw=1.2)
    axes[2].plot(seq_x, all_errors_test[name]['MaxErr'], label=name, alpha=0.7, lw=1.2)

axes[0].set_title('Per-Sequence MAE (Test Set)')
axes[0].set_ylabel('Watts')
axes[0].legend(loc='upper right')

axes[1].set_title('Per-Sequence RMSE (Test Set)')
axes[1].set_ylabel('Watts')
axes[1].legend(loc='upper right')

axes[2].set_title('Per-Sequence MaxErr (Test Set)')
axes[2].set_xlabel('Test Sequence Index')
axes[2].set_ylabel('Watts')
axes[2].legend(loc='upper right')

plt.tight_layout()
plt.savefig('figures/Per_Sequence_Errors.png')
plt.show()
plt.close()

# Visual Ground Truth vs Prediction Comparison
y_test_watt   = y_test.squeeze() * max_app
active_test   = np.where(y_test_watt.max(axis=1) > 0)[0]
inactive_test = np.where(y_test_watt.max(axis=1) == 0)[0]

n_on  = min(2, len(active_test))
n_off = min(2, len(inactive_test))
sel_idx = (list(rng.choice(active_test,   n_on,  replace=False)) +
           list(rng.choice(inactive_test, n_off, replace=False)))
labels  = ['ON'] * n_on + ['OFF'] * n_off

pred_ens_watt = pred_ensemble_te.squeeze() * max_app

for k, (si, status) in enumerate(zip(sel_idx, labels), 1):
    plt.figure(figsize=(11, 4))
    plt.plot(MINUTES, y_test_watt[si], color='black', lw=2.5, label='Ground Truth', zorder=6)

    for name in factories:
        pred_w = predictions_test[name][si].squeeze() * max_app
        plt.plot(MINUTES, pred_w, color=COLORS[name], lw=1.2, ls='--', label=name, alpha=0.8)

    plt.plot(MINUTES, pred_ens_watt[si], color=COLORS['Ensemble'], lw=2.0, ls='-', label='Ensemble', alpha=0.95, zorder=5)

    plt.title(f'Test Sequence {si} | Coffee Machine: {status}  (Plot {k}/{n_on+n_off})')
    plt.xlabel('Time (minutes)'); plt.ylabel('Coffee Machine (W)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'figures/Test_Sequence_{si}.png')
    plt.show()
    plt.close()

# Results
rows = [{'Model': n,
         'Test RMSE (W)':  all_errors_test[n]['RMSE'].mean(),
         'Test MAE (W)':   all_errors_test[n]['MAE'].mean(),
         'Test MaxErr (W)':all_errors_test[n]['MaxErr'].mean(),
         'Train RMSE (W)': all_errors_train[n]['RMSE'].mean(),
         'Train MAE (W)':  all_errors_train[n]['MAE'].mean(),
         'Train MaxErr (W)':all_errors_train[n]['MaxErr'].mean()}
        for n in all_model_names]

df = pd.DataFrame(rows).set_index('Model')
df.to_csv('summary_results.csv', float_format='%.3f')
print("\nSaved: summary_results.csv and all plots in 'figures' folder.")
print(f"Total runtime: {(time.time()-start_time)/60:.2f} minutes")