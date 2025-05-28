import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# --- NEW ---
import os, argparse, logging
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, UpSampling2D

df = pd.read_parquet("hf://datasets/dpanangian/roof-segmentation-control-net/data/train-00000-of-00001.parquet")

# ---------------------------------------------------------------------------
# CLI – choose hardware optimisations
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Solar roof trainer")
parser.add_argument("--gpu", default="auto",
                    help="gpu type: L4, A100, cpu, auto (default=auto)")
parser.add_argument("--epochs", type=int, default=30,
                    help="number of training epochs (default=30)")
args, _ = parser.parse_known_args()  # keeps notebook argv intact

gpu_type   = args.gpu.lower()
MIXED_PREC = False  # will toggle later

if gpu_type == "cpu":
    # Force CPU by masking CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
elif gpu_type in ("l4", "a100", "ada", "ampere"):
    # Nvidia cards with Tensor Cores – enable fp16 + XLA
    tf.config.optimizer.set_jit(True)                      # XLA JIT
    mixed_precision.set_global_policy("mixed_float16")     # fp16 compute
    MIXED_PREC = True
else:
    # 'auto' or unknown → do nothing special
    pass

# ---------------------------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------------------------
# The parquet stores each picture as an Arrow‑struct like
#   {"bytes": b"...raw JPG/PNG bytes..."}
# We'll unwrap those dicts so TensorFlow receives plain bytes.
def extract_bytes(cell):
    # Handle dicts coming from Arrow and already‑raw bytes alike
    return cell["bytes"] if isinstance(cell, dict) and "bytes" in cell else cell

image_bytes = df["image"].apply(extract_bytes).tolist()
mask_bytes  = df["conditioning_image"].apply(extract_bytes).tolist()

IMG_SIZE = (256, 256)          # spatial resolution for the network
BATCH    = 16                  # tweak to fit your GPU / CPU memory

def load_image_pair(img_bytes, mask_bytes):

    # RGB rooftop ------------------------------------------------------------
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    # Mask --------------------------------------------
    mask = tf.io.decode_image(mask_bytes, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, IMG_SIZE)
    mask = tf.cast(mask, tf.float32) / 255.0

    return img, mask

# Train / validation split ---------------------------------------------------
train_img, val_img, train_mask, val_mask = train_test_split(
    image_bytes, mask_bytes, test_size=0.15, random_state=42
)

train_ds = (tf.data.Dataset.from_tensor_slices((train_img, train_mask))
            .map(load_image_pair, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH)
            .prefetch(tf.data.AUTOTUNE))

val_ds   = (tf.data.Dataset.from_tensor_slices((val_img, val_mask))
            .map(load_image_pair, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH)
            .prefetch(tf.data.AUTOTUNE))

# ---------------------------------------------------------------------------
# MODEL DEFINITION  – a lightweight U‑Net
# ---------------------------------------------------------------------------
def unet_model(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)

    # Encoder
    c1 = Conv2D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal")(inputs)
    c1 = Conv2D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal")(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(p1)
    c2 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(p2)
    c3 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(c3)
    p3 = MaxPooling2D()(c3)

    # Bottleneck
    b  = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(p3)
    b  = Conv2D(256, 3, activation="relu", padding="same", kernel_initializer="he_normal")(b)

    # Decoder
    u3 = UpSampling2D()(b)
    u3 = concatenate([u3, c3])
    c4 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(u3)
    c4 = Conv2D(128, 3, activation="relu", padding="same", kernel_initializer="he_normal")(c4)

    u2 = UpSampling2D()(c4)
    u2 = concatenate([u2, c2])
    c5 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(u2)
    c5 = Conv2D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal")(c5)

    u1 = UpSampling2D()(c5)
    u1 = concatenate([u1, c1])
    c6 = Conv2D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal")(u1)
    c6 = Conv2D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal")(c6)

    outputs = Conv2D(
        1, 1, activation=None, dtype="float32" if MIXED_PREC else None,
        name="mask")(c6)

    return tf.keras.Model(inputs, outputs, name="solar_roof_unet")

model = unet_model(input_shape=IMG_SIZE + (3,))
model.compile(optimizer="adam",
              loss="mse",
              metrics=["mse"])
model.summary()
# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
# Early‑stopping: stop if val_cosine_loss stalls for 3 epochs (min delta 1e‑4)
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    min_delta=1e-4,
    restore_best_weights=True,
    verbose=1)
EPOCHS = args.epochs
model.fit(train_ds,
          validation_data=val_ds,
          epochs=EPOCHS,
          callbacks=[early_stop])

# ---------------------------------------------------------------------------
# SAVE THE TRAINED NETWORK
# ---------------------------------------------------------------------------
model.save("model-output/solar_roof_unet.keras")
print("Model saved to solar_roof_unet.keras")