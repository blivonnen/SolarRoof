import pandas as pd, numpy as np, io, base64, random
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

# 1 — load the trained network ------------------------------------------------
model = tf.keras.models.load_model("model-output/solar_roof_unet.h5", compile=False)

# 2 — read the parquet again --------------------------------------------------
df = pd.read_parquet("hf://datasets/dpanangian/roof-segmentation-control-net/data/train-00000-of-00001.parquet")

def extract_bytes(cell):
    return cell["bytes"] if isinstance(cell, dict) and "bytes" in cell else cell

def decode(cell, rgb=True):
    raw = extract_bytes(cell)
    img = Image.open(io.BytesIO(raw))
    return img.convert("RGB" if rgb else "L")

# 3 — utility: model expects (256, 256) tensors in [0,1] ----------------------
IMG_SIZE = (256, 256)

def preprocess(pil_img, rgb=True):
    arr = np.array(pil_img.resize(IMG_SIZE))      # RGB → (H,W,3);  grey → (H,W)
    if not rgb:                                   # expand greyscale to (H,W,1)
        arr = arr[..., np.newaxis]
    return arr.astype("float32") / 255.0

# 4 — pick n rows, run inference, plot ---------------------------------------
n = 3                                # change as you like
rows = random.sample(range(len(df)), n)

fig, axes = plt.subplots(n, 3, figsize=(9, 3*n))
titles = ["Original", "Conditioning", "Predicted"]

for i, idx in enumerate(rows):
    rgb  = decode(df.at[idx, "image"],               rgb=True)
    grey = decode(df.at[idx, "conditioning_image"],  rgb=False)

    mask_arr = np.array(grey)
    print(f"Sample {idx} conditioning_image raw array:")
    print(mask_arr)

    # Predict
    inp  = preprocess(rgb, rgb=True)[np.newaxis, ...]        # (1,256,256,3)
    pred_vec = model.predict(inp, verbose=0)[0]                # (H,W,1) in [0,1]
    grey_pred = (pred_vec[..., 0] * 255).astype("uint8")
    pred_img = Image.fromarray(grey_pred)
    print(f"Sample {idx} prediction raw array:")
    print(grey_pred)

    #— show
    for j, img in enumerate([rgb, grey, pred_img]):
        ax = axes[i, j]
        ax.imshow(img if j==0 else img, cmap=None if j==0 else "gray")
        ax.set_title(titles[j] if i==0 else "", fontsize=10)
        ax.axis("off")

plt.tight_layout()
plt.show()