import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import sys
import base64
import io
import numpy as np

df = pd.read_parquet("roof-segmentation-control-net/data/train-00000-of-00001.parquet")
sample_rows = df.head(2)              # grab first three examples


def decode_image(cell, want_rgb=True):
    """
    Accepts:
      • dict   – with a 'bytes' key (Arrow-struct column),
      • bytes  – raw PNG/JPEG,
      • str    – base‑64 string.
    Returns a Pillow Image.
    """
    if isinstance(cell, dict) and "bytes" in cell:
        cell = cell["bytes"]                      # unwrap from Arrow struct
    if isinstance(cell, str):
        cell = base64.b64decode(cell)             # base64 → bytes
    # at this point we expect raw bytes
    if not isinstance(cell, (bytes, bytearray)):
        raise TypeError("Unsupported cell type for decode_image")
    img = Image.open(io.BytesIO(cell))
    return img.convert("RGB" if want_rgb else "L")


# ---- Plot original image and masks and for the first three examples in one figure ----
fig, axs = plt.subplots(len(sample_rows), 2, figsize=(8, 4 * len(sample_rows)))
for ax_row, (idx, row) in zip(axs, sample_rows.iterrows()):
    # Decode greyscale mask
    rgb = decode_image(row["image"], want_rgb=True)
    grey = decode_image(row["conditioning_image"], want_rgb=False)

    ax_original, ax_mask = ax_row
    # Original image display
    ax_original.imshow(rgb)
    ax_original.axis("off")
    ax_original.set_title(f"Row {idx} original")

    # Mask display
    ax_mask.imshow(grey, cmap="gray")
    ax_mask.axis("off")
    ax_mask.set_title(f"Row {idx} mask")


plt.tight_layout()
plt.show()