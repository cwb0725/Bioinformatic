# convert_minmax_cv2.py
import os
import numpy as np
import cv2
import tifffile as tiff

IN_DIR  = "/path/to/input_tifs"
OUT_DIR = "/path/to/output_8bit"

os.makedirs(OUT_DIR, exist_ok=True)

exts = (".tif", ".tiff", ".TIF", ".TIFF")
files = [f for f in sorted(os.listdir(IN_DIR)) if f.endswith(exts)]

print("Found tif:", len(files))

for i, fn in enumerate(files, 1):
    src = os.path.join(IN_DIR, fn)
    img16 = tiff.imread(src)

    # 只取第一通道（如果有多通道/多维）
    if img16.ndim > 2:
        img16 = img16[..., 0]
    img16 = np.asarray(img16)

    # ✅ 你的方法：每张图按 min/max 拉伸到 0~255
    img8 = cv2.normalize(img16, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    out_name = os.path.splitext(fn)[0] + "_8bit.tif"
    dst = os.path.join(OUT_DIR, out_name)
    tiff.imwrite(dst, img8)

    print(f"[{i}/{len(files)}] {fn} -> {out_name}")

print("Done.")

