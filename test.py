import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import re
import cv2
import tifffile as tiff


# =========================
# 参数区（按你路径改）
# =========================
MODEL_PATH = "../model/U-Net_patch512.keras"     # 512 patch 训练出来的模型
TEST_DIR   = "../test"                          # 放 3072 tif 的目录
OUT_DIR    = "../pred_tif"                      # 输出目录（tif）
MEAN_PATH  = "../npydata/train_mean.npy"        # 可选：训练集 mean (3072,3072,1)

# 推理参数
WIN = 512
OVERLAP = 128
STRIDE = WIN - OVERLAP
THRESHOLD = 0.5   # 二值化阈值（如果全黑，先看 prob 最大值再调）

# 可选：禁用 GPU（需要就取消注释）
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def u16_to_u8_cv_minmax(img_u16: np.ndarray) -> np.ndarray:
    x = np.asarray(img_u16)
    if x.dtype != np.uint16:
        raise TypeError("expect uint16")
    y = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
    return y.astype(np.uint8)





def type_8bit(img, vmin=None, vmax=None, eps=1e-6,
              mode="percentile", p_low=0.1, p_high=99.9,
              ignore_zero=True):
    """
    mode:
      - "percentile": 用分位数窗口（推荐，最像 Fiji/Enhance Contrast 的效果）
      - "fixed": 用传入的 vmin/vmax（最像你在 Fiji 手动调 B/C 后再转换）
    """
    x = np.asarray(img)
    x = np.squeeze(x).astype(np.float32)

    if ignore_zero:
        xs = x[x > 0]
    else:
        xs = x.ravel()

    if xs.size == 0:
        return np.zeros_like(x, dtype=np.uint8)

    if mode == "fixed":
        if vmin is None or vmax is None:
            raise ValueError("mode='fixed' 需要提供 vmin/vmax")
        lo, hi = float(vmin), float(vmax)
    elif mode == "minmax":
        lo, hi = float(xs.min()), float(xs.max())
    else:  # "percentile"
        lo = float(np.percentile(xs, p_low))
        hi = float(np.percentile(xs, p_high))

    if hi - lo < eps:
        return np.zeros_like(x, dtype=np.uint8)

    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0 + 0.5).astype(np.uint8)


def list_tifs(folder):
    exts = (".tif", ".tiff", ".TIF", ".TIFF")
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith(exts)]


def load_image_tif(path):
    img = tiff.imread(path)
    if img.ndim > 2:
        img = img[..., 0]
    img = np.asarray(img)

    if img.dtype == np.uint16:
        img8 = u16_to_u8_cv_minmax(img)   # <= 新增这一行
        img_f = img8.astype(np.float32) / 255.0
        return img_f, 255.0


    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0, 255.0

    return img.astype(np.float32), 1.0



def get_weight_window(win):
    """
    overlap 拼接用的加权窗：2D Hann 窗，减少接缝
    """
    w1 = np.hanning(win).astype(np.float32)
    w2 = np.outer(w1, w1)
    w2 = np.maximum(w2, 1e-6)  # 防止全0
    return w2


def preprocess(img01, mean_img=None):
    """
    img01: float32, range 大致 0~1
    mean_img: (H,W) 或 (H,W,1)
    返回 (H,W) float32
    """
    x = img01.astype(np.float32)
    if mean_img is not None:
        m = mean_img
        if m.ndim == 3:
            m = m[..., 0]
        x = x - m.astype(np.float32)
    else:
        # 没有逐像素 mean，只能退化为标量去均值
        x = x - float(x.mean())
    return x


def sliding_predict(model, img, win=512, stride=384):
    """
    img: (H,W) float32
    返回 prob: (H,W) float32 in [0,1]
    """
    H, W = img.shape
    weight = get_weight_window(win)

    prob_sum = np.zeros((H, W), dtype=np.float32)
    w_sum = np.zeros((H, W), dtype=np.float32)

    # 遍历窗口，确保覆盖到边界
    ys = list(range(0, H - win + 1, stride))
    xs = list(range(0, W - win + 1, stride))
    if ys[-1] != H - win:
        ys.append(H - win)
    if xs[-1] != W - win:
        xs.append(W - win)

    for y in ys:
        for x in xs:
            patch = img[y:y+win, x:x+win]
            patch = patch[np.newaxis, ..., np.newaxis]  # (1,win,win,1)

            pred = model.predict(patch, verbose=0)[0, ..., 0]  # (win,win)

            prob_sum[y:y+win, x:x+win] += pred * weight
            w_sum[y:y+win, x:x+win] += weight

    prob = prob_sum / np.maximum(w_sum, 1e-6)
    prob = np.clip(prob, 0.0, 1.0)
    return prob


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 加载 mean（如果存在）
    mean_img = None
    if os.path.exists(MEAN_PATH):
        mean_img = np.load(MEAN_PATH)
        # 兼容 (H,W,1) / (N,H,W,1) 这种情况
        if mean_img.ndim == 4:
            mean_img = mean_img[0]
        print("Loaded mean:", MEAN_PATH, "shape=", mean_img.shape)
    else:
        print("Mean file not found, fallback to per-image mean subtraction:", MEAN_PATH)

    # 加载模型
    print("Loading model:", MODEL_PATH)
    model = load_model(MODEL_PATH, compile=False)
    print("Model loaded.")

    files = list_tifs(TEST_DIR)
    print("Found tif:", len(files))
    if not files:
        raise RuntimeError(f"No tif found in {TEST_DIR}")

    for i, f in enumerate(files, 1):
        name = os.path.splitext(os.path.basename(f))[0]
        print(f"[{i}/{len(files)}] Predicting:", name)

        img01, _scale = load_image_tif(f)

        # 这里不强制 3072，任意尺寸都能滑窗
        H, W = img01.shape
        if H < WIN or W < WIN:
            raise RuntimeError(f"Image too small for window={WIN}: {f} shape={img01.shape}")

        x = preprocess(img01, mean_img=mean_img)

        prob = sliding_predict(model, x, win=WIN, stride=STRIDE)

        # 输出调试信息（帮你判断为什么黑）
        print("  prob stats:",
              "min", float(prob.min()),
              "max", float(prob.max()),
              "mean", float(prob.mean()),
              ">=thr ratio", float((prob >= THRESHOLD).mean()))

        # 保存概率图（float32 tif）
        prob_path = os.path.join(OUT_DIR, f"{name}_prob.tif")
        tiff.imwrite(prob_path, prob.astype(np.float32), photometric="minisblack")

        # 保存二值 mask（uint8 tif, 0/255）
        mask = (prob >= THRESHOLD).astype(np.uint8) * 255
        mask_path = os.path.join(OUT_DIR, f"{name}_mask.tif")
        tiff.imwrite(mask_path, mask, photometric="minisblack")

    print("Done. Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()

