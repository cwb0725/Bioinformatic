from tifffile import imread, imwrite
import numpy as np
import pandas as pd

in_path   = r"C:\path\labels.ome.tif"              # 原始标签图
out_path  = r"C:\path\labels_denoised.tif"         # 输出去噪标签图
pairs_txt = r"C:\path\drop_pairs_t_reassigned.txt" # 上一步R导出的 (t,reassigned_label) 文本

# 1) 读标签栈；若时间轴不是第0维，先搬轴到第0维
lab = imread(in_path)
# 例如若你的数据是 (Y,X,T)，请取消下一行注释：
# lab = np.moveaxis(lab, -1, 0)

assert lab.ndim == 3, f"期望 (T,Y,X)，当前形状：{lab.shape}"
lab = lab.astype(np.int64, copy=False)

# 2) 读 (t,reassigned_label) 列表
pairs = pd.read_csv(pairs_txt, header=None, names=["t","reassigned_label"])
pairs["t"] = pairs["t"].astype(int)
pairs["reassigned_label"] = pairs["reassigned_label"].astype(int)

lab_out = lab.copy()

# 3) 逐帧清零（每帧的 reassigned_label 集在各自帧内匹配）
for t, sub in pairs.groupby("t"):
    if t < 0 or t >= lab_out.shape[0]:
        continue
    frame = lab_out[t]
    drops = sub["reassigned_label"].to_numpy(dtype=np.int64)

    # 用查找表(LUT)高效清零
    max_id = int(frame.max())
    lut = np.arange(max_id + 1, dtype=frame.dtype)
    drops = drops[(drops > 0) & (drops <= max_id)]
    lut[drops] = 0
    lab_out[t] = lut[frame]

# 4) 保存（自动选16/32位）
dtype = np.uint16 if lab_out.max() <= 65535 else np.uint32
imwrite(out_path, lab_out.astype(dtype), compression='zlib')
print("saved:", out_path)
