import math, numpy as np, tifffile as tiff
from skimage.morphology import remove_small_objects

labels_path = r"E:\swin_unet\exp_123745-SIM561_RedCh_SIrecon_pair006_007.ome-TYX-T10p0_Y0p0306_X0p0306-ch0-t0_to_1-im_obj_label_reassigned.ome.tif"
save_path   = r"E:\swin_unet\exp_123745_006_007_denoised.ome.tif"

px_um = 0.030589          # 你的像素尺寸
min_area_um2 = 0.12       # 面积阈值(µm²)
min_px = math.ceil(min_area_um2 / (px_um * px_um))

lab = tiff.imread(labels_path)         # (T, Y, X) int32
assert lab.ndim == 3
T = lab.shape[0]

lab_out = lab.copy()
total_removed = 0

for t in range(T):
    plane = lab_out[t]
    pos = plane > 0                              # —— 布尔掩膜（不看具体label编号！）
    before = int(pos.sum())
    # 8-连通：connectivity=2（2D 时 8-连通；若想 4-连通就用 1）
    pos_clean = remove_small_objects(pos, min_size=min_px, connectivity=2)
    # 把被去掉的位置置 0（保留原 label 值的地方不变）
    plane[~pos_clean] = 0
    removed = before - int(pos_clean.sum())
    total_removed += removed
    print(f"t={t}: removed_pixels={removed}")

tiff.imwrite(save_path, lab_out.astype(lab.dtype))
print("Saved:", save_path, "| total_removed:", total_removed)
mask = lab_out > 0
appear = mask.sum(axis=0)                     # 每个(Y,X)出现了多少帧
short_lived_max_frames = 2
short_xy = appear <= short_lived_max_frames   # 这些位置算“短命”
removed_short = int(mask[:, short_xy].sum())
lab_out[:, short_xy] = 0
print("temporal removed:", removed_short)
