#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
把一个多帧 tiff 按“相邻两帧一组”拆分：
(0,1), (1,2), (2,3), ... (T-2, T-1)

假设时间轴在第 0 维，shape = (T, ...)

如果时间轴不是第 0 维，可以参考下面注释，用 np.moveaxis 先换轴。
"""

import os
import sys
import tifffile as tiff
import numpy as np


def main(input_path, output_dir):
    # 读取整段 tiff
    print(f"读取输入文件: {input_path}")
    data = tiff.imread(input_path)  # shape: (T, ...)

    print(f"数据维度: {data.shape} (时间轴假定为第 0 维)")
    if data.ndim < 3:
        print(f"警告：读取到的数组维度是 {data.ndim}，看起来不像是多帧时间序列（至少应该是 3D）。")

    # ===== 如果时间轴不是第 0 维，在这里调整 =====
    # 比如 shape = (Z, T, Y, X)，时间在第 1 维，可以这样写：
    # data = np.moveaxis(data, 1, 0)
    # ============================================

    n_frames = data.shape[0]
    if n_frames < 2:
        print("帧数少于 2，无法按相邻帧成对拆分。")
        return

    print(f"总帧数: {n_frames}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 去掉扩展名，作为输出文件前缀
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    pair_index = 0
    # 相邻两帧一组：0-1, 1-2, ..., (n_frames-2)-(n_frames-1)
    for start in range(0, n_frames - 1):
        end = start + 2  # 不包含 end，下标 start 和 start+1

        chunk = data[start:end]  # shape: (2, ...)

        # 文件名里标一下是哪两帧，更直观
        out_name = f"{base_name}_pair{start:03d}_{start+1:03d}.tif"
        out_path = os.path.join(output_dir, out_name)

        print(f"写出: {out_path}  (包含帧 {start} 和 {start+1})")
        tiff.imwrite(out_path, chunk)

        pair_index += 1

    print("完成。总共生成相邻帧对数:", pair_index)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python split_tifframe.py input.tif output_dir")
        sys.exit(1)

    input_tif = sys.argv[1]
    output_dir = sys.argv[2]
    main(input_tif, output_dir)
