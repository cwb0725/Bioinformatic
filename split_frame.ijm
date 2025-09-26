// Save each frame/slice of current stack as im_skel001.tif, im_skel002.tif, ...
dir = getDirectory("Choose a Directory to Save Frames");

// 读取维度：width, height, channels, slices(Z), frames(T)
getDimensions(w, h, c, z, t);

digits = 3; // 位数：改成2或4都行

if (t > 1) {
    // 按时间帧导出
    for (i = 1; i <= t; i++) {
        Stack.setFrame(i);         // 选中第 i 帧 (T)
        run("Duplicate...", "title=tmp");  // 复制当前帧为单图
        saveAs("Tiff", dir + "roi_seq" + IJ.pad(i-1, digits) + ".tif");
        close();                   // 关闭临时图
    }
} else {
    // 没有时间维度，按 Z-slices 导出
    for (i = 1; i <= z; i++) {
        setSlice(i);               // 选中第 i 层 (Z)
        run("Duplicate...", "title=tmp");
        saveAs("Tiff", dir + "roi_seq_ori  " + IJ.pad(i-1, digits) + ".tif");
        close();
    }
}
