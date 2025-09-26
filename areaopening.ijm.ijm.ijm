outDir = getDirectory("Choose an output folder"); 
orig = getTitle();
Stack.getDimensions(w, h, c, z, frames); // XYCZT
nd = 2; // 帧号零填充位数，例如 001..030，可按需改
setBatchMode(true);

for (t = 1; t <= frames; t++) {
    // 回到原始堆栈的第 t 帧
    selectWindow(orig);
    setSlice(t); // C=1, Z=1 时，setSlice(t) 即第 t 帧

    // 运行 Area Opening（注意：参数名用你 Recorder 录到的为准：有的版本是 "min"，有的是 "pixel"）
    // 常见录制形式：run("Area Opening", "min=327 connectivity=8");
    run("Area Opening", "pixel=327");

    // 结果窗口现在是激活的 —— 重命名、保存并关闭
    fname = "frame_" + IJ.pad(t, nd);
    run("Rename...", "title="+fname);
    saveAs("Tiff", outDir + fname + ".tif");
    close();
}
setBatchMode(false);
selectWindow(orig);
print("Done. Saved to: " + outDir, "Min area (px^2) = 327 pixel" );