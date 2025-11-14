# ===== 固定阈值 + 短命小点 + 全局小点（中位面积）去噪 =====
# 需要列：t, reassigned_label_raw, organelle_area_raw (单位 µm²)
exp_221155 <- read.csv("C:/Users/11622/Desktop/MT_tiff/cos7MFN1TMRE/exp_221155-SIM561_RedCh_SIrecon-TYX-T10p0_Y0p0306_X0p0306-ch0-t0_to_90-features_organelles.csv")
# ===== 固定阈值 + 短命小点(中位面积) + 全局逐行筛选 =====
# 需要列：t, reassigned_label_raw, organelle_area_raw (单位 µm²)

# ===== 固定阈值 + 短命小点(中位面积) + 全局逐行筛选 + 导出所有被删行 =====
# 需要列：t, reassigned_label_raw, organelle_area_raw (µm²)
# 若原表含 'label' 列会直接使用；若没有则用 NA 占位（可按需改成复制 reassigned_label_raw）

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})
clean_exp <- function(expname,outname,csvdir){
# === 参数（按需修改） ===
out_csv  <- csvdir

min_area_t0 <- 0.12                # 步骤1：t==0 小面积
short_lived_max_frames <- 2        # 步骤2：短命（出现帧数 ≤ N）
min_area_short <- 0.18             # 步骤2：短命阈值（中位面积）
min_area_rowwise <- 0.12           # 步骤3：逐行筛选阈值（每一行）
inclusive <- FALSE                 # TRUE: >=；FALSE: >

# === 读取与校验 ===
df <- expname
need_cols <- c("t", "reassigned_label_raw", "organelle_area_raw")
miss <- setdiff(need_cols, names(df))
if (length(miss) > 0) stop("缺少必要列：", paste(miss, collapse = ", "))

# 若无 label 列，先造一个占位列
if (!"label" %in% names(df)) {
  df$label <- NA_character_
}

# 规范类型
df <- df %>%
  mutate(
    t = suppressWarnings(as.integer(t)),
    reassigned_label_raw = trimws(as.character(reassigned_label_raw)),
    label = if (!is.character(label)) as.character(label) else label,
    organelle_area_raw = suppressWarnings(as.numeric(organelle_area_raw))
  )

# ---------- 公用统计 ----------
id_stats <- df %>%
  group_by(reassigned_label_raw) %>%
  summarise(
    n_t        = n_distinct(t),
    med_area   = suppressWarnings(stats::median(organelle_area_raw, na.rm = TRUE)),
    .groups = "drop"
  )

# === 步骤1：t==0 小面积 ID ===
small_ids_t0 <- df %>%
  filter(t == 0, !is.na(organelle_area_raw), organelle_area_raw < min_area_t0) %>%
  filter(!is.na(reassigned_label_raw)) %>%
  distinct(reassigned_label_raw) %>%
  pull(reassigned_label_raw)

# 记录被删行（步骤1）
removed_step1_rows <- df %>%
  filter(reassigned_label_raw %in% small_ids_t0) %>%
  transmute(t, label, reassigned_label_raw, removed_by = "t0")

# 删除后数据
df_step1 <- df %>% filter(!(reassigned_label_raw %in% small_ids_t0))

# === 步骤2：短命小点（中位面积 < min_area_short 且 n_t ≤ 阈值）===
short_lived_ids <- id_stats %>%
  filter(!(reassigned_label_raw %in% small_ids_t0)) %>%
  filter(!is.na(med_area), n_t <= short_lived_max_frames, med_area < min_area_short) %>%
  pull(reassigned_label_raw)

removed_step2_rows <- df_step1 %>%
  filter(reassigned_label_raw %in% short_lived_ids) %>%
  transmute(t, label, reassigned_label_raw, removed_by = "short_lived")

df_step2 <- df_step1 %>% filter(!(reassigned_label_raw %in% short_lived_ids))

# === 步骤3：全局逐行筛选（每一行面积阈值）===
# === 3) 全局逐行筛选：对每一行都应用面积阈值 ===
if (isTRUE(inclusive)) {
  keep_mask <- is.na(df_step2$organelle_area_raw) | (df_step2$organelle_area_raw >= min_area_rowwise)
} else {
  keep_mask <- is.na(df_step2$organelle_area_raw) | (df_step2$organelle_area_raw >  min_area_rowwise)
}
outname <- df_step2[keep_mask, ]

removed_step3_rows <- df_step2 %>%
  mutate(keep_flag = keep_mask) %>%
  filter(!keep_flag) %>%
  select(t, label, reassigned_label_raw, organelle_area_raw) %>%
  arrange(organelle_area_raw)

# === 导出主结果 ===
readr::write_csv(outname, out_csv)

# === 导出各步骤日志（可选）===
# 1) t0 被删 ID 行
readr::write_csv(removed_step1_rows, sub("\\.csv$", "_removed_rows_t0.csv", out_csv))
# 2) 短命小点被删行
readr::write_csv(removed_step2_rows, sub("\\.csv$", "_removed_rows_short_lived.csv", out_csv))
# 3) 逐行筛选被删行
readr::write_csv(removed_step3_rows, sub("\\.csv$", "_removed_rows_rowwise.csv", out_csv))

# === 汇总导出：所有被删行（去重合并）===
removed_rows_all <- bind_rows(removed_step1_rows, removed_step2_rows, removed_step3_rows) %>%
  distinct(t, label, reassigned_label_raw, removed_by, .keep_all = TRUE)

readr::write_csv(removed_rows_all, sub("\\.csv$", "_removed_rows_all.csv", out_csv))

df_clean <- df[, colSums(!is.na(df)) > 0]
# === 控制台摘要 ===
cat(sprintf(
  paste0(
    "完成：原始 %d -> 清洗后 %d（总删 %d）。\n",
    "阈值：t0=%.3f；短命=%.3f（≤%d 帧，中位面积）；逐行=%.3f（%s）。\n",
    "导出：_removed_rows_t0 / _removed_rows_short_lived / _removed_rows_rowwise / _removed_rows_all\n"
  ),
  nrow(df), nrow(df_clean), nrow(df) - nrow(df_clean),
  min_area_t0, min_area_short, short_lived_max_frames,
  min_area_rowwise, ifelse(inclusive, ">=", ">")
))
return(outname)
}