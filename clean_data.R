# cleardata.R
cleandata2<- function(exp_clean){
drop_cols <- c("X", "t", "label","reassigned_label_raw","x_raw","y_raw","z_raw")
df <- exp_clean[ , !(names(exp_clean) %in% drop_cols)]
# 建议先安装/加载需要的包：
# install.packages(c("readr", "dplyr", "stringr"))
library(readr)
library(dplyr)
library(stringr)


# 4) 识别数值列（排除标签列 type）
num_cols <- names(df)[names(df) != "type" & sapply(df, is.numeric)]

# 5) 统一处理 Inf / -Inf 为 NA
df[num_cols] <- lapply(df[num_cols], function(x) {
  x[is.infinite(x)] <- NA_real_
  x
})

# 6) 用整体中位数填补缺失（NaN/NA）
med <- vapply(df[num_cols],
              function(x) median(x, na.rm = TRUE),
              numeric(1))
for (nm in num_cols) {
  x <- df[[nm]]
  x[is.na(x)] <- med[[nm]]
  df[[nm]] <- x
}

# 7) 找出名字中包含 intensity 的“强度列”
intensity_cols <- num_cols[grepl("intensity", num_cols, ignore.case = TRUE)]

# 对这些强度列先 clip 到 >= 0，再做 log1p
for (nm in intensity_cols) {
  x <- df[[nm]]
  x[x < 0] <- 0          # clip(lower=0)
  df[[nm]] <- log1p(x)   # log1p
}

# 8) 对所有数值列做稳健截断 (0.1% 和 99.9% 分位数)
for (nm in num_cols) {
  x <- df[[nm]]
  qs <- quantile(x, probs = c(0.001, 0.999),
                 na.rm = TRUE, names = FALSE)
  x[x < qs[1]] <- qs[1]
  x[x > qs[2]] <- qs[2]
  df[[nm]] <- x
}

return(df)
}
# 9) Python 里是 astype('float32')，R 默认是 double
# 如果只是为了后面模型训练，一般保持 numeric 就可以了；
# 真要省内存可以考虑使用 float 包（这里先不强制转）

# 10) 写出结果
# write_csv(df, dst)
# 
# cat("clean saved ->", dst, "\n")
# cat("n_rows:", nrow(df), " n_num_cols:", length(num_cols), "\n")
# cat("transformed intensity columns (first 10):\n")
# print(head(intensity_cols, 10))
# cat("... total", length(intensity_cols), "\n")
