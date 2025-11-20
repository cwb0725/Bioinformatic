# Bioinformatic_Mtitochondria
#单帧处理
python export_mitograph_like.py \
  --skel /path/skeleton.tif \
  --out_dir /path/frame_000 \
  --mask /path/mask.tif \        # 可选，用于计算width_(um)，没有就不填
  --raw /path/raw.tif \          # 可选，用于采样pixel_intensity
  --px_um 0.1 \                  # 像素尺寸(µm/px)
  --prune_len 8 \                # 可选，剪掉小于8px的端点短枝
  --min_cc 20                    # 可选，去除小碎屑连通域

#多帧批量处理
python export_mitograph_like.py \
  --glob "/path/skel_frames/*.tif" \
  --out_root /path/out \
  --mask_glob "/path/mask_frames/*.tif" \   # 可选
  --raw_glob  "/path/raw_frames/*.tif" \    # 可选
  --px_um 0.1 --prune_len 8 --min_cc 20

####去除小碎片 以及上色 ######（Fiji）
Plugins → MorphoLibJ → Binary Images → Area Opening…
Plugins → MorphoLibJ → Binary Images → Connected Components Labeling


###Tabtransformer###
##训练##
CUBLAS_WORKSPACE_CONFIG=:4096:8 python /home/CWB/MTdata/pythonproject/tabtransformer_full_pipeline_v2p3.py   --data ./MT_tiff/exp_all_clean.csv   --outdir ./MTtiff/exp_all_out   --target type   --folds 5 --epochs 100 --batch_size 256   --lr 3e-4 --weight_decay 1e-6   --d_model 256 --depth 4 --heads 8   --dropout 0.05 --label_smoothing 0.0   --scaler robust --rare_threshold 2  --perm_repeats 1  --no_amp
