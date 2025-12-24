import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D,
    Dropout, Concatenate, Conv2DTranspose
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ----------------------------
# 512x512 Patch Generator
# ----------------------------
def patch_generator(imgs, masks,
                    patch=512,
                    steps=200,
                    p_fg=0.8,
                    min_fg_ratio=0.002,
                    max_tries=30,
                    seed=123):
    """
    imgs:  (N, H, W, 1) float32, 已完成 /255 和 mean-sub
    masks: (N, H, W, 1) float32, 0/1
    yield: (1, patch, patch, 1), (1, patch, patch, 1)
    """
    rng = np.random.default_rng(seed)
    N, H, W, _ = imgs.shape
    assert H >= patch and W >= patch, "Image smaller than patch size."

    while True:
        for _ in range(steps):
            # 选一张图
            idx = int(rng.integers(0, N))

            # 决定是否偏向前景
            want_fg = (rng.random() < p_fg)

            y0 = x0 = None

            if want_fg:
                # 尝试多次，找到含前景的 patch
                found = False
                for _t in range(max_tries):
                    yy = int(rng.integers(0, H - patch + 1))
                    xx = int(rng.integers(0, W - patch + 1))
                    m = masks[idx, yy:yy + patch, xx:xx + patch, 0]
                    if float(m.mean()) >= min_fg_ratio:
                        y0, x0 = yy, xx
                        found = True
                        break
                if not found:
                    # 找不到就退回随机
                    y0 = int(rng.integers(0, H - patch + 1))
                    x0 = int(rng.integers(0, W - patch + 1))
            else:
                # 纯随机 patch
                y0 = int(rng.integers(0, H - patch + 1))
                x0 = int(rng.integers(0, W - patch + 1))

            x_patch = imgs[idx, y0:y0 + patch, x0:x0 + patch, :]
            y_patch = masks[idx, y0:y0 + patch, x0:x0 + patch, :]

            # 保持与你原训练一致：输出 float32
            x_patch = x_patch.astype(np.float32)
            y_patch = y_patch.astype(np.float32)

            # Keras generator 期望 batch 维度
            yield np.expand_dims(x_patch, axis=0), np.expand_dims(y_patch, axis=0)


# ----------------------------
# UNet (与你原 train.py 完全同结构，只是输入改为512)
# ----------------------------
class myUnet(object):
    def __init__(self, img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 1))

        conv1 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2DTranspose(512, 2, activation='relu', padding='same',
                              kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        merge6 = Concatenate(axis=3)([drop4, up6])
        conv6 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv6)

        up7 = Conv2DTranspose(256, 2, activation='relu', padding='same',
                              kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = Concatenate(axis=3)([conv3, up7])
        conv7 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv7)

        up8 = Conv2DTranspose(128, 2, activation='relu', padding='same',
                              kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = Concatenate(axis=3)([conv2, up8])
        conv8 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv8)

        up9 = Conv2DTranspose(64, 2, activation='relu', padding='same',
                              kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = Concatenate(axis=3)([conv1, up9])
        conv9 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same',
                       kernel_initializer='he_normal')(conv9)

        outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=outputs)

        # 重点：用更抗不平衡的 loss（你前景4.3%）
        loss_fn = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.75)

        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss=loss_fn,
            metrics=[]
        )
        return model

    def train(self):
        print("loading data")

        imgs_train = np.load("../npydata/imgs_train.npy").astype("float32")
        imgs_mask_train = np.load("../npydata/imgs_mask_train.npy").astype("float32")

        # 与你原脚本一致：归一化
        imgs_train /= 255.0
        imgs_mask_train /= 255.0

        # 二值化（你已验证 mask unique [0,1]，这里保持一致）
        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0

        # 与你原脚本一致：逐像素 mean-sub（但注意这会占内存）
        mean = imgs_train.mean(axis=0)
        imgs_train -= mean

        print("mask unique:", np.unique(imgs_mask_train))
        print("foreground ratio:", float(imgs_mask_train.mean()))
        print("loading data done")

        # ---- 按“图像级”划分 train/val，避免泄漏 ----
        N = imgs_train.shape[0]
        idx = np.arange(N)
        np.random.shuffle(idx)
        split = int(N * 0.8)
        tr_idx, va_idx = idx[:split], idx[split:]

        x_tr, y_tr = imgs_train[tr_idx], imgs_mask_train[tr_idx]
        x_va, y_va = imgs_train[va_idx], imgs_mask_train[va_idx]

        model = self.get_unet()
        print("got unet")

        os.makedirs("../model", exist_ok=True)
        model_checkpoint = ModelCheckpoint(
            "../model/U-Net_patch512.keras",
            monitor="val_loss",
            mode="min",
            verbose=1,
            save_best_only=True
        )

        # ---- Patch 训练参数（你可以按需调）----
        PATCH = 512
        STEPS_PER_EPOCH = 200       # 每个 epoch 抽 200 个 patch（可调大）
        VAL_STEPS = 50              # 验证抽 50 个 patch（可调）
        EPOCHS = 50

        # 前景偏置：80% 采样尽量含线粒体
        P_FG = 0.8
        MIN_FG_RATIO = 0.002        # patch 内前景>=0.2%就算含前景（可调：0.001~0.01）
        MAX_TRIES = 30

        train_gen = patch_generator(
            x_tr, y_tr,
            patch=PATCH,
            steps=STEPS_PER_EPOCH,
            p_fg=P_FG,
            min_fg_ratio=MIN_FG_RATIO,
            max_tries=MAX_TRIES,
            seed=123
        )

        val_gen = patch_generator(
            x_va, y_va,
            patch=PATCH,
            steps=VAL_STEPS,
            p_fg=0.5,               # 验证不必太偏置，适中即可
            min_fg_ratio=MIN_FG_RATIO,
            max_tries=MAX_TRIES,
            seed=999
        )

        starttrain = datetime.datetime.now()
        print("Fitting model...")

        history = model.fit(
            train_gen,
            steps_per_epoch=STEPS_PER_EPOCH,
            epochs=EPOCHS,
            validation_data=val_gen,
            validation_steps=VAL_STEPS,
            callbacks=[model_checkpoint],
            verbose=1
        )

        endtrain = datetime.datetime.now()
        print("train time:", endtrain - starttrain)

        # 训练完简单检查：看看输出是不是还全贴近0
        x1, _ = next(train_gen)
        p = model.predict(x1, verbose=0)[0, ..., 0]
        print("quick pred stats:", float(p.min()), float(p.max()), float(p.mean()))


if __name__ == "__main__":
    myunet = myUnet(512, 512)
    myunet.train()
