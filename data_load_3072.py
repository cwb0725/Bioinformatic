import glob
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img


class DataProcess(object):

    def __init__(self,
                 out_rows,
                 out_cols,
                 data_path="/home/CWB/Swin_unet/ori_fig",
                 label_path="/home/CWB/Swin_unet/mask",
                 test_path="/home/CWB/Swin_unet/test",
                 npy_path="/home/CWB/Swin_unet/modl_unet/npy_data/",
                 img_type="tif"):

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.test_path = test_path
        self.npy_path = npy_path
        self.img_type = img_type

    # create training data (.npy)
    
    def create_train_data(self):
        print('-' * 30)
        print('Creating training images...')
        print('-' * 30)

        imgs = glob.glob(self.data_path + "/*." + self.img_type)
        print("Total images:", len(imgs))

        imgdatas = np.ndarray(
            (len(imgs), self.out_rows, self.out_cols, 1),
            dtype=np.uint8
        )
        imglabels = np.ndarray(
            (len(imgs), self.out_rows, self.out_cols, 1),
            dtype=np.uint8
        )

        for i, imgname in enumerate(imgs):

            midname = imgname[imgname.rindex("/") + 1:]

            img = load_img(
                self.data_path + "/" + midname,
                color_mode='grayscale',
                target_size=(self.out_rows, self.out_cols)
            )
            label = load_img(
                self.label_path + "/" + midname,
                color_mode='grayscale',
                target_size=(self.out_rows, self.out_cols)
            )

            img = img_to_array(img)
            label = img_to_array(label)

            imgdatas[i] = img
            imglabels[i] = label

            if i % 10 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))

        print('Saving .npy files...')
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)

        print('Saving done.')


    # load training data
    
    def load_train_data(self):
        train = np.load(self.npy_path + "/imgs_train.npy")
        mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")

        train = train.astype('float32')
        mask_train = mask_train.astype('float32')

        train /= 255.0
        mean = train.mean(axis=0)
        train -= mean

        mask_train /= 255.0
        mask_train[mask_train > 0.5] = 1
        mask_train[mask_train <= 0.5] = 0

        return train, mask_train


# -------------------------------------------------
# standalone run
# -------------------------------------------------
if __name__ == "__main__":
    mydata = DataProcess(3072, 3072)
    mydata.create_train_data()
