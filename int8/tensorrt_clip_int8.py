import numpy as np
# import torch
# import torch.nn as nn
import util_trt
import glob, os, cv2

BATCH_SIZE = 4
BATCH = 30
height = 224
width = 224
CALIB_IMG_DIR = '/home/clip2onnx/coco128/'
onnx_model_path = '/home/clip2onnx/v.onnx'


def preprocess(img):
    resized_img = cv2.resize(img, (224, 224),interpolation=cv2.INTER_LINEAR)
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    # 归一化到0-1范围
    normalized_img = rgb_img.astype(np.float32) / 255.0
    # 减去均值除以方差（假设mean和stddev是预先计算好的）
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    stddev = np.array([0.26862954, 0.26130258, 0.27577711])
    # 减去均值除以方差
    normalized_img = (normalized_img - mean)/ stddev
    # 转换为CHW布局
    chw_img = normalized_img.transpose((2, 0, 1))
    return chw_img


class DataLoader:
    def __init__(self):
        self.index = 0
        self.length = BATCH
        self.batch_size = BATCH_SIZE
        # self.img_list = [i.strip() for i in open('calib.txt').readlines()]
        self.img_list = glob.glob(os.path.join(CALIB_IMG_DIR, "*.jpg"))
        assert len(self.img_list) > self.batch_size * self.length, '{} must contains more than '.format(
            CALIB_IMG_DIR) + str(self.batch_size * self.length) + ' images to calib'
        print('found all {} images to calib.'.format(len(self.img_list)))
        self.calibration_data = np.zeros((self.batch_size, 3, height, width), dtype=np.float32)

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), 'not found!!'
                img = cv2.imread(self.img_list[i + self.index * self.batch_size])
                img = preprocess(img)
                self.calibration_data[i] = img

            self.index += 1

            # example only
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.length


def main():
    # onnx2trt
    fp16_mode = False
    int8_mode = True
    print('*** onnx to tensorrt begin ***')
    # calibration
    calibration_stream = DataLoader()
    engine_model_path = "VmodelInt8.engine"
    calibration_table = 'best_calibration.cache'
    # fixed_engine,校准产生校准表
    engine_fixed = util_trt.get_engine(BATCH_SIZE, onnx_model_path, engine_model_path, fp16_mode=fp16_mode,
                                       int8_mode=int8_mode, calibration_stream=calibration_stream,
                                       calibration_table_path=calibration_table, save_engine=True)
    print(engine_fixed)
    assert engine_fixed, 'Brokenls engine_fixed'
    print('*** onnx to tensorrt completed ***\n')


if __name__ == '__main__':
    main()