# coding=utf-8
# project

PROJECT_PATH = "J:/Sleep-Wake-Joint-Fusion-main/"

# data_type = 'airport'
data_type = 'M3FD'


if data_type == 'airport':
    DATA_PATH = "D:/new_airport"
    DATA = {"CLASSES":['airplane', 'man', 'car'],
            "NUM":3}
elif data_type == 'M3FD':
    DATA_PATH = "J:/tan/M3FD/M3FD_Detection"
    DATA = {"CLASSES":['People','Car','Bus', 'Motorcycle', 'Truck','Lamp'],
            "NUM":6}

# train
TRAIN = {
         "TRAIN_IMG_SIZE":320,
         "AUGMENT":True,
         "MULTI_SCALE_TRAIN":False,
         "IOU_THRESHOLD_LOSS":0.5,
         "NUMBER_WORKERS":0,
         "MOMENTUM":0.9,
         "WEIGHT_DECAY":0.0005,
         }


# test
TEST = {
        "TEST_IMG_SIZE":320,
        "NUMBER_WORKERS":4,
        "CONF_THRESH":0.1,
        "NMS_THRESH":0.5,
        "MULTI_SCALE_TEST":False,
        "FLIP_TEST":False
        }
