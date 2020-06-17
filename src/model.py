from keras.models import Model
from keras.layers import (
    Conv2D,
    Input,
    AveragePooling2D,
    Flatten,
    Dense,
    Dropout,
    Softmax,
    Add,
    BatchNormalization,
    ReLU,
)
from src.layer import Conv2D as RQConv2D
from src.layer import Dense as RQDense


subfig_scale = 64  # sub-fig size
scale = 512  # origin image size
subfig_num = (scale // subfig_scale) ** 2  # No. of sub-fig how one image can split
train_num = 80  # No. of image in test set
val_num = 10  # No. of image in validation set
test_num = 10  # No. of image in test set


def RQ_conv_res_bn_model():
    inputs = Input((subfig_scale, subfig_scale, 3))
    net = Conv2D(4, 3, padding="same")(inputs)
    net = RQConv2D(64, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = ReLU()(net)
    net = RQConv2D(64, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = ReLU()(net)
    net = RQConv2D(64, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = ReLU()(net)
    net = RQConv2D(64, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = ReLU()(net)
    net = RQConv2D(64, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = ReLU()(net)
    net = RQConv2D(64, 3, padding="same")(net)
    net = BatchNormalization()(net)
    net = ReLU()(net)
    net = Conv2D(3, 3, padding="same")(net)
    net = Add()([inputs, net])
    return Model(inputs=inputs, outputs=net)
