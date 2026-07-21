from tensorflow import keras
from keras import layers
from keras.regularizers import L1L2

CUTS = {"c1", "c2", "c3", "c4"}


def _reg(l1=0.0, l2=0.0):
    return L1L2(l1=l1, l2=l2) if (l1 != 0.0 or l2 != 0.0) else None


def conv3x3(x, filters, stride=1, name=None, l1=0.0, l2=0.0):
    x = layers.ZeroPadding2D(padding=1, name=f"{name}_pad")(x)
    return layers.Conv2D(filters, 3, strides=stride, use_bias=False, name=name, kernel_regularizer=_reg(l1, l2))(x)


def basic_block(x, filters, stride=1, name=None, l1=0.0, l2=0.0):
    identity = x
    out = conv3x3(x, filters, stride=stride, name=f"{name}.conv1", l1=l1, l2=l2)
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"{name}.bn1")(out)
    out = layers.ReLU(name=f"{name}.relu1")(out)
    out = conv3x3(out, filters, stride=1, name=f"{name}.conv2", l1=l1, l2=l2)
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"{name}.bn2")(out)
    if stride != 1 or int(identity.shape[-1]) != filters:
        identity = layers.Conv2D(filters, 1, strides=stride, use_bias=False, name=f"{name}.downsample.0", kernel_regularizer=_reg(l1, l2))(identity)
        identity = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"{name}.downsample.1")(identity)
    out = layers.Add(name=f"{name}.add")([identity, out])
    return layers.ReLU(name=f"{name}.relu2")(out)


def stem(x, l1=0.0, l2=0.0):
    x = layers.ZeroPadding2D(padding=1, name="conv1_pad")(x)
    x = layers.Conv2D(64, 3, strides=1, use_bias=False, name="conv1", kernel_regularizer=_reg(l1, l2))(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="bn1")(x)
    return layers.ReLU(name="relu1")(x)


def layer1(x, l1=0.0, l2=0.0):
    x = basic_block(x, 64, stride=1, name="layer1.0", l1=l1, l2=l2)
    return basic_block(x, 64, stride=1, name="layer1.1", l1=l1, l2=l2)


def layer2(x, l1=0.0, l2=0.0):
    x = basic_block(x, 128, stride=2, name="layer2.0", l1=l1, l2=l2)
    return basic_block(x, 128, stride=1, name="layer2.1", l1=l1, l2=l2)


def layer3(x, l1=0.0, l2=0.0):
    x = basic_block(x, 256, stride=2, name="layer3.0", l1=l1, l2=l2)
    return basic_block(x, 256, stride=1, name="layer3.1", l1=l1, l2=l2)


def layer4(x, l1=0.0, l2=0.0):
    x = basic_block(x, 512, stride=2, name="layer4.0", l1=l1, l2=l2)
    return basic_block(x, 512, stride=1, name="layer4.1", l1=l1, l2=l2)


def make_f(cut, input_shape, width="standard", l1=0.0, l2=0.0):
    cut = str(cut).lower()
    if cut not in CUTS:
        raise ValueError(f"Unsupported ResNet-18 cut {cut}; expected c1-c4")
    xin = layers.Input(input_shape)
    x = stem(xin, l1=l1, l2=l2)
    x = layer1(x, l1=l1, l2=l2)
    if cut in {"c2", "c3", "c4"}:
        x = layer2(x, l1=l1, l2=l2)
    if cut in {"c3", "c4"}:
        x = layer3(x, l1=l1, l2=l2)
    if cut == "c4":
        x = layer4(x, l1=l1, l2=l2)
    return keras.Model(xin, x)


def make_g(cut, input_shape, num_classes=10, include_h=False, dropout=0.0, width="standard", l1=0.0, l2=0.0):
    cut = str(cut).lower()
    if cut not in CUTS:
        raise ValueError(f"Unsupported ResNet-18 cut {cut}; expected c1-c4")
    if include_h:
        raise NotImplementedError("ResNet-18 adaptation currently supports vanilla SL only.")
    xin = layers.Input(input_shape)
    x = xin
    if cut == "c1":
        x = layer2(x, l1=l1, l2=l2)
    if cut in {"c1", "c2"}:
        x = layer3(x, l1=l1, l2=l2)
    if cut in {"c1", "c2", "c3"}:
        x = layer4(x, l1=l1, l2=l2)
    x = layers.GlobalAveragePooling2D(name="avgpool")(x)
    if dropout > 0.0:
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(num_classes, name="fc", kernel_regularizer=_reg(l1, l2))(x)
    return keras.Model(xin, x)


def make_h(num_classes=10, dropout=0.0, l1=0.0, l2=0.0):
    raise NotImplementedError("ResNet-18 adaptation currently supports vanilla SL only.")
