import os

os.environ |= {"CUDA_VISIBLE_DEVICES": "-1", "TF_CPP_MIN_LOG_LEVEL": "2"}

import tensorflow as tf

from train import build_model

model = build_model()
model.load_weights(tf.io.gfile.glob("./train/*.h5")[-1])
model.save("./e6tag.h5")
# cvter = tf.lite.TFLiteConverter.from_keras_model(model)
# cvter.optimizations = [tf.lite.Optimize.DEFAULT]
# quant = cvter.convert()
# quant.write_bytes("./yiff_tag.tflite")
