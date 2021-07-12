import os

os.environ |= {"CUDA_VISIBLE_DEVICES": "-1", "TF_CPP_MIN_LOG_LEVEL": "2"}

import tensorflow as tf

from download import SHARD_LENGTH
from train import build_model, make_dataset

model = build_model()
model.load_weights(tf.train.latest_checkpoint("./train"))
model.save("./e6tag.h5", overwrite=True, save_traces=True, include_optimizer=False)


def representative_dataset():
    for x, _ in iter(make_dataset().take(SHARD_LENGTH)):
        yield [x]


cvter = tf.lite.TFLiteConverter.from_keras_model(model)
cvter.optimizations = [tf.lite.Optimize.DEFAULT]
cvter.representative_dataset = representative_dataset
quant = cvter.convert()
quant.write_bytes("./e6tag.tflite")
