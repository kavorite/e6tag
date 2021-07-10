import os
import subprocess as sp
import urllib.request as http

os.environ |= dict(TF_CPP_MIN_LOG_LEVEL="2", CUDA_VISIBLE_DEVICES="-1")
import tensorflow as tf

from train import build_model, tags

ckpt = tf.io.gfile.glob("./train/*.h5")[-1]
model = build_model()
model.load_weights(ckpt)
img_uri = (
    sp.run("PowerShell Get-Clipboard", capture_output=True)
    .stdout.decode("utf8")
    .strip()
)

with http.urlopen(http.Request(img_uri, headers={"User-Agent": "e6tag test"})) as rsp:
    img_str = rsp.read()

img = tf.image.decode_image(img_str)[..., :3]
dim = tf.reduce_min(img.shape[:-1])
img = tf.image.resize(img, [dim, dim])
target_dim = model.inputs[0].shape[1:3]
img = tf.image.resize(img, target_dim)
img = tf.expand_dims(img, axis=0)
yhat = tf.squeeze(model(img, training=False))
yhat_min = tf.reduce_min(yhat)
yhat = tf.math.divide_no_nan(yhat - yhat_min, tf.reduce_max(yhat) - yhat_min)
hits = tf.gather(tags, tf.squeeze(tf.where(yhat > 0.5)))
hits = " ".join(t.numpy().decode("utf8") for t in hits)
print(hits)
