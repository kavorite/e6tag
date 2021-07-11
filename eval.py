import tensorflow as tf

from train import build_model, make_dataset, shard_names

SHARD_ROOT = "D:/yiff"
model = build_model()
model.load_weights(tf.train.latest_checkpoint("./train"))
model = tf.keras.Model(
    model.inputs, tf.keras.layers.Activation(tf.nn.sigmoid)(model.outputs[0])
)
dataset = make_dataset(shard_names(SHARD_ROOT), model.inputs[0].shape[1:]).batch(32)
model.compile(
    metrics=[
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.BinaryAccuracy(name="acc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]
)
model.evaluate(dataset, verbose=1)
