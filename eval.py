import tensorflow as tf

from train import build_model, make_dataset, shard_names

SHARD_ROOT = "D:/yiff"
model = build_model()
dataset = make_dataset(shard_names(SHARD_ROOT), model.inputs[0].shape[1:]).batch(32)
model.compile(
    metrics=[
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.BinaryAccuracy(name="acc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]
)
model.load_weights(tf.io.gfile.glob("./train/*.h5")[-1])
model.evaluate(dataset, verbose=1)
