import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import os

import tensorflow as tf
import tensorflow_addons as tfa

import heteroscedastic
from DeepDanbooru.deepdanbooru import deepdanbooru as dd
from robust_loss.adaptive import AdaptiveLossFunction

with open("./tags.txt", encoding="utf8") as istrm:
    tags = istrm.read().split()


def jensen_shannon():
    @tf.function
    def loss(y, p):
        d = tf.keras.losses.kl_divergence
        m = 0.5 * y + 0.5 * p
        return 0.5 * d(y, m) + 0.5 * d(p, m)

    return loss


def focal_loss(
    alpha=0.25, gamma=2.00, epsilon=1e-7, inner=tf.keras.losses.binary_crossentropy
):
    @tf.function
    def loss(y, p):
        y = tf.math.maximum(tf.cast(y, tf.float32), epsilon)
        p = tf.math.maximum(tf.cast(p, tf.float32), epsilon)
        base = inner(y, p)[:, None]
        negative = p * y + (1 - p) * (1 - y)
        positive = alpha * y + (1 - alpha) * (1 - y)
        return base * positive * ((1 - negative) ** gamma)

    return loss


def make_adaptive(inner, num_channels=len(tags)):
    adapt = AdaptiveLossFunction(num_channels=num_channels, float_dtype=tf.float32)

    @tf.function
    def loss(y, p):
        u = inner(y, p)
        with tf.variable_creator_scope("adaloss"):
            return adapt(u)

    return loss


def preprocessor(image_shape, num_tags=len(tags), crop=False):
    def preprocess(x, y):
        x = tf.io.decode_image(tf.squeeze(x), expand_animations=False)
        x = tf.cast(x, tf.float32)
        x = x[..., : image_shape[-1]]
        if crop and tf.reduce_all(tf.shape(x) > image_shape):
            x = tf.image.random_crop(x, image_shape)
        if tf.shape(x)[-1] < 3:
            x = tf.reduce_mean(x, axis=-1)
            x = tf.expand_dims(x, axis=-1)
            x = tf.image.grayscale_to_rgb(x)
        x = tf.image.resize(x, image_shape[:-1])
        return x, y[:num_tags]

    return preprocess


def record_parser():
    feature_desc = dict(
        tag_indxs=tf.io.VarLenFeature(tf.int64),
        tag_names=tf.io.FixedLenFeature((), tf.string),
        image_str=tf.io.FixedLenFeature((), tf.string),
        post_id=tf.io.FixedLenFeature((), tf.int64),
    )

    @tf.function
    def parse(record):
        return tf.io.parse_single_example(record, feature_desc)

    return parse


def record_deserializer(num_tags=len(tags)):
    @tf.function
    def decode(record):
        x = record["image_str"]
        ids = tf.sparse.to_dense(record["tag_indxs"])
        y = tf.SparseTensor(ids[:, None], tf.ones_like(ids), [len(tags)])
        y = tf.sparse.to_dense(y)[:num_tags]
        y = tf.cast(y, tf.float32)
        return x, y

    return decode


def shard_names(root):
    return tf.io.gfile.glob(f"{root}/*.tfrecords")


def read_records(shards):
    return (
        tf.data.Dataset.from_tensor_slices(shards)
        .shuffle(len(shards))
        .repeat()
        .interleave(
            lambda shard: (
                tf.data.TFRecordDataset(shard).map(
                    record_parser(),
                    num_parallel_calls=1,
                )
            ),
            cycle_length=1,
            block_length=256,
            num_parallel_calls=1,
            deterministic=False,
        )
        .prefetch(tf.data.AUTOTUNE)
    )


def make_dataset(shards, image_shape, num_tags=len(tags)):
    return (
        read_records(shards)
        .map(record_deserializer(num_tags))
        .map(
            preprocessor(image_shape, num_tags=num_tags),
            num_parallel_calls=4,
            deterministic=False,
        )
        .apply(tf.data.experimental.ignore_errors())
    )


def fwd_attention_head(x, out_units=len(tags)):
    x = tf.keras.layers.MultiHeadAttention(2, 2)(x, x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LocallyConnected2D(*x.shape[-2:][::-1])(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    return heteroscedastic.MCSigmoidDenseFA(
        out_units,
        logits_only=True,
        name="denoising",
    )(x)


def res_attention_head(x, enc_units=512, out_units=len(tags)):
    dsampled = tf.keras.layers.LayerNormalization()(x)
    conv_cfg = dict(
        filters=dsampled.shape[3],
        kernel_size=max(dsampled.shape[1:3]),
        strides=min(dsampled.shape[1:3]) // 2,
    )
    attended = tf.keras.layers.MultiHeadAttention(2, 2)(dsampled, dsampled)
    attended = tf.keras.layers.SeparableConv2D(**conv_cfg)(attended)

    dsampled = tf.keras.layers.SeparableConv2D(**conv_cfg)(dsampled)
    squeezed = tf.keras.layers.Dense(enc_units)(dsampled)
    squeezed = tf.keras.layers.Activation(tf.nn.silu)(squeezed)
    squeezed = tf.keras.layers.Dense(attended.shape[-1])(squeezed)

    outputs = tf.keras.layers.Add()([squeezed, attended])
    outputs = tf.keras.layers.LayerNormalization()(outputs)
    outputs = tf.keras.layers.Flatten()(outputs)
    logits = heteroscedastic.MCSigmoidDenseFA(
        out_units, name="denoising", logits_only=True, temperature=4.0
    )(outputs)
    return tf.keras.layers.Activation(tf.nn.sigmoid)(logits)


def build_model(num_tags=len(tags)):
    image_shape = (224, 224, 3)
    inputs = tf.keras.layers.Input(shape=image_shape, name="images")
    preprocess = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip(),
            tf.keras.layers.experimental.preprocessing.RandomContrast(0.25),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.25),
            tf.keras.layers.experimental.preprocessing.Resizing(*image_shape[:-1]),
            tf.keras.layers.Lambda(tf.keras.applications.resnet.preprocess_input),
        ],
        name="preprocess",
    )
    outputs = preprocess(inputs)
    outputs = tf.keras.applications.EfficientNetB0(
        weights=None, include_top=False, input_shape=image_shape
    )(outputs)
    outputs = res_attention_head(outputs, out_units=num_tags)
    return tf.keras.Model(inputs, outputs)


class CycleSchedule(tf.optimizers.schedules.LearningRateSchedule):
    def __init__(self, min_lr, max_lr, total_steps, epsilon=1e-7, cycles=1, **kwargs):
        super().__init__(**kwargs)
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.epsilon = epsilon
        self.cycles = cycles

    def __call__(self, step):
        q = tf.clip_by_value(
            (step * self.cycles % self.total_steps) / self.total_steps,
            self.epsilon,
            1 - self.epsilon,
        )
        lr = self.max_lr - 2 * self.max_lr * tf.math.abs(q - 0.5)
        return tf.clip_by_value(lr, self.min_lr, self.max_lr)

    def get_config(self):
        return {
            "min_lr": self.min_lr,
            "max_lr": self.max_lr,
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "cycles": self.cycles,
        }


if __name__ == "__main__":
    model = build_model()
    image_shape = model.inputs[0].shape[1:]

    n_logs = len(tf.io.gfile.glob("./logs/*"))
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "./train/e6tag_sm-{epoch:05}-{auc:.2f}.ckpt",
            monitor="auc",
            mode="max",
            save_best_only=False,
            save_weights_only=True,
            save_freq=1024,
        ),
        tf.keras.callbacks.TensorBoard(
            f"./logs/{n_logs+1}",
            update_freq=1,
            write_graph=False,
            write_steps_per_second=True,
        ),
    ]

    initial_epoch = 0
    if ckpt := tf.train.latest_checkpoint("./train"):
        model.load_weights(ckpt)
        initial_epoch, _ = os.path.splitext(ckpt.split("-")[1])
        initial_epoch = int(initial_epoch) - 1

    shards = shard_names("D:/yiff")  # [::2][:64]
    batch_size = 16
    dataset = make_dataset(shards, image_shape).batch(batch_size)
    metrics = [
        tf.keras.metrics.AUC(name="auc"),
    ]

    shard_size = 256
    total_epochs = 8
    steps_per_epoch = int(2 ** 16 / batch_size / total_epochs)
    lr_schedule = CycleSchedule(0.1, 4.0, steps_per_epoch * total_epochs, cycles=1)
    train_config = dict(
        x=dataset,
        verbose=1,
        epochs=total_epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        shuffle=False,
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.99),
        loss=make_adaptive(focal_loss()),
        metrics=metrics,
    )
    model.fit(**train_config)
