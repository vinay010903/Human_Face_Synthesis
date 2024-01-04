#
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
import os

z_size = 512
width = 512
height = 512
channels = 1
ctrl_sqrt = 6
start = 0
images_saved = 0

# Datasets loading for diffeerent res.
data_dir = "Female"

BATCH_SIZE = 16


def preprocess_image(image_path, resolution):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, [resolution, resolution])
    image = (image - 127.5) / 127.5
    return image


def create_generator_model(resolution):
    inputs = tf.keras.Input(shape=(z_size,))

    x = layers.Dense(4 * 4 * 256, name="dense", kernel_initializer="glorot_uniform")(
        inputs
    )
    x = layers.Reshape((4, 4, 256), name="reshape")(x)
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    x = layers.LeakyReLU(name="leaky_relu_1")(x)
    x = layers.Conv2D(
        256, (3, 3), padding="same", name="conv2d_1", kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization(name="batch_norm_2")(x)
    x = layers.LeakyReLU(name="leaky_relu_2")(x)
    four_out = layers.Conv2D(
        1,
        (3, 3),
        padding="same",
        activation="tanh",
        name="four_out",
        kernel_initializer="glorot_uniform",
    )(x)

    x = layers.UpSampling2D((2, 2), name="upsample_1")(x)  # Output: 8x8x256
    x = layers.Conv2D(
        256, (3, 3), padding="same", name="conv2d_3", kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization(name="batch_norm_3")(x)
    x = layers.LeakyReLU(name="leaky_relu_3")(x)
    x = layers.Conv2D(
        512, (3, 3), padding="same", name="conv2d_4", kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization(name="batch_norm_4")(x)
    x = layers.LeakyReLU(name="leaky_relu_4")(x)
    eight_out = layers.Conv2D(
        1,
        (3, 3),
        padding="same",
        activation="tanh",
        name="eight_out",
        kernel_initializer="glorot_uniform",
    )(x)

    x = layers.UpSampling2D((2, 2), name="upsample_2")(x)  # Output: 16x16x256
    x = layers.Conv2D(
        512, (3, 3), padding="same", name="conv2d_5", kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization(name="batch_norm_5")(x)
    x = layers.LeakyReLU(name="leaky_relu_5")(x)
    x = layers.Conv2D(
        512, (3, 3), padding="same", name="conv2d_6", kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization(name="batch_norm_6")(x)
    x = layers.LeakyReLU(name="leaky_relu_6")(x)
    sixteen_out = layers.Conv2D(
        1,
        (3, 3),
        padding="same",
        activation="tanh",
        name="sixteen_out",
        kernel_initializer="glorot_uniform",
    )(x)

    x = layers.UpSampling2D((2, 2), name="upsample_3")(x)  # Output: 32x32x256
    x = layers.Conv2D(
        512, (3, 3), padding="same", name="conv2d_7", kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization(name="batch_norm_7")(x)
    x = layers.LeakyReLU(name="leaky_relu_7")(x)
    x = layers.Conv2D(
        640, (3, 3), padding="same", name="conv2d_8", kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization(name="batch_norm_8")(x)
    x = layers.LeakyReLU(name="leaky_relu_8")(x)
    thirtytwo_out = layers.Conv2D(
        1,
        (3, 3),
        padding="same",
        activation="tanh",
        name="thirtwo_out",
        kernel_initializer="glorot_uniform",
    )(x)

    x = layers.UpSampling2D((2, 2), name="upsample_4")(x)  # Output: 64x64x256
    x = layers.Conv2D(
        512, (3, 3), padding="same", name="conv2d_9", kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization(name="batch_norm_9")(x)
    x = layers.LeakyReLU(name="leaky_relu_9")(x)
    x = layers.Conv2D(
        512, (3, 3), padding="same", name="conv2d_10", kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization(name="batch_norm_10")(x)
    x = layers.LeakyReLU(name="leaky_relu_10")(x)
    sixtyfour_out = layers.Conv2D(
        1,
        (3, 3),
        padding="same",
        activation="tanh",
        name="sixtfor_out",
        kernel_initializer="glorot_uniform",
    )(x)

    x = layers.UpSampling2D((2, 2), name="upsample_5")(x)  # Output: 128x128x512
    x = layers.Conv2D(
        512, (3, 3), padding="same", name="conv2d_11", kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization(name="batch_norm_11")(x)
    x = layers.LeakyReLU(name="leaky_relu_11")(x)
    x = layers.Conv2D(
        512, (3, 3), padding="same", name="conv2d_12", kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization(name="batch_norm_12")(x)
    x = layers.LeakyReLU(name="leaky_relu_12")(x)
    onetwoeight_out = layers.Conv2D(
        1,
        (3, 3),
        padding="same",
        activation="tanh",
        name="onetwoeight_out",
        kernel_initializer="glorot_uniform",
    )(x)

    x = layers.UpSampling2D((2, 2), name="upsample_6")(x)  # Output: 256x256x512
    x = layers.Conv2D(
        512, (3, 3), padding="same", name="conv2d_13", kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization(name="batch_norm_13")(x)
    x = layers.LeakyReLU(name="leaky_relu_13")(x)
    x = layers.Conv2D(
        512, (3, 3), padding="same", name="conv2d_14", kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization(name="batch_norm_14")(x)
    x = layers.LeakyReLU(name="leaky_relu_14")(x)
    twofivesix_out = layers.Conv2D(
        1,
        (3, 3),
        padding="same",
        activation="tanh",
        name="twofivesix_out",
        kernel_initializer="glorot_uniform",
    )(x)

    x = layers.UpSampling2D((2, 2), name="upsample_7")(x)  # Output: 512x512x512
    x = layers.Conv2D(
        1024, (3, 3), padding="same", name="conv2d_15", kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization(name="batch_norm_15")(x)
    x = layers.LeakyReLU(name="leaky_relu_15")(x)
    x = layers.Conv2D(
        1024, (3, 3), padding="same", name="conv2d_16", kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization(name="batch_norm_16")(x)
    x = layers.LeakyReLU(name="leaky_relu_16")(x)
    fiveonetwo_out = layers.Conv2D(
        1,
        (3, 3),
        padding="same",
        activation="tanh",
        name="fiveonetwo_out",
        kernel_initializer="glorot_uniform",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=four_out, name="Generator_4x4")
    if resolution == 8:
        model = tf.keras.Model(inputs=inputs, outputs=eight_out, name="Generator_8x8")
    if resolution == 16:
        model = tf.keras.Model(
            inputs=inputs, outputs=sixteen_out, name="Generator_16x16"
        )
    if resolution == 32:
        model = tf.keras.Model(
            inputs=inputs, outputs=thirtytwo_out, name="Generator_32x32"
        )
    if resolution == 64:
        model = tf.keras.Model(
            inputs=inputs, outputs=sixtyfour_out, name="Generator_64x64"
        )
    if resolution == 128:
        model = tf.keras.Model(
            inputs=inputs, outputs=onetwoeight_out, name="Generator_128x128"
        )
    if resolution == 256:
        model = tf.keras.Model(
            inputs=inputs, outputs=twofivesix_out, name="Generator_256x256"
        )
    if resolution == 512:
        model = tf.keras.Model(
            inputs=inputs, outputs=fiveonetwo_out, name="Generator_512x512"
        )

    return model


def create_discriminator_model(resolution):
    inputs = tf.keras.Input(shape=(resolution, resolution, 1))
    if resolution == 512:
        x = layers.Conv2D(
            512,
            (5, 5),
            strides=(2, 2),
            padding="same",
            name="conv2d_1",
            kernel_initializer="he_normal",
        )(inputs)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(
            512,
            (3, 3),
            strides=(1, 1),
            padding="same",
            name="conv2d_2",
            kernel_initializer="he_normal",
        )(x)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        inputs = layers.Dropout(0.2)(x)
        # Image shape: 256, 256

    if resolution >= 256:
        x = layers.Conv2D(
            512,
            (5, 5),
            strides=(2, 2),
            padding="same",
            name="conv2d_3",
            kernel_initializer="he_normal",
        )(inputs)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(
            256,
            (3, 3),
            strides=(1, 1),
            padding="same",
            name="conv2d_4",
            kernel_initializer="he_normal",
        )(x)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        inputs = layers.Dropout(0.2)(x)
        # Image shape: 128, 128

    if resolution >= 128:
        x = layers.Conv2D(
            256,
            (5, 5),
            strides=(2, 2),
            padding="same",
            name="conv2d_5",
            kernel_initializer="he_normal",
        )(inputs)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(
            256,
            (3, 3),
            strides=(1, 1),
            padding="same",
            name="conv2d_6",
            kernel_initializer="he_normal",
        )(x)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        inputs = layers.Dropout(0.2)(x)
        # Image shape: 64, 64

    if resolution >= 64:
        x = layers.Conv2D(
            512,
            (5, 5),
            strides=(2, 2),
            padding="same",
            name="conv2d_7",
            kernel_initializer="he_normal",
        )(inputs)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(
            512,
            (3, 3),
            strides=(1, 1),
            padding="same",
            name="conv2d_8",
            kernel_initializer="he_normal",
        )(x)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        inputs = layers.Dropout(0.2)(x)
        # Image shape: 32, 32

    if resolution >= 32:
        x = layers.Conv2D(
            384,
            (5, 5),
            strides=(2, 2),
            padding="same",
            name="conv2d_9",
            kernel_initializer="he_normal",
        )(inputs)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(
            256,
            (3, 3),
            strides=(1, 1),
            padding="same",
            name="conv2d_10",
            kernel_initializer="he_normal",
        )(x)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        inputs = layers.Dropout(0.2)(x)
        # Image shape: 16, 16

    if resolution >= 16:
        x = layers.Conv2D(
            128,
            (5, 5),
            strides=(2, 2),
            padding="same",
            name="conv2d_11",
            kernel_initializer="he_normal",
        )(inputs)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(
            64,
            (3, 3),
            strides=(1, 1),
            padding="same",
            name="conv2d_12",
            kernel_initializer="he_normal",
        )(x)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        inputs = layers.Dropout(0.2)(x)
        # Image shape: 8, 8

    if resolution >= 8:
        x = layers.Conv2D(
            64,
            (5, 5),
            strides=(2, 2),
            padding="same",
            name="conv2d_13",
            kernel_initializer="he_normal",
        )(inputs)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(
            32,
            (3, 3),
            strides=(1, 1),
            padding="same",
            name="conv2d_14",
            kernel_initializer="he_normal",
        )(x)
        x = layers.LeakyReLU()(x)
        x = layers.BatchNormalization()(x)
        inputs = layers.Dropout(0.2)(x)
        # Image shape: 4, 4

    x = layers.Conv2D(
        32,
        (3, 3),
        strides=(1, 1),
        padding="same",
        name="conv2d_15",
        kernel_initializer="he_normal",
    )(inputs)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(
        16,
        (3, 3),
        strides=(1, 1),
        padding="same",
        name="conv2d_16",
        kernel_initializer="he_normal",
    )(x)
    x = layers.LeakyReLU()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, kernel_initializer="glorot_uniform")(x)
    x = layers.LeakyReLU()(x)
    outputs = layers.Dense(1, kernel_initializer="glorot_uniform")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Discriminator")

    return model


def initialize(resolution):
    # models
    generator_model, discriminator_model = create_generator_model(
        resolution
    ), create_discriminator_model(resolution)

    # dataset
    image_paths = tf.data.Dataset.list_files(data_dir + "/*.png")
    dataset = image_paths.map(lambda x: preprocess_image(x, resolution))
    dataset = dataset.batch(BATCH_SIZE)

    # extra
    # generator_models = []
    # discriminator_models = []
    # datasets = []
    # for resolution in resolutions:
    # generator_models.append(generator_model)
    # discriminator_models.append(discriminator_model)
    # datasets.append(dataset)

    return generator_model, discriminator_model, dataset


# loss and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(0.0004)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0001)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


resolutions = [4, 8, 16, 32, 64, 128, 256, 512]
generator, discriminator, dataset = initialize(resolutions[4])
# noise = tf.random.uniform(shape=(4, 512), minval=0.0, maxval=1.0)
# generated_image = generator(noise, training=False)


save_dir = "res1"
weight_dir = "weights"
file_path = "%s/generated_%d.png"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir, exist_ok=True)
if not os.path.isdir(weight_dir):
    os.makedirs(weight_dir, exist_ok=True)

noise_dim = 512
num_examples_to_generate = 16
tf.random.set_seed(42)
seed = tf.random.normal([num_examples_to_generate, noise_dim])


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True) 

        gen_loss = generator_loss(fake_output)*1.2
        disc_loss = discriminator_loss(real_output, fake_output)*0.8

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )
    return gen_loss, disc_loss, tf.reduce_mean(real_output), tf.reduce_mean(fake_output)


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(10, 10))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")

    plt.savefig(save_dir + "/image_at_epoch_{:04d}.png".format(epoch))
    plt.close(fig)


def train(dataset, start, end):
    for epoch in range(start, end):
        start = time.time()
        i = 0
        for image_batch in dataset:
            g_l, d_l, ro, fo = train_step(image_batch)
            print(
                "Step: {} Epoch: {} Dis_Loss: {} Gen_Loss: {} Real_out: {},Fake_out:{}".format(
                    i, epoch + 1, d_l, g_l, ro, fo
                )
            )
            if i % 100 == 0:
                generate_and_save_images(generator, epoch + 1, seed)
                # generated_images = generator(seed, training=False)
                # save_images(image_batch, generated_images, epoch+1)
                generator.save_weights(weight_dir + "/generator64.h5")
                discriminator.save_weights(weight_dir + "/discriminator64.h5")
            i += 1
        generate_and_save_images(generator, epoch + 1, seed)
        generator.save_weights(weight_dir + "/generator64.h5")
        discriminator.save_weights(weight_dir + "/discriminator64.h5")
        print("epoch: {} Time: {} sec".format(epoch + 1, time.time() - start))

        # generate_and_save_images(generator,
        #                          epochs,
        #                          seed)


generator.load_weights(
    weight_dir + "/generator322.h5", by_name=True, skip_mismatch=True
)
discriminator.load_weights(
    weight_dir + "/discriminator322.h5", by_name=True, skip_mismatch=True
)


def get_layer_names(model):
    layer_names = []
    for layer in model.layers:
        layer_names.append(layer.name)
    return layer_names


def freeze_layers(model, layer_names):
    for layer in model.layers:
        if layer.name in layer_names:
            layer.trainable = False


def unfreeze_layers(model, layer_names):
    for layer in model.layers:
        if layer.name in layer_names:
            layer.trainable = True


generator_32x32, discriminator_32x32, dataset_32x32 = initialize(resolutions[3])

gen_layers32x32 = get_layer_names(generator_32x32) 
dis_layers32x32 = get_layer_names(discriminator_32x32)
del generator_32x32
del discriminator_32x32
del dataset_32x32
# freeze_layers(generator, gen_layers32x32)
# freeze_layers(discriminator, dis_layers32x32)
train(dataset, 115, 120)
