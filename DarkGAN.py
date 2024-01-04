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
data_dir = "/DATA/summ_intern_1/Female"

BATCH_SIZE = 18


def preprocess_image(image_path, resolution):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, [resolution, resolution])
    image = (image - 127.5) / 127.5
    return image


def create_encoder_decoder_model():
    input_shape = (68*2, 1)
    latent_dim = 32

    # Encoder
    encoder_inputs = tf.keras.Input(shape=input_shape, name="encoder_input")

    x = layers.Dense(128, activation="relu", kernel_initializer="glorot_uniform", name="encoder_dense_1")(encoder_inputs)
    x = layers.Dense(64, activation="relu", kernel_initializer="glorot_uniform", name="encoder_dense_2")(x)
    x = layers.Dense(32, activation="relu", kernel_initializer="glorot_uniform", name="encoder_dense_3")(x)

    encoder_outputs = x

    # Decoder
    decoder_inputs = tf.keras.Input(shape=(latent_dim,), name="decoder_input")

    x = layers.Dense(64, activation="relu", kernel_initializer="glorot_uniform", name="decoder_dense_1")(decoder_inputs)
    x = layers.Dense(128, activation="relu", kernel_initializer="glorot_uniform", name="decoder_dense_2")(x)
    x = layers.Dense(68*2, activation="linear", kernel_initializer="glorot_uniform", name="decoder_dense_3")(x)

    decoder_outputs = x

    # Model
    encoder_decoder = tf.keras.Model(inputs=encoder_inputs, outputs=[decoder_outputs, encoder_outputs], name="EncoderDecoderModel")

    return encoder_decoder

feature_size=32


def create_generator_model(resolution):
    noise_inputs = tf.keras.Input(shape=(z_size,))
    feature_inputs = tf.keras.Input(shape=(feature_size,))
    x = layers.BatchNormalization(name="batch_norm_0")(noise_inputs)
    x = layers.Dense(2 * 128, name="dense_1", activation="relu", kernel_initializer="glorot_uniform")(x)
    x = layers.Dense(2 * 128, name="dense_2", activation="relu", kernel_initializer="glorot_uniform")(x)
    x = layers.Dense(2 * 128, name="dense_3", activation="relu", kernel_initializer="glorot_uniform")(x)
    x = layers.Dense(4 * 128, name="dense_4", activation="relu", kernel_initializer="glorot_uniform")(x)
    x = layers.Dense(4 * 128, name="dense_5", activation="relu", kernel_initializer="glorot_uniform")(x)
    x = layers.Dense(8 * 128, name="dense_6", activation="relu", kernel_initializer="glorot_uniform")(x)
    x = layers.Dense(8 * 128, name="dense_7", activation="relu", kernel_initializer="glorot_uniform")(x)
    x = layers.Dense(4 * 4 * 128,name="dense_8",activation="relu",kernel_initializer="glorot_uniform")(x)

    x = layers.Reshape((4, 4, 128), name="reshape")(x)
    #LANDMARK POINTS KE FEATURE CONCATENATE

    # 4x4
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv2d", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization(name="batch_norm_2")(x)
    x = layers.LeakyReLU(name="leaky_relu_1")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv2d_1", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization(name="batch_norm_3")(x)
    x = layers.LeakyReLU(name="leaky_relu_2")(x)

    # 8x8
    x = layers.UpSampling2D((2, 2), name="upsample_1")(x)
    x = layers.BatchNormalization(name="batch_norm_4")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv2d_2", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization(name="batch_norm_5")(x)
    x = layers.LeakyReLU(name="leaky_relu_3")(x)
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv2d_3", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization(name="batch_norm_6")(x)
    x = layers.LeakyReLU(name="leaky_relu_4")(x)

    # 16x16
    x = layers.UpSampling2D((2, 2), name="upsample_2")(x)
    x = layers.BatchNormalization(name="batch_norm_7")(x)
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv2d_4", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization(name="batch_norm_8")(x)
    x = layers.LeakyReLU(name="leaky_relu_5")(x)
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv2d_5", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization(name="batch_norm_9")(x)
    x = layers.LeakyReLU(name="leaky_relu_6")(x)

    # 32X32
    x = layers.UpSampling2D((2, 2), name="upsample_3")(x)
    x = layers.BatchNormalization(name="batch_norm_10")(x)
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv2d_6", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization(name="batch_norm_11")(x)
    x = layers.LeakyReLU(name="leaky_relu_7")(x)
    x = layers.Conv2D(512, (3, 3), padding="same", name="conv2d_7", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization(name="batch_norm_12")(x)
    x = layers.LeakyReLU(name="leaky_relu_8")(x)

    # 64x64
    x = layers.UpSampling2D((2, 2), name="upsample_4")(x)
    x = layers.BatchNormalization(name="batch_norm_13")(x)
    x = layers.Conv2D(512, (3, 3), padding="same", name="conv2d_8", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization(name="batch_norm_14")(x)
    x = layers.LeakyReLU(name="leaky_relu_9")(x)
    x = layers.Conv2D(512, (3, 3), padding="same", name="conv2d_9", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization(name="batch_norm_15")(x)
    x = layers.LeakyReLU(name="leaky_relu_10")(x)
    sixtyfour_out = layers.Conv2D(1,(3, 3),padding="same",activation="tanh",name="sixtyfour_out",kernel_initializer="glorot_uniform")(x)

    x1 = layers.UpSampling2D((2, 2), interpolation="nearest", name="upsample_5")(x)
    x1 = layers.Conv2D(256, (3, 3), padding="same", name="conv2d_11", kernel_initializer="he_normal")(x1)
    x1 = layers.LeakyReLU(name="leaky_relu_11")(x1)
    x1 = layers.BatchNormalization(name="batch_norm_16")(x1)

    x2 = layers.UpSampling2D((2, 2), interpolation="bilinear", name="upsample_6")(x)
    x2 = layers.Conv2D(256, (3, 3), padding="same", name="conv2d_12", kernel_initializer="he_normal")(x2)
    x2 = layers.LeakyReLU(name="leaky_relu_12")(x2)
    x2 = layers.BatchNormalization(name="batch_norm_17")(x2)

    x = layers.Concatenate()([x1, x2])

    x = layers.Conv2D(128, (3, 3), padding="same", name="conv2d_13", kernel_initializer="he_normal")(x)
    x = layers.LeakyReLU(name="leaky_relu_13")(x)
    x = layers.BatchNormalization(name="batch_norm_18")(x)

    onetwoeight_out=layers.Conv2D(1,(3, 3),padding="same",activation="tanh",name="onetwoeight_out",kernel_initializer="glorot_uniform")(x)



    model = tf.keras.Model(inputs=noise_inputs, outputs=onetwoeight_out, name="Generator_128x128")
    return model


def create_discriminator_model(resolution):
    inputs = tf.keras.Input(shape=(resolution, resolution, 1))
    # Image shape: 128, 128
    x = layers.Conv2D(384,(3, 3),strides=(2, 2),padding="same",name="conv2d_8",kernel_initializer="he_normal")(inputs)
    x = layers.LeakyReLU(name="leaky_relu_9")(x)
    x = layers.BatchNormalization(name="batch_norm_9")(x)
    x = layers.Dropout(0.3)(x)
    
    #Image shape: 64, 64
    x = layers.Conv2D(384,(5, 5),strides=(2, 2),padding="same",name="conv2d_7",kernel_initializer="he_normal")(x)
    x = layers.LeakyReLU(name="leaky_relu_8")(x)
    x = layers.BatchNormalization(name="batch_norm_8")(x)
    x = layers.Dropout(0.3)(x)
    # Image shape: 32, 32


    x = layers.Conv2D(256,(5, 5),strides=(2, 2),padding="same",name="conv2d_9",kernel_initializer="he_normal")(x)
    x = layers.LeakyReLU(name="leaky_relu_10")(x)
    x = layers.BatchNormalization(name="batch_norm_10")(x)
    x = layers.Dropout(0.3)(x)
    # Image shape: 16, 16

    x = layers.Conv2D(128,(5, 5),strides=(2, 2),padding="same",name="conv2d_11",kernel_initializer="he_normal")(x)
    x = layers.LeakyReLU(name="leaky_relu_12")(x)
    x = layers.BatchNormalization(name="batch_norm_12")(x)
    x = layers.Dropout(0.3)(x)

    # Image shape: 8, 8
    x = layers.Conv2D(64,(5, 5),strides=(2, 2),padding="same",name="conv2d_13",kernel_initializer="he_normal")(x)
    x = layers.LeakyReLU(name="leaky_relu_14")(x)
    x = layers.BatchNormalization(name="batch_norm_14")(x)
    x = layers.Dropout(0.3)(x)

    # Image shape: 4, 4

    x = layers.Conv2D(32,(3, 3),strides=(1, 1), padding="same",name="conv2d_15",kernel_initializer="he_normal")(x)
    x = layers.LeakyReLU(name="leaky_relu_15")(x)
    x = layers.BatchNormalization(name="batch_norm_15")(x)
    x = layers.Conv2D(16,(3, 3),strides=(1, 1),padding="same",name="conv2d_16",kernel_initializer="he_normal")(x)
    x = layers.LeakyReLU(name="leaky_relu_16")(x)
    x = layers.BatchNormalization(name="batch_norm_16")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, name="Dense_1",kernel_initializer="glorot_uniform")(x)
    x = layers.LeakyReLU(name="leaky_relu_17")(x)

    x = layers.Dense(256, name="Dense_2",kernel_initializer="glorot_uniform")(x)
    x = layers.LeakyReLU(name="leaky_relu_18")(x)
    outputs = layers.Dense(1, name="Dense_3", kernel_initializer="glorot_uniform")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Discriminator")

    return model

def initialize(resolution):
    # models
    generator_model, discriminator_model = create_generator_model(
        resolution
    ), create_discriminator_model(resolution)

    # dataset
    image_paths = tf.data.Dataset.list_files(data_dir + "/*.jpg")
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
generator_optimizer = tf.keras.optimizers.Adam(0.0005)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0001)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


resolution = [4, 8, 16, 32, 64, 128, 256, 512]
generator, discriminator, dataset = initialize(resolution[5])
# noise = tf.random.uniform(shape=(4, 512), minval=0.0, maxval=1.0)
# generated_image = generator(noise, training=False)


save_dir = "res4"
weight_dir = "/DATA/summ_intern_1/weights3"

if not os.path.isdir(save_dir):
    os.makedirs(save_dir, exist_ok=True)
if not os.path.isdir(weight_dir):
    os.makedirs(weight_dir, exist_ok=True)

noise_dim = 512
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
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
            i += 1
            # if i % 100 == 0:
            # generate_and_save_images(generator, epoch + 1, seed)
            # generated_images = generator(seed, training=False)
            # save_images(image_batch, generated_images, epoch+1)
            # generator.save_weights(weight_dir + "/gen4.h5")
            # discriminator.save_weights(weight_dir + "/dis4.h5")
        generate_and_save_images(generator, epoch + 1, seed)
        generator.save_weights(weight_dir + "/gen1286.h5")
        discriminator.save_weights(weight_dir + "/dis1286.h5")
        print("epoch: {} Time: {} sec".format(epoch + 1, time.time() - start))



generator.load_weights(weight_dir+"/gen1285.h5",
                       by_name=True,skip_mismatch=True)
discriminator.load_weights(weight_dir+"/dis1285.h5",
                           by_name=True,skip_mismatch=True)

# train(dataset, 570, 580)

#generator.save_model(weight_dir+"/gen_mode1281.h5")
#discriminator.save_model(weight_dir+"/dis_mode1281.h5")