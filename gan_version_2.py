import numpy as np
import csv
import random
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LeakyReLU, BatchNormalization, Dropout
from keras.optimizers.legacy import Adam
from sklearn.model_selection import train_test_split
import pandas as pd

# Manual weight initialization
def set_weights(layer, weights):
    if isinstance(layer, Dense):
        layer.set_weights(weights)

# Seed initialization
seeds = [random.randint(1, 10000000) for i in range(10)]

results = []

for seed in seeds:
    random.seed(seed)
    np.random.seed(seed)

    # Length of the binary sequences
    seq_length = 200
    latent_dim = 100  # Dimension of the generator's input noise vector
    learning_rate = 0.00005
    beta_1 = 0.5

    # Generator model
    def build_generator():
        model = Sequential()
        model.add(Dense(1024, input_dim=latent_dim))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Dense(2048))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Dense(seq_length, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    # Enhanced Discriminator
    def build_discriminator():
        model = Sequential()
        model.add(Dense(1024, input_dim=seq_length))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))
        model.add(Dense(5, activation='softmax'))  # 5 classes for 5 labels
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=Adam(learning_rate, beta_1),
                      metrics=['accuracy'])
        return model

    # Define manual weights (example weights, you can choose different ones)
    def get_manual_weights(shape):
        if len(shape) == 2:  # For Dense layer weights
            stddev = np.sqrt(2 / shape[0])
            return [np.random.normal(0, stddev, size=shape), np.zeros(shape[1])]
        elif len(shape) == 1:  # For biases
            return [np.zeros(shape)]
        else:
            raise ValueError(
                "Unsupported shape for manual weight initialization")

    # Set weights for the models
    def set_weights(layer, weights):
        if isinstance(layer, Dense):
            layer.set_weights(weights)

    # GAN model combining generator and discriminator
    def build_gan(generator, discriminator):
        discriminator.trainable = False
        gan_input = Input(shape=(latent_dim,))
        x = generator(gan_input)
        gan_output = discriminator(x)
        gan = Model(gan_input, gan_output)
        gan.compile(loss='sparse_categorical_crossentropy',
                    optimizer=Adam(learning_rate, beta_1))
        return gan

    # Generate sequences using the trained generator
    def generate_sequences(generator, num_sequences=64):
        noise = np.random.randn(num_sequences, latent_dim)
        generated_sequences = generator.predict(noise)
        return (generated_sequences > 0.5).astype(int)

    # Training loop
    def train_gan(handfake_train, tricky_train, mom_train, epochs=500, batch_size=64, d_steps=1, g_steps=1):
        """
        Train the GAN.

        Parameters:
            epochs (int): Number of epochs to train.
            batch_size (int): Number of samples per batch.
            d_steps (int): Number of discriminator training steps per generator training step.
            g_steps (int): Number of generator training steps per batch (typically 1).
            handfake_train (np.array): Preloaded handfake sequences for training.
            tricky_train (np.array): Preloaded tricky sequences for training.
            mom_train (np.array): Preloaded MOM sequences for training.
        """
        handfake_train = np.array(handfake_train)
        tricky_train = np.array(tricky_train)
        mom_train = np.array(mom_train)

        num_handfake = handfake_train.shape[0]
        num_tricky = tricky_train.shape[0]
        num_mom = mom_train.shape[0]

        for epoch in range(epochs):
            # Train discriminator more frequently
            for _ in range(d_steps):
                # Generate real sequences
                real_sequences = np.random.randint(0, 2, size=(
                batch_size, seq_length))

                # Generate fake sequences
                noise = np.random.randn(batch_size, latent_dim)
                generated_sequences = generator.predict(noise)

                # Randomly sample a batch from the preloaded sequences
                handfake_indices = np.random.randint(0, num_handfake,
                                                     batch_size)
                tricky_indices = np.random.randint(0, num_tricky, batch_size)
                mom_indices = np.random.randint(0, num_mom, batch_size)

                handfake_sequences = handfake_train[handfake_indices]
                tricky_sequences = tricky_train[tricky_indices]
                mom_sequences = mom_train[mom_indices]

                # Define labels for each type of sequence
                real_labels = np.zeros(batch_size,
                                       dtype=int)  # Label 0 for real sequences
                gan_labels = np.ones(batch_size,
                                     dtype=int)  # Label 1 for GAN sequences
                mom_labels = np.full(batch_size, 2,
                                     dtype=int)  # Label 2 for MOM sequences
                handfake_labels = np.full(batch_size, 3,
                                          dtype=int)  # Label 3 for handfake sequences
                tricky_labels = np.full(batch_size, 4,
                                        dtype=int)  # Label 4 for tricky sequences

                # Flatten labels to be of shape (batch_size,)
                real_labels = real_labels.flatten()
                gan_labels = gan_labels.flatten()
                mom_labels = mom_labels.flatten()
                handfake_labels = handfake_labels.flatten()
                tricky_labels = tricky_labels.flatten()

                # Train discriminator on real sequences
                d_loss_real = discriminator.train_on_batch(real_sequences,
                                                           real_labels)

                # Train discriminator on fake sequences generated by the GAN
                d_loss_fake = discriminator.train_on_batch(generated_sequences,
                                                           gan_labels)

                # Train discriminator on MOM sequences
                d_loss_mom = discriminator.train_on_batch(mom_sequences,
                                                          mom_labels)

                # Train discriminator on handfake sequences
                d_loss_handfake = discriminator.train_on_batch(
                    handfake_sequences, handfake_labels)

                # Train discriminator on tricky sequences
                d_loss_tricky = discriminator.train_on_batch(tricky_sequences,
                                                             tricky_labels)

                # Average discriminator loss
                d_loss = 0.2 * np.add(
                    np.add(np.add(np.add(d_loss_real, d_loss_fake), d_loss_mom),
                           d_loss_handfake), d_loss_tricky)

            # Train the generator
            for _ in range(g_steps):
                noise = np.random.randn(batch_size, latent_dim)
                y_gen = np.zeros(batch_size,
                                 dtype=int)  # Target labels for generator (0, meaning real for generator)
                g_loss = gan.train_on_batch(noise, y_gen)

            # Print progress
            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch} | D loss: {d_loss[0]} | G loss: {g_loss} | D accuracy: {100 * d_loss[1]}")

    # Build and compile the models
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator,
                    discriminator)  # This line should define and compile `gan`

    def read_csv_as_array(name):
        data_frame = pd.read_csv(name, header=None)
        data_array = data_frame.values
        return data_array

    hand_fakes = read_csv_as_array('Handwritten_seq.csv').tolist()
    hand_fakes.pop(0)
    print('No. hand fakes: ' + str(len(hand_fakes)))

    tricky_fakes = read_csv_as_array('Tricky_seq.csv').tolist()
    print('No. tricky fakes: ' + str(len(tricky_fakes)))

    dmom_fakes = read_csv_as_array('DMOM_seq.csv').tolist()
    print('No. gan fakes: ' + str(len(dmom_fakes)))

    # Initialize an empty list to store the data
    reals = [[random.randint(0, 1) for i in range(200)] for j in range(4 * len(hand_fakes))]

    # Take 80% of each type of sequence for training and leaving rest for testing
    train_ratio = 0.8

    reals_training, reals_testing = train_test_split(reals,
                                                     train_size=train_ratio,
                                                     random_state=42)
    print('No. reals training: ' + str(len(reals_training)))
    print('No. reals testing: ' + str(len(reals_testing)) + '\n')

    hand_fakes_training, hand_fakes_testing = train_test_split(hand_fakes,
                                                               train_size=train_ratio,
                                                               random_state=43)
    print('No. hand fakes training: ' + str(len(hand_fakes_training)))
    print('No. hand fakes testing: ' + str(len(hand_fakes_testing)) + '\n')

    tricky_fakes_training, tricky_fakes_testing = train_test_split(tricky_fakes,
                                                                   train_size=train_ratio,
                                                                   random_state=44)
    print('No. tricky fakes training: ' + str(len(tricky_fakes_training)))
    print('No. tricky fakes testing: ' + str(len(tricky_fakes_testing)) + '\n')

    mom_fakes_training, mom_fakes_testing = train_test_split(dmom_fakes,
                                                             train_size=train_ratio,
                                                             random_state=45)
    print('No. dmom fakes training: ' + str(len(mom_fakes_training)))
    print('No. dmom fakes testing: ' + str(len(mom_fakes_testing)) + '\n')

    # Training
    train_gan(handfake_train=hand_fakes_training,
              tricky_train=tricky_fakes_training, mom_train=mom_fakes_training,
              epochs=500, batch_size=64)

    generated_sequences = generate_sequences(generator,
                                             num_sequences=137).tolist()
    for i in generated_sequences:
        print(len(i), i)

    for i in range(1, len(generated_sequences)):
        print(generated_sequences[i] == generated_sequences[i - 1])

    file_name = 'GAN_seq.csv'

    # Writing to CSV
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(generated_sequences)

    gan_fakes_training, gan_fakes_testing = train_test_split(
        generated_sequences, train_size=train_ratio, random_state=46)
    print('No. GAN fakes training: ' + str(len(gan_fakes_training)))
    print('No. GAN fakes testing: ' + str(len(gan_fakes_testing)) + '\n')

    # Testing
    test_sequences = reals_testing + gan_fakes_testing + mom_fakes_testing + hand_fakes_testing + tricky_fakes_testing
    test_labels = np.concatenate([
        np.zeros(len(reals_testing)),  # Real sequences (label 0)
        np.ones(len(gan_fakes_testing)),  # GAN sequences (label 1)
        np.full(len(mom_fakes_testing), 2),  # MOM sequences (label 2)
        np.full(len(hand_fakes_testing), 3),  # Handfake sequences (label 3)
        np.full(len(tricky_fakes_testing), 4)  # Tricky sequences (label 4)
    ])

    # Predict and calculate accuracy for each label
    predictions = np.argmax(discriminator.predict(np.array(test_sequences)),
                            axis=1)

    real_accuracy = np.sum(predictions[:len(reals_testing)] == 0) / len(
        reals_testing)
    gan_accuracy = np.sum(predictions[
                          len(reals_testing):len(reals_testing) + len(
                              gan_fakes_testing)] == 1) / len(gan_fakes_testing)
    mom_accuracy = np.sum(predictions[
                          len(reals_testing) + len(gan_fakes_testing):len(
                              reals_testing) + len(gan_fakes_testing) + len(
                              mom_fakes_testing)] == 2) / len(mom_fakes_testing)
    handfake_accuracy = np.sum(predictions[len(reals_testing) + len(
        gan_fakes_testing) + len(mom_fakes_testing):len(reals_testing) + len(
        gan_fakes_testing) + len(mom_fakes_testing) + len(
        hand_fakes_testing)] == 3) / len(hand_fakes_testing)
    tricky_accuracy = np.sum(predictions[
                             len(reals_testing) + len(gan_fakes_testing) + len(
                                 mom_fakes_testing) + len(
                                 hand_fakes_testing):] == 4) / len(
        tricky_fakes_testing)

    overall_accuracy = np.mean(predictions == test_labels)

    print(f"Real accuracy: {real_accuracy}")
    print(f"GAN accuracy: {gan_accuracy}")
    print(f"MOM accuracy: {mom_accuracy}")
    print(f"Handfake accuracy: {handfake_accuracy}")
    print(f"Tricky accuracy: {tricky_accuracy}")
    print(f"Overall accuracy: {overall_accuracy}")

    results.append(
        [seed, real_accuracy, gan_accuracy, mom_accuracy, handfake_accuracy,
         tricky_accuracy, overall_accuracy])

    quit()

# Calculate the average accuracy over all seeds
average_accuracy = np.mean([result[3] for result in results])
print(f"Average accuracy over all seeds: {average_accuracy}")
