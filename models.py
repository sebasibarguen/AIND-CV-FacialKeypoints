import json

import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax

from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint

from utils import *

# Load training set
X_train, y_train = load_data()
print("X_train.shape == {}".format(X_train.shape))
print("y_train.shape == {}; y_train.min == {:.3f}; y_train.max == {:.3f}".format(
    y_train.shape, y_train.min(), y_train.max()))

# Load testing set
X_test, _ = load_data(test=True)
print("X_test.shape == {}".format(X_test.shape))


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def train_model(model, model_name):
    print('Going to train:', model_name)

    dir_path = './checkpoints/{}/'.format(model_name)
    filepath = dir_path + '{epoch:02d}-{val_loss:.4f}.hdf5'

    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1)
    tensorboard = keras.callbacks.TensorBoard(log_dir='./checkpoints/logs')

    adagrad = Adagrad(lr=0.01, decay=1e-6)
    model.compile(
        loss='mean_squared_error',
        optimizer=adagrad,
        metrics=['acc', 'mae']
    )

    hist = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        batch_size=32,
        epochs=20,
        callbacks=[checkpointer, tensorboard])

    model.save('./checkpoints/{}_model.h5'.format(model_name))

    with open('{}{}_model-metrics.json'.format(dir_path, model_name), 'w') as f:
        json_metrics = json.dumps(hist.history)
        f.write(json_metrics)

    return hist


# ************** Model 1 ******************** #
model_name = 'conv1_dense1'

conv1_dense1 = Sequential()
conv1_dense1.add(
    Convolution2D(32, (3, 3), activation='relu', input_shape=(96, 96, 1))
)
conv1_dense1.add(Flatten())
conv1_dense1.add(Dense(30))

adagrad = Adagrad(lr=0.01, decay=1e-6)
conv1_dense1.compile(loss='mean_squared_error', optimizer=adagrad)

# conv1_dense1_result = train_model(conv1_dense1, model_name)

# ************** Model 2 ******************** #
model_name = 'conv2_dense1'

conv2_dense1 = Sequential()
conv2_dense1.add(
    Convolution2D(32, (3, 3), activation='relu', input_shape=(96, 96, 1))
)
conv2_dense1.add(Convolution2D(32, (3, 3), activation='relu'))
conv2_dense1.add(Flatten())
conv2_dense1.add(Dense(30))

# conv2_dense1_result = train_model(conv2_dense1, model_name)

# ************** Model 3 ******************** #

model_name = 'conv3_dense1'

conv3_dense1 = Sequential()
conv3_dense1.add(
    Convolution2D(32, (3, 3), activation='relu', input_shape=(96, 96, 1))
)
conv3_dense1.add(Convolution2D(32, (3, 3), activation='relu'))
conv3_dense1.add(Convolution2D(32, (3, 3), activation='relu'))
conv3_dense1.add(Flatten())
conv3_dense1.add(Dense(30))

conv3_dense1_result = train_model(conv3_dense1, model_name)

# ************** Model 4 ******************** #

model_name = 'conv4_dense1'

conv4_dense1 = Sequential()
conv4_dense1.add(
    Convolution2D(32, (3, 3), activation='relu', input_shape=(96, 96, 1))
)
conv4_dense1.add(Convolution2D(32, (3, 3), activation='relu'))
conv4_dense1.add(Convolution2D(32, (3, 3), activation='relu'))
conv4_dense1.add(Convolution2D(32, (3, 3), activation='relu'))
conv4_dense1.add(Flatten())
conv4_dense1.add(Dense(30))

# conv4_dense1_result = train_model(conv4_dense1, model_name)


# ************** Model 4 ******************** #

# base_model = Sequential()
# base_model.add(
#     Convolution2D(32, (3, 3), activation='relu', input_shape=(96, 96, 1))
# )
# base_model.add(MaxPooling2D(pool_size=(2, 2)))
# base_model.add(Convolution2D(64, (3, 3), activation='relu'))
# base_model.add(MaxPooling2D(pool_size=(2, 2)))
# base_model.add(Convolution2D(128, (3, 3), activation='relu'))
# base_model.add(MaxPooling2D(pool_size=(2, 2)))
#
# base_model.add(Flatten())
# base_model.add(Dense(500, activation='relu'))
# base_model.add(Dropout(0.5))
# base_model.add(Dense(100, activation='relu'))
# base_model.add(Dropout(0.5))
#
# base_model.add(Dense(30))
# # Summarize the model
# base_model.summary()
