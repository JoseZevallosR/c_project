from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential


class FeedForward:

    def __init__(self, input_size=None, output_size=None, opt=None, opt_param=None):
        self._model = Sequential()
        self.opt = opt
        self.opt_param = opt_param
        self.input_size = input_size
        self.output_size = output_size
        self._build()

    def _build(self):
        self._model.add(Dense(units=64, input_dim=self.input_size, activation="relu"))
        self._model.add(Dense(units=32, activation="relu"))
        self._model.add(Dense(units=8, activation="relu"))
        self._model.add(Dense(self.output_size, activation="linear"))
        self._model.compile(loss="mse", optimizer=self.opt(**self.opt_param))

    def predict(self, args):
        return self._model.predict(*args)

    @property
    def model(self):
        return self._model

    @property
    def summary(self):
        return self._model.summary()
