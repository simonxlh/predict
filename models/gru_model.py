from keras.layers import GRU, Dense, Input, Lambda, Reshape, Conv1D, Activation
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.models import Model
import keras.backend as K

ESTIMATES_COUNT = 7
TRUNCATED_SEQUENCE_LENGTH = 150
PREDICTION_LENGTH = 60

def smape_loss(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true) + K.abs(y_pred), K.epsilon(), None))
    return 200 * K.mean(diff, axis=-1)

def combine_estimates(data):
    estimates, coefs = data
    return K.sum(estimates * coefs, axis=-1)

# feed visit counts during latest TRUNCATED_SEQUENCE_LENGTH days here
raw_data = Input(shape=(TRUNCATED_SEQUENCE_LENGTH,))
expanded = Reshape((TRUNCATED_SEQUENCE_LENGTH, 1))(raw_data)
conv_result = Activation('relu')(concatenate([
    Conv1D(32, 3, padding='same')(expanded),
    Conv1D(32, 3, dilation_rate=7, padding='same')(expanded),
]))
conv_result = Activation('relu')(concatenate([
    Conv1D(32, 3, padding='same')(conv_result),
    Conv1D(32, 3, dilation_rate=7, padding='same')(conv_result),
]))
conv_result = Activation('relu')(concatenate([
    Conv1D(32, 3, padding='same')(conv_result),
    Conv1D(32, 3, dilation_rate=7, padding='same')(conv_result),
]))
conv_result = Conv1D(64, 3)(conv_result)

rnn_result = GRU(256)(conv_result)

rnn_result = Dense(PREDICTION_LENGTH * ESTIMATES_COUNT, activation='relu')(rnn_result)
coefs = Reshape((PREDICTION_LENGTH, ESTIMATES_COUNT))(rnn_result)

# estimates_input[i] is expected to contain features (like median of visits during
# same weekday) that will be combined to predict visit counts for i-th day
estimates_input = Input(shape=(PREDICTION_LENGTH, ESTIMATES_COUNT))
    
result = Lambda(combine_estimates)([estimates_input, coefs])

model = Model(inputs=[raw_data, estimates_input], outputs=result)
model.compile(optimizer=Adam(lr=1e-3), loss=smape_loss)