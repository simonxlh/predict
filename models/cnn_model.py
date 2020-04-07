    
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import Reshape
from keras.layers import MaxPooling1D
from keras.layers import AveragePooling1D
from keras.layers import concatenate
from keras.optimizers import Adam 
models_ens = []
n_ens = 5  # Number of ensembles
ens_list = list(range(n_ens))

for run in ens_list:
    print('Run', run, end=': ')
    models = []
    for gp in gp_list:
        print('Group-', gp, sep='', end=' ')
        layer_0 = Input(shape=(x_length,), name='x_input')
        layer_t = Reshape((-1, 1))(layer_0)
        layer_t = Conv1D(140, kernel_size=3, activation='relu')(layer_t)
        layer_t = AveragePooling1D(pool_size=2)(layer_t)
        layer_cnn_x = Flatten()(layer_t)

        layer_a = Input(shape=(a_length,), name='a_input')

        layer_t = concatenate([layer_cnn_x, layer_a])

        layer_t = Dense(130, activation='relu')(layer_t)
        layer_t = Dropout(0.25)(layer_t)
        layer_t = Dense(120, activation='relu')(layer_t)
        layer_t = Dropout(0.5)(layer_t)
        layer_f = Dense(y_length)(layer_t)

        model = Model(inputs=[layer_0, layer_a], outputs=layer_f)
        model.compile(optimizer='adam',
                        loss='mean_absolute_error', metrics=[k_smape])
        models.append(model)
    models_ens.append(models)
    print('')