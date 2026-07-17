from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


def create_model(n_inputs, n_outputs, dropout_rate, hidden_layer_neurons = None):
    model = Sequential()
    if(len(hidden_layer_neurons) > 0):
        model.add(Dense(hidden_layer_neurons[0], activation='relu', input_shape=(n_inputs,)))
        model.add(Dropout(dropout_rate))
        for i in range(1, len(hidden_layer_neurons)):
            model.add(Dense(hidden_layer_neurons[i], activation='relu'))
            model.add(Dropout(dropout_rate))
        
        model.add(Dense(n_outputs, activation='linear'))

    else:
        model.add(Dense(n_outputs, activation='linear', input_shape=(n_inputs,)))
    
    return model