from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from src.utils import r2_metric
from keras.callbacks import EarlyStopping


def create_mlp_model(n_inputs, n_outputs, dropout_rate = 0.2, hidden_layer_neurons = ()):
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


def compile_mlp_model(model, learning_rate = 0.001):
    return model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mae', r2_metric]
    )


def train_mlp_model(model, X_train, y_train, X_validate, y_validate):
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data = (X_validate, y_validate),
        epochs = 100,
        batch_size = 32,
        callbacks = [early_stop]
    )

    return history