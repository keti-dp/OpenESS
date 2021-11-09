from keras.models import Sequential
from keras.layers import Dense, Conv1D, LeakyReLU, Flatten
from keras.metrics import mean_squared_error
from keras import backend as K
from keras import optimizers

dir = './data/'

trainset = [("B0007","Battery_Data_Set"), ("B0005","Battery_Data_Set"), ("B0018","Battery_Data_Set") ]
testset = [("B0006","Battery_Data_Set")]

min_soc = 0.2
max_soc = 1

soc_margin = 0.03 # soc 간격이 1.5%가 아니라 더 높게해야 잘나옴

model_pick = 'cnn'
input_size = 10

epochs = 200
batch_size = 4

def neural_model():
    model = Sequential()

    if model_pick == "cnn":
        model.add(Conv1D(filters = 8, kernel_size = 3,input_shape=(input_size,1))) #input_dim = 1
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv1D(filters = 32, kernel_size = 3))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv1D(filters = 16, kernel_size = 3))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dense(1))
    elif model_pick == "ann":
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dense(1))

    optimizer = optimizers.Adam()

    model.compile(loss=root_mean_squared_error,
                    optimizer = optimizer,
                    #metrics = [mean_squared_error] #["mae", "mse", 'mape', 'msle']
                    )
    #model.summary()
    return model

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(mean_squared_error(y_true, y_pred))
