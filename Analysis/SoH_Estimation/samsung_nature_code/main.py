import matplotlib.pyplot as plt
from config import *
from method import *
from dataset import *
from vsei_feature import get_vsei_feature
import tensorflow as tf

if __name__ =="__main__":
    train_data = {}
    train_data['charge'] = []
    train_data['discharge'] = []

    for td in trainset:
        charge_data, discharge_data = get_matData(td) 
        train_data['charge'].append(charge_data)
        train_data['discharge'].append(discharge_data)
    
    test_data = {}
    test_data['charge'] = []
    test_data['discharge'] = []
    for td in testset:
        charge_data, discharge_data = get_matData(td) 
        test_data['charge'].append(charge_data)
        test_data['discharge'].append(discharge_data)

    # ex ) train_data['charge'][0][120]['Voltage_measured'] // state = charge ; 0 = dataset ; 120 = cycle ; Voltage_measured = feature ;
    # charging feature is Voltage_measured ; Current_measured ; Temperature_measured ; Current_charge ; Voltage_charge ; Time ;
    # discharging feature is Voltage_measured ; Current_measured ; Temperature_measured ; Current_load ; Voltage_load ; Time ; Capacity ;

    
    trainX, trainY = get_vsei_feature(train_data,trainset)
    testX, testY, testCycles = get_vsei_feature(test_data,testset, is_testset= True)

    trainX = np.array(trainX); trainY = np.array(trainY)
    testX = np.array(testX); testY = np.array(testY)
    
    '''
    s = np.arange(trainX.shape[0])
    np.random.shuffle(s)

    trainX = trainX[s]
    trainY = trainY[s]
    '''
    print(trainX.shape)
    trainX = np.expand_dims(trainX,axis=3)
    testX = np.expand_dims(testX,axis=3)

    model = neural_model()
    
    history = model.fit(trainX,trainY, epochs=epochs, batch_size= batch_size, validation_data = (testX,testY))

    predict_result = model.predict(testX)
    #loss, mae, mse = model.evaluate(testX, testY)
    testCycles = np.array(testCycles)

    plt.figure()
    plt.plot(testCycles,testY, label = "original")
    plt.plot(testCycles,predict_result, label = "predict")
    plt.legend()
    plt.savefig("./result.jpg")
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'][20:], label = 'original_loss')
    plt.plot(history.history['val_loss'][20:], label = 'validation_loss')
    plt.legend()
    plt.savefig("./result_loss.jpg")
    plt.show()