## ["An Incremental Voltage Diference Based Technique for Online State of Health Estimation of Li-ion Batteries"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7293255/) **구현코드**





## 학습 파라미터 설정(config.py)

### dataset 바꾸기
``` Python
trainset = [("B0007","Battery_Data_Set"), ("B0005","Battery_Data_Set"), ("B0018","Battery_Data_Set") ] # trainset, testset 설정
testset = [("B0006","Battery_Data_Set")]
```
trainset과 testset의 파일 이름과 data set 변경

### Vsei에 대한 시작점 설정
``` Python
min_soc = 0.2
max_soc = 0.6

soc_margin = 0.03 # soc 간격이 1.5%가 아니라 더 높게해야 잘나옴
```
### 사용할 모델 설정
``` Python
model_pick = 'cnn'
```
### 입력 size 설정
``` Python
input_size = 10
```
### 학습 환경 설정
``` Python
epochs = 200
batch_size = 4
```

### 학습 모델 설정
``` Python
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
```
## 데이터(main.py)
```
ex ) train_data['charge'][0][120]['Voltage_measured'] // state = charge ; 0 = dataset ; 120 = cycle ; Voltage_measured = feature ;
charging feature is Voltage_measured ; Current_measured ; Temperature_measured ; Current_charge ; Voltage_charge ; Time ;
discharging feature is Voltage_measured ; Current_measured ; Temperature_measured ; Current_load ; Voltage_load ; Time ; Capacity ;
```
### 학습 데이터를 랜덤으로 바꾸기
```
'''
s = np.arange(trainX.shape[0])
np.random.shuffle(s)

trainX = trainX[s]
trainY = trainY[s]
'''
```
코드 활성화

## Vsei 특징 추출(vsei_feature.py)
``` Python
def get_vsei_feature(data, dataset_name, is_testset = False):
    dataX = []
    dataY = []
    save_cycle = []
    for i in range(len(dataset_name)):
        cycles = len(data['charge'][i])
        cmax = data['discharge'][i][0]['Capacity']
        rf = get_Rf(data['charge'][i][0])
        for cycle, charge_data, discharge_data in zip(range(cycles),data['charge'][i],data['discharge'][i]):
            vsei_feature = get_Vsei(charge_data, rf)

            dataX.append(vsei_feature)
            dataY.append(discharge_data['Capacity']/cmax * 100)
            save_cycle.append(cycle)
    if is_testset:
        return dataX, dataY, save_cycle
    return dataX, dataY
```

## 이외 각 연산 함수(method.py)
``` Python
def get_capacity(batt_data, discharging = False):
    capacity = 0
    for i,t in zip(batt_data['Current_measured'],batt_data['Time']):
        capacity += i*t
    
    if discharging:
        return -capacity
    else:
        return capacity

def get_Rf(volt_curr_data, init_data = True):
    if init_data:
        tv = 0
        ti = 0
        for v,i in zip(volt_curr_data['Voltage_measured'], volt_curr_data['Current_measured']):
            tv += v
            ti += i
        return tv/ti 
    else: # 안됨
        eod = np.argmin(volt_curr_data['Voltage_measured'])
        volt_eod = volt_curr_data['Voltage_measured'][eod]
        curr_eod = volt_curr_data['Current_measured'][eod]
        sor = np.where(volt_curr_data['Current_measured'] > -0.1)[0]
        for s in sor:
            if s > 5:
                volt_sor = volt_curr_data['Voltage_measured'][s]
                return (volt_eod - volt_sor) / curr_eod
        

def get_SOC(batt_data,capacity):
    soc = [0]
    length = len(batt_data['Current_measured'])
    for k, i, t in zip(range(1,length+1), batt_data['Current_measured'], batt_data['Time']):
        now_soc = soc[k-1] + (i*t / capacity)
        soc.append(now_soc)
    return soc

def get_Vsei(data, r):
    data['Voltage_measured'] = np.array(data['Voltage_measured'])
    # nan 값 수정
    nan_idx = np.argwhere(pd.isnull(data['Voltage_measured']))
    for nidx in nan_idx:
        data['Voltage_measured'][nidx] = (data['Voltage_measured'][nidx-1] + data['Voltage_measured'][nidx+1]) / 2
        data['Current_measured'][nidx] = (data['Current_measured'][nidx-1] + data['Current_measured'][nidx+1]) / 2
        data['Temperature_measured'][nidx] = (data['Temperature_measured'][nidx-1] + data['Temperature_measured'][nidx+1]) / 2

    cap = get_capacity(data)
    soc = get_SOC(data,cap)

    volt_threshold = np.mean(data['Voltage_measured']) * 0.90 # 미정 // Vsei를 시작할 부분
    init_threshold = np.where(data['Voltage_measured'] > volt_threshold)[0]
    vsei = []
    temperature = []
    pre_soc = 0
    
    for i in init_threshold:
        if soc[i] > min_soc and soc[i] < max_soc and soc[i] >= (pre_soc + soc_margin):
            v = data['Voltage_measured'][i]
            curr = data['Current_measured'][i]
            vsei.append(v - r*curr)
            temperature.append(data['Temperature_measured'][i])
            pre_soc = soc[i]

        if len(vsei) > input_size-1:
            feature_set = []
            for j in range(len(vsei)-1):
                feature_set.append(vsei[j+1] -vsei[j])
            feature_set.append(np.mean(temperature))
            return feature_set
    print("error")
    exit()

def get_SOH(cmax,c): # 안됨
    return c/cmax * 100
```


