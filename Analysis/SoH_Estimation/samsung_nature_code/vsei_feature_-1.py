from method import get_Rf, get_Vsei

def get_vsei_feature(data, dataset_name, is_testset = False):
    dataX = []
    dataY = []
    save_cycle = []
    #print(data['charge'])
    for i in range(len(dataset_name)):
        cycles = len(data['charge'][i])
        cmax = data['discharge'][i][0]['Capacity']
        #cmax = get_capacity(train_data['discharge'][i][0],discharging = True)
        rf = get_Rf(data['charge'][i][0])
        for cycle, charge_data, discharge_data in zip(range(cycles),data['charge'][i],data['discharge'][i]):
            #print(cycle)
            #rf = get_Rf(charge_data) if cycle == 0 else get_Rf(discharge_data,init_data = False)
            vsei_feature = get_Vsei(charge_data, rf)

            dataX.append(vsei_feature)
            dataY.append(discharge_data['Capacity']/cmax * 100)
            save_cycle.append(cycle)
            #trainY[i].append(get_SOH(cmax,get_capacity(charge_data))) # is soh
    if is_testset:
        return dataX, dataY, save_cycle
    return dataX, dataY