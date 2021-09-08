from method import get_Rf, get_Vsei

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