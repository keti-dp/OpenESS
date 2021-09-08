import scipy.io
from config import dir

def get_matData(dataName_and_set):
    mat_name = dataName_and_set[0]
    dataset = dataName_and_set[1]

    if dataset == "Battery_Data_Set":

        train_data = {}
        train_data['charge'] = []
        train_data['discharge'] = []

        file_name = dir + mat_name + ".mat"
        mat_datas = scipy.io.loadmat(file_name)
        mat_datas = mat_datas[mat_name]['cycle'][0][0][0]

        charge_data = []
        discharge_data = []
        for mat_data in mat_datas:
            if mat_data['type'][0] == 'impedance':
                continue
            else:
                tmp = {}
                for dict in mat_data['data'].dtype.names:
                    tmp[dict] = mat_data['data'][dict][0][0][0] # 데이터 배열이 이상하므로 맞춰줌

                if mat_data['type'][0] == 'charge':
                    charge_data.append(tmp)
                elif mat_data['type'][0] == 'discharge':
                    discharge_data.append(tmp)

        return charge_data, discharge_data

