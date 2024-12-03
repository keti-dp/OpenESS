# 판리 2호 Bank 1(Rack 9개, Rack 1개당 Tray 20S, Tray 1개당 12S60P --> Rack 1개당 240S60P) / 
# Bank 2(Rack 8개, Rack 1개당 Tray 20S, Tray 1개당 12S20P --> Rack 1개당 240S60P)
# 컨테이너(Bank 1, 2 포함) : 3,320kWh

# 백마 Bank 1(Rack 9개, Rack 1개당 Tray 20S, Tray 1개당 12S60P --> Rack 1개당 240S60P)
# 컨테이너(Bank 1만) : 1,476kWh

# 황금 Bank 1(Rack 11개, Rack 1개당 Tray 17S, Tray 1개당 12S60P --> Rack 1개당 204*60P)
# 컨테이너(Bank 1만) : 1,826kWh

# 판리 2호(Bank 1의 Rack 한개) 초기 용량 : 3,320kWh * 9/17 = 1757.647hWh --> Rack 1개(1757.647hWh/9 = 195.294kWh) 
# --> 195.294kWh/(3.56 * 240) = 228.57Ah
# 백마(Bank 1의 Rack 한개) 초기 용량 : 1,476kWh --> Rack 1개(1,476kWh/9 = 164kWh) --> 164kWh/(3.56 * 240) = 191.95Ah
# 황금(Bank 1의 Rack 한개) 초기 용량 : 1,826kWh --> Rack 1개(1,826kWh/11 = 166kWh) --> 166kWh/(3.56 * 204) = 228.57Ah

# Ri, Rdiff : S/P
# Cdiff : P/S

rack_data = pd.read_csv('경로 선정 필요')

# Bank, Rack ID 선정
rack_voltage = rack_data[(rack_data['BANK_ID'] == 1) & (rack_data['RACK_ID'] == 4)]['RACK_VOLTAGE'].values
rack_current = rack_data[(rack_data['BANK_ID'] == 1) & (rack_data['RACK_ID'] == 4)]['RACK_CURRENT'].values
rack_soc = rack_data[(rack_data['BANK_ID'] == 1) & (rack_data['RACK_ID'] == 4)]['RACK_SOC'].values

SOC = np.linspace(1, 0, num=21)

OCV_PL2 = loadmat('경로 선정 필요')

OCV_ESS = OCV_PL2

Time = 1
Init_cap = 228.57
Init_Ri = 0.0310707902382320 * 4
Init_Rdiff = 0.0190371443335961 * 4
Init_Cdiff = 6093.350870660123 / 4
ErrorCovariance = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
ForgettingFactor = 0.99999999

Vector_b0 = Init_Ri
Vector_b1 = (-Init_Ri + 1 / Init_Cdiff + Init_Ri / (Init_Rdiff * Init_Cdiff))
Vector_a1 = (1 / (Init_Rdiff * Init_Cdiff) - 1)

SOC_OCV_lookup = dict(zip(SOC, OCV_ESS))
OCV_SOC_lookup = dict(zip(OCV_ESS, SOC))

# SOC_OCV_lookup 테이블을 역으로 변환하여 SOC에 따른 OCV를 찾는 Lookup table 생성
sorted_SOC_OCV_lookup = sorted(SOC_OCV_lookup.items())
sorted_OCV_SOC_lookup = sorted(OCV_SOC_lookup.items())

def find_soc_from_voltage(voltage, lookup_table):
    ocvs, socs = zip(*lookup_table)  
    ocvs = np.array(ocvs)
    socs = np.array(socs)

    if voltage < ocvs.min():
        slope = (socs[1] - socs[0]) / (ocvs[1] - ocvs[0])
        extrapolated_soc = socs[0] + slope * (voltage - ocvs[0])
    elif voltage > ocvs.max():
        slope = (socs[-1] - socs[-2]) / (ocvs[-1] - ocvs[-2])
        extrapolated_soc = socs[-1] + slope * (voltage - ocvs[-1])
    else:
        extrapolated_soc = np.interp(voltage, ocvs, socs)
    
    return extrapolated_soc

def find_ocv_from_soc(soc, lookup_table):
    socs, ocvs = zip(*lookup_table)  
    socs = np.array(socs)
    ocvs = np.array(ocvs)

    if soc < socs.min():
        slope = (ocvs[1] - ocvs[0]) / (socs[1] - socs[0])
        extrapolated_ocv = ocvs[0] + slope * (soc - socs[0])
    elif soc > socs.max():
        slope = (ocvs[-1] - ocvs[-2]) / (socs[-1] - socs[-2])
        extrapolated_ocv = ocvs[-1] + slope * (soc - socs[-1])
    else:
        extrapolated_ocv = np.interp(soc, socs, ocvs)
    
    return extrapolated_ocv
