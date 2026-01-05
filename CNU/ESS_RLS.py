# 판리 2호 Bank 1(Rack 9개, Rack 1개당 Tray 20S, Tray 1개당 12S60P --> Rack 1개당 240S60P) / Bank 2(Rack 8개, Rack 1개당 Tray 20S, Tray 1개당 12S20P --> Rack 1개당 240S60P)
# 컨테이너(Bank 1, 2 포함) : 3,320kWh

# 백마 Bank 1(Rack 9개, Rack 1개당 Tray 20S, Tray 1개당 12S60P --> Rack 1개당 240S60P)
# 컨테이너(Bank 1만) : 1,476kWh

# 황금 Bank 1(Rack 11개, Rack 1개당 Tray 17S, Tray 1개당 12S60P --> Rack 1개당 204*60P)
# 컨테이너(Bank 1만) : 1,826kWh

rack_data = pd.read_csv('D:/내 과제/한국전자기술연구원/ESS 운영사이트 이상현상 관련 자료/판리 2호/판리/20240315_rack.csv')
#rack_data = pd.read_csv('D:/내 과제/한국전자기술연구원/ESS 운영사이트 이상현상 관련 자료/백마/24.05.09/20240406_rack.csv')
#rack_data = pd.read_csv('D:/내 과제/한국전자기술연구원/ESS 운영사이트 이상현상 관련 자료/황금 6호/황금6호/20230921_rack.csv')

rack_voltage = rack_data[(rack_data['BANK_ID'] == 1) & (rack_data['RACK_ID'] == 1)]['RACK_VOLTAGE'].values
rack_current = rack_data[(rack_data['BANK_ID'] == 1) & (rack_data['RACK_ID'] == 1)]['RACK_CURRENT'].values
rack_soc = rack_data[(rack_data['BANK_ID'] == 1) & (rack_data['RACK_ID'] == 1)]['RACK_SOC'].values

SOC = np.linspace(1, 0, num=21)

OCV_PL2 = loadmat('D:/내 과제/한국전자기술연구원/매트랩 코드/OCV_PL2_real.mat')['OCV_PL2_real'].flatten()
OCV_BM = loadmat('D:/내 과제/한국전자기술연구원/매트랩 코드/OCV_BM_real.mat')['OCV_BM_real'].flatten()
OCV_HG = loadmat('D:/내 과제/한국전자기술연구원/매트랩 코드/OCV_HG_real.mat')['OCV_HG_real'].flatten()

OCV_ESS = OCV_PL2

# 판리 2호(Bank 1의 Rack 한개) 초기 용량 : 3,320kWh * 9/17 = 1757.647hWh --> Rack 1개(1757.647hWh/9 = 195.294kWh) --> 195.294kWh/(3.56 * 240) = 228.57Ah
# 백마(Bank 1의 Rack 한개) 초기 용량 : 1,476kWh --> Rack 1개(1,476kWh/9 = 164kWh) --> 164kWh/(3.56 * 240) = 191.95Ah
# 황금(Bank 1의 Rack 한개) 초기 용량 : 1,826kWh --> Rack 1개(1,826kWh/11 = 166kWh) --> 166kWh/(3.56 * 204) = 228.57Ah

# Ri, Rdiff : S/P
# Cdiff : P/S

#황금만 초기 저항 값 3.4배
Time = 1
Init_cap = 228.57
Init_Ri = 0.0287465777883587 * 4
Init_Rdiff = 0.0183935245804069 * 4
Init_Cdiff = 69100.4051150639 / 4
ErrorCovariance = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
ForgettingFactor = 0.999999999999999

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

# Initial SOC calculation from voltage
Init_soc_rack = [find_soc_from_voltage(vol, sorted_OCV_SOC_lookup) for vol in rack_voltage]

# Ah counting to calculate SOC
SOC_ref_rack = []
current_SOC = Init_soc_rack[0]

for i in range(len(rack_current)):
    delta_SOC = (rack_current[i] / Init_cap) * (Time / 3600)
    current_SOC += delta_SOC
    SOC_ref_rack.append(current_SOC)

OCV_rack = [find_ocv_from_soc(soc, sorted_SOC_OCV_lookup) for soc in SOC_ref_rack]

theta = np.array([Vector_b0, Vector_b1, Vector_a1]).reshape(-1, 1)
P = ErrorCovariance

Ri_estimates = [Init_Ri]
Rdiff_estimates = [Init_Rdiff]
Cdiff_estimates = [Init_Cdiff]

def calculate_gain_and_covariance(phi, P, forgetting_factor):
    P_phi = P @ phi
    gain_denominator = forgetting_factor + phi.T @ P @ phi 
    gain = P_phi / gain_denominator
    covariance = (P - gain @ phi.T @ P) / forgetting_factor
    covariance = np.clip(covariance, 0, 1e-25)  # Clip the values of covariance
    return gain, covariance

gain_values = []
overpotential_values = []
RlsEstimatedVoltage_values = []
RlsVoltageError_values = []
covariance_values = []
theta_values = []
phi_values = []

for t in range(1, len(rack_voltage)):
    voltage = rack_voltage[t]
    current = rack_current[t]
    prev_current = rack_current[t-1]
    ocv = OCV_rack[t]

    overpotential = ocv - voltage
    overpotential_values.append(overpotential)

    phi = np.array([current, prev_current, overpotential]).reshape(-1, 1)
    phi_values.append(phi.flatten())

    RlsEstimatedVoltage = theta.T @ phi
    RlsEstimatedVoltage_values.append(RlsEstimatedVoltage[0, 0])

    RlsVoltageError = voltage - ocv - RlsEstimatedVoltage
    RlsVoltageError_values.append(RlsVoltageError[0, 0])


    gain, covariance = calculate_gain_and_covariance(phi, P, ForgettingFactor)
    gain_values.append(gain.flatten())
    covariance_values.append(covariance.flatten())

    P = covariance

    theta = theta + gain @ RlsVoltageError
    theta = np.abs(theta)
    theta_values.append(theta.flatten())

    Ri = theta[0, 0]
    Rdiff = (theta[1, 0] - theta[2, 0] * theta[0, 0]) / (1 - theta[2, 0])
    Cdiff = 1 / (theta[1, 0] - theta[2, 0] * theta[0, 0])

    Ri = np.abs(Ri)
    Rdiff = np.abs(Rdiff)
    Cdiff = np.abs(Cdiff)

    Ri_estimates.append(Ri)
    Rdiff_estimates.append(Rdiff)
    Cdiff_estimates.append(Cdiff)

gain_values = np.array(gain_values).reshape(-1, 3)
covariance_values = np.array(covariance_values).reshape(-1, 9)  # 3x3 matrix flattened
theta_values = np.array(theta_values).reshape(-1, 3)
phi_values = np.array(phi_values).reshape(-1, 3)

# ECM 설계
Vi = np.array(rack_current) * np.array(Ri_estimates)
I_diff_1RC = np.zeros(len(rack_current))
ECM_voltage_rack = np.zeros(len(rack_current))
Vdiff_array = np.zeros(len(rack_current))  # Vdiff를 저장할 배열

# 초기값 설정
I_diff_1RC[0] = rack_current[0]

# I_diff_1RC 및 ECM 전압 계산
for t in range(1, len(rack_current)):
    I_diff_1RC[t] = (I_diff_1RC[t-1] * np.exp(-1 / (Rdiff_estimates[t] * Cdiff_estimates[t])) + 
                     rack_current[t-1] * (1 - np.exp(-1 / (Rdiff_estimates[t] * Cdiff_estimates[t]))))
    
    Vdiff = I_diff_1RC[t] * Rdiff_estimates[t]
    Vdiff_array[t] = Vdiff
    ECM_voltage_rack[t] = OCV_rack[t] + Vi[t] + Vdiff
    
# 실제 단자 전압과 ECM 추정 전압 비교
plt.figure(figsize=(10, 6))
plt.plot(rack_voltage, label='Real terminal voltage')
plt.plot(ECM_voltage_rack, label='ECM estimated voltage')
plt.xlabel('Time (s)')
plt.xlim([0, len(rack_voltage) - 1])
plt.ylim(800, 1000)
plt.ylabel('Voltage (V)')
plt.legend()
plt.show()

# Gain 값 플롯
plt.figure(figsize=(10, 6))
plt.plot(gain_values[:, 0], label='Gain[0]')
plt.plot(gain_values[:, 1], label='Gain[1]')
plt.plot(gain_values[:, 2], label='Gain[2]')
plt.xlabel('Time Step')
plt.ylabel('Gain Value')
plt.title('Gain Values Over Time')
plt.legend()
plt.show()

# Overpotential 값 플롯
plt.figure(figsize=(10, 6))
plt.plot(overpotential_values, label='Overpotential')
plt.xlabel('Time Step')
plt.ylabel('Overpotential')
plt.title('Overpotential Over Time')
plt.legend()
plt.show()

# RlsEstimatedVoltage 값 플롯
plt.figure(figsize=(10, 6))
plt.plot(RlsEstimatedVoltage_values, label='RlsEstimatedVoltage')
plt.xlabel('Time Step')
plt.ylabel('RlsEstimatedVoltage')
plt.title('RlsEstimatedVoltage Over Time')
plt.legend()
plt.show()

# RlsVoltageError 값 플롯
plt.figure(figsize=(10, 6))
plt.plot(RlsVoltageError_values, label='RlsVoltageError')
plt.xlabel('Time Step')
plt.ylabel('RlsVoltageError')
plt.title('RlsVoltageError Over Time')
plt.legend()
plt.show()


# Covariance Update 값 플롯
plt.figure(figsize=(10, 6))
plt.plot(covariance_values[:, 0], label='Covariance[0]')
plt.plot(covariance_values[:, 4], label='Covariance[4]')
plt.plot(covariance_values[:, 8], label='Covariance[8]')
plt.xlabel('Time Step')
plt.ylabel('Covariance Update Value')
plt.title('Covariance Update Values Over Time')
plt.legend()
plt.show()
