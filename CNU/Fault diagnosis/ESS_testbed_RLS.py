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

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat


SOC = np.linspace(1, 0, num=21)

OCV_PL2 = loadmat('경로 선정 필요')

OCV_ESS = OCV_PL2

SOC = np.linspace(1, 0, num=21)

# Create SOC_OCV and OCV_SOC lookup tables
SOC_OCV_lookup = dict(zip(SOC, OCV_ESS))
OCV_SOC_lookup = dict(zip(OCV_ESS, SOC))

rack_data = pd.read_csv('경로 선정 필요')

# 비교 분석할 Bank, Rack ID 선정
rack_voltage_1 = rack_data[(rack_data['BANK_ID'] == 1) & (rack_data['RACK_ID'] == 1)]['RACK_VOLTAGE'].values
rack_voltage_4 = rack_data[(rack_data['BANK_ID'] == 1) & (rack_data['RACK_ID'] == 4)]['RACK_VOLTAGE'].values
rack_current_1 = rack_data[(rack_data['BANK_ID'] == 1) & (rack_data['RACK_ID'] == 1)]['RACK_CURRENT'].values
rack_current_4 = rack_data[(rack_data['BANK_ID'] == 1) & (rack_data['RACK_ID'] == 4)]['RACK_CURRENT'].values

# Define initial parameters
Init_cap = 228.57  # Ah
Init_Ri = 0.0310707902382320 * 4  # Initial internal resistance (adjusted for rack size)
Init_Rdiff = 0.0190371443335961 * 4  # Initial Rdiff
Init_Cdiff = 6093.350870660123 / 4  # Initial Cdiff
ErrorCovariance = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
ForgettingFactor = 0.99999999

Vector_b0 = Init_Ri
Vector_b1 = (-Init_Ri + 1 / Init_Cdiff + Init_Ri / (Init_Rdiff * Init_Cdiff))
Vector_a1 = (1 / (Init_Rdiff * Init_Cdiff) - 1)

# SOC lookup functions
def find_soc_from_voltage(voltage, lookup_table):
    ocvs, socs = zip(*lookup_table.items())  # Get items from dictionary
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
    socs, ocvs = zip(*lookup_table.items())  # Get items from dictionary
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

# Initial SOC calculation from voltage for Rack 1 and Rack 4
Init_soc_rack_1 = [find_soc_from_voltage(vol, SOC_OCV_lookup) for vol in rack_voltage_1]
Init_soc_rack_4 = [find_soc_from_voltage(vol, SOC_OCV_lookup) for vol in rack_voltage_4]

# SOC Ref for Rack 1 and Rack 4 (calculating SOC over time)
SOC_ref_rack_1 = []
current_SOC_1 = Init_soc_rack_1[0]

for i in range(len(rack_current_1)):
    delta_SOC = (rack_current_1[i] / Init_cap) * (1 / 3600)
    current_SOC_1 += delta_SOC
    SOC_ref_rack_1.append(current_SOC_1)

SOC_ref_rack_4 = []
current_SOC_4 = Init_soc_rack_4[0]

for i in range(len(rack_current_4)):
    delta_SOC = (rack_current_4[i] / Init_cap) * (1 / 3600)
    current_SOC_4 += delta_SOC
    SOC_ref_rack_4.append(current_SOC_4)

# OCV for both racks
OCV_rack_1 = [find_ocv_from_soc(soc, SOC_OCV_lookup) for soc in SOC_ref_rack_1]
OCV_rack_4 = [find_ocv_from_soc(soc, SOC_OCV_lookup) for soc in SOC_ref_rack_4]

# RLS method for estimation of Ri, Rdiff, Cdiff
def calculate_gain_and_covariance(phi, P, forgetting_factor):
    P_phi = P @ phi
    gain_denominator = forgetting_factor + phi.T @ P @ phi 
    gain = P_phi / gain_denominator
    covariance = (P - gain @ phi.T @ P) / forgetting_factor
    covariance = np.clip(covariance, 0, 1e-10)  # Clip the values of covariance
    return gain, covariance

# Estimation for Rack 1 and Rack 4
Ri_estimates_1, Rdiff_estimates_1, Cdiff_estimates_1 = [Init_Ri], [Init_Rdiff], [Init_Cdiff]
Ri_estimates_4, Rdiff_estimates_4, Cdiff_estimates_4 = [Init_Ri], [Init_Rdiff], [Init_Cdiff]

for t in range(1, len(rack_voltage_1)):
    voltage_1 = rack_voltage_1[t]
    current_1 = rack_current_1[t]
    prev_current_1 = rack_current_1[t-1]
    ocv_1 = OCV_rack_1[t]
    
    voltage_4 = rack_voltage_4[t]
    current_4 = rack_current_4[t]
    prev_current_4 = rack_current_4[t-1]
    ocv_4 = OCV_rack_4[t]

    phi_1 = np.array([current_1, prev_current_1, ocv_1 - voltage_1]).reshape(-1, 1)
    gain_1, covariance_1 = calculate_gain_and_covariance(phi_1, ErrorCovariance, ForgettingFactor)
    ErrorCovariance = covariance_1
    Ri_1 = np.abs(gain_1[0, 0])
    Rdiff_1 = np.abs(gain_1[1, 0] - gain_1[2, 0] * gain_1[0, 0]) / (1 - gain_1[2, 0])
    Cdiff_1 = 1 / (gain_1[1, 0] - gain_1[2, 0] * gain_1[0, 0])

    Ri_estimates_1.append(Ri_1)
    Rdiff_estimates_1.append(Rdiff_1)
    Cdiff_estimates_1.append(Cdiff_1)

    phi_4 = np.array([current_4, prev_current_4, ocv_4 - voltage_4]).reshape(-1, 1)
    gain_4, covariance_4 = calculate_gain_and_covariance(phi_4, ErrorCovariance, ForgettingFactor)
    ErrorCovariance = covariance_4
    Ri_4 = np.abs(gain_4[0, 0])
    Rdiff_4 = np.abs(gain_4[1, 0] - gain_4[2, 0] * gain_4[0, 0]) / (1 - gain_4[2, 0])
    Cdiff_4 = 1 / (gain_4[1, 0] - gain_4[2, 0] * gain_4[0, 0])

    Ri_estimates_4.append(Ri_4)
    Rdiff_estimates_4.append(Rdiff_4)
    Cdiff_estimates_4.append(Cdiff_4)

# Calculate variations between Rack 1 and Rack 4
Rdiff_variations = []
Cdiff_variations = []

for rmax, rmin, cmax, cmin in zip(Rdiff_estimates_1, Rdiff_estimates_4, Cdiff_estimates_1, Cdiff_estimates_4):
    Rdiff_variation = abs((rmax - rmin) / rmax) 
    Cdiff_variation = abs((cmax - cmin) / cmax) 
    Rdiff_variations.append(Rdiff_variation)
    Cdiff_variations.append(Cdiff_variation)

# Plot results (Rdiff and Cdiff variations)
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(rack_voltage_1, label="Rack 1 Voltage")
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Voltage (V)')
ax1.set_ylim(800, 1000)
ax1.set_xlim(1, len(rack_voltage_1))

# Plot Rdiff and Cdiff variations on the same plot with a secondary y-axis
ax2 = ax1.twinx()
ax2.plot(Rdiff_variations, color='orange', label='Rdiff Variation', linestyle='-')
ax2.plot(Cdiff_variations, color='green', label='Cdiff Variation', linestyle='-')
ax2.set_ylabel('Variation (%)')

ax1.grid(True)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.show()
