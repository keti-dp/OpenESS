col_sel = ['TIMESTAMP', 'BANK_ID',  'RACK_ID',
           'RACK_MAX_CELL_VOLTAGE', 'RACK_MIN_CELL_VOLTAGE',
           'RACK_CELL_VOLTAGE_GAP', 'RACK_CURRENT',
           'RACK_MAX_CELL_TEMPERATURE', 'RACK_MIN_CELL_TEMPERATURE',
           'RACK_CELL_TEMPERATURE_GAP']


# 상한 안정도 정보 (ζ^4 일때를 기준)
# 과충전/과방전, 과전류, 온도불평형, 전압불평형, 고온/저온
safety_inf = {"OVER_VOLTAGE":{"upper_safety":4.05, "maximum_safety":4.014},
            "UNDER_VOLTAGE":{"upper_safety":3.2, "maximum_safety":3.34},
            "VOLTAGE_UNBALANCE":{"upper_safety":0.3, "maximum_safety":0.158},              
            "OVER_CURRENT":{"upper_safety":100, "maximum_safety":85.75},
            "OVER_TEMPERATURE":{"upper_safety":50, "maximum_safety":47.3},
            "UNDER_TEMPERATURE":{"upper_safety":0, "maximum_safety":5.3},
            "TEMPERATURE_UNBALANCE":{"upper_safety":20, "maximum_safety":17.35}}

condi_1 = ["OVER_VOLTAGE", "VOLTAGE_UNBALANCE", "OVER_TEMPERATURE", "TEMPERATURE_UNBALANCE", "OVER_CURRENT"]
condi_2 = ["UNDER_VOLTAGE", "UNDER_TEMPERATURE"]

