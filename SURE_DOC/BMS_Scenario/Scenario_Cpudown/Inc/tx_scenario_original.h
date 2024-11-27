/*
 * tx_scenario_original.h
 *
 *  Created on: 2024. 3. 22.
 *      Author: sure
 */

#ifndef INC_TX_SCENARIO_ORIGINAL_H_
#define INC_TX_SCENARIO_ORIGINAL_H_

typedef unsigned long UBaseType_t;
typedef struct cell
{
	uint8_t CELL_ID;
	float CELL_temperature;
	float CELL_VOLTAGE;
	float CELL_CURRENT;
	float CELL_VOLTAGE_difference;
	float CELL_temperature_difference;
	float CELL_SOC;
}cell_info;

typedef struct cell_comparison
{
	float TRAY_MIN_CELL_temperature;
	uint8_t TRAY_MIN_CELL_temperature_POSITION;
	float TRAY_MAX_CELL_temperature;
	uint8_t TRAY_MAX_CELL_temperature_POSITION;
	float TRAY_MIN_CELL_VOLTAGE;
	uint8_t TRAY_MIN_CELL_VOLTAGE_POSITION;
	float TRAY_MAX_CELL_VOLTAGE;
	uint8_t TRAY_MAX_CELL_VOLTAGE_POSITION;
	float TRAY_MIN_CELL_CURRENT;
	uint8_t TRAY_MIN_CELL_CURRENT_POSITION;
	float TRAY_MAX_CELL_CURRENT;
	uint8_t TRAY_MAX_CELL_CURRENT_POSITION;
	float TRAY_temperature_difference;
}cell_comparison;

typedef struct tray_comparison
{
	float RACK_MIN_CELL_temperature;
	uint8_t RACK_MIN_CELL_temperature_POSITION;
	float RACK_MAX_CELL_temperature;
	uint8_t RACK_MAX_CELL_temperature_POSITION;
	float RACK_MIN_CELL_VOLTAGE;
	uint8_t RACK_MIN_CELL_VOLTAGE_POSITION;
	float RACK_MAX_CELL_VOLTAGE;
	uint8_t RACK_MAX_CELL_VOLTAGE_POSITION;
	float RACK_MIN_CELL_CURRENT;
	uint8_t RACK_MIN_CELL_CURRENT_POSITION;
	float RACK_MAX_CELL_CURRENT;
	uint8_t RACK_MAX_CELL_CURRENT_POSITION;
	float RACK_SOC;
	float RACK_current_difference;
} tray_comparison;

#endif /* INC_TX_SCENARIO_ORIGINAL_H_ */
