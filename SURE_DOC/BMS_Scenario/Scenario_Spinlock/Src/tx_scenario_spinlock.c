#include "main.h"
#include "cmsis_os.h"
#include <stdio.h> //기본 라이브러리
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "cpu_utils.h"
#include "FreeRTOS.h"
#include "task.h"
#include "tx_scenario_original.h"

/* task's priority */
extern ADC_HandleTypeDef hadc1;
extern UART_HandleTypeDef huart2;
extern UART_HandleTypeDef huart4;
extern volatile unsigned long ulHighFrequencyTimerTicks;
extern UBaseType_t stack_value;
extern uint32_t heapvalue_max;
extern uint32_t heapvalue_current;
extern RNG_HandleTypeDef hrng;
extern double cpu_usage;
extern uint32_t heapvalue_current;
extern uint32_t heapvalue_max;
extern uint8_t tx_data[32];

#if 1
/* The task functions. */
static void TxDataTrans();
static cell_comparison Cell_Tray();
static tray_comparison Tray_Rack();
static void Cell_create(cell_info cell_buffer[]);
//static void measureTaskCPUUsage(void);

static void malloc_add();
static void malloc_free();


uint16_t calc=0;
cell_info cell_err_buf[12]={0,};
cell_comparison tray_err_buf[17]={0,};
tray_comparison rack_err_buf;

uint16_t cellPosition=0;
uint16_t trayPosition=0;
uint16_t cellErrorBufPosition=0;
uint16_t  trayErrorBufPosition=0;

uint8_t cell=0;
uint8_t tray=0;
uint8_t rack=0;
cell_info cell_buffer[12]={0,};
cell_comparison tray_buffer;
cell_comparison tray_rack_buffer[17]={0,};
tray_comparison rack_buffer;
uint16_t normal_task_classification = 0;
float min_voltage = 2.0f;
float max_voltage = 5.0f;
float min_soc = 0.0f;
float max_soc = 100.0f;
float cell_voltage_difference = 0.0f;
float cell_temperature_difference = 0.0f;
float current_voltage =0.0f;
float soc = 0.0f;
float tray_temperature_difference =0.0f;
float rack_current_difference =0.0f;
float out = 0.f;

typedef struct error
{
	float error;
	float sum;
}Pi_Errors;
typedef struct gain
{
	float P;
	float I;
}Pi_Gains;

Pi_Errors error_test;
Pi_Gains gain_test;


#define Cell_Overcharge_Error  1
#define Cell_Overcharge_Emergency_Stop 2
#define Cell_Overdischarge_Error 3
#define Cell_Overdischarge_Emergency_Stop 4
#define Cell_Over_Voltage_Error 5
#define Cell_Over_Voltage_Emergency_Stop 6
#define Cell_Voltage_Unbalance_Error 7
#define Cell_Voltage_Unbalance_Emergency_Stop 8
#define Cell_Temperature_Unbalance_Error 9
#define Cell_Temperature_Unbalance_Emergency_Stop 10
#define Tray_Temperature_Unbalance_error 11
#define Tray_Temperature_Unbalance_Emergency_Stop 12
#define Rack_Current_Unbalance_Error 13
#define Rack_Current_Unbalance_Emergency_Stop 14
#define Task_Count_Num	995
#define Deadlock_On 40000000

volatile BaseType_t xTaskReady = pdFALSE;
uint16_t func1condition=0;

static uint32_t func_count=0;
uint16_t calcLoop=0;
#endif

static void MCU_Temperature(void);

osThreadId TaskHandle;
osThreadId TaskHandle1;
osThreadId TaskHandle2;
osThreadId TaskHandle3;
uint16_t usage=0;
uint16_t cpuUsage=0;
uint8_t task_deadlock=0;
uint8_t task_classification = 0;
uint8_t malloc_abnormal=0;
uint32_t temp=0;
static uint8_t taskID = 0;
float adc, temperature;
uint8_t task_random=0;
uint32_t normal_random_parameter=0;
uint32_t test1=0;
void Monitor_TASK(void const * argument)//monitor
{
	//uint8_t u8_test1[1026]={0,};
    while(1)
    {
    	task_random =(HAL_RNG_GetRandomNumber(&hrng) % 3);
    	usage = osGetCPUUsage();
        heapvalue_current = xPortGetFreeHeapSize();
		heapvalue_max = xPortGetMinimumEverFreeHeapSize();
		MCU_Temperature();

		TxDataTrans();
    	osDelay(100);
    }
}

void vTaskSpinlockExample(void)
{
	while(xTaskReady == pdFALSE)
	{
		task_deadlock=1;
		test1++;
		if(test1 > 20000000)
		{
			test1=0;
			usage = osGetCPUUsage();
			xTaskReady = pdTRUE;
		}
	}
	  xTaskReady = pdFALSE;
}

void Memory_TASK(void const * argument)
{
   	//uint8_t u8_test2[900]={0,};//hardfault
	while(1)
	{
#if 1
		Cell_Tray();
		Tray_Rack();

		normal_random_parameter=(HAL_RNG_GetRandomNumber(&hrng)%1000);

		if(normal_task_classification > 1000 && normal_random_parameter > 993)
		{
			vTaskSpinlockExample();
			normal_task_classification = 0;
		}

		task_deadlock = 0;
#endif
		normal_task_classification++;
		vTaskPrioritySet(TaskHandle1, task_random);
		osDelay(100);
	}
}

static void Calc_TASK(void const * argument)
{
    while(1)
    {
		for(calc=0; calc<12;calc++)
		{
			cell_voltage_difference = ((cell_buffer[calc].CELL_VOLTAGE - cell_buffer[calc-1].CELL_VOLTAGE) * 0.001);
			cell_temperature_difference = (cell_buffer[calc].CELL_temperature - cell_buffer[calc-1].CELL_temperature);
			current_voltage = cell_buffer[calc].CELL_VOLTAGE;

			soc = (current_voltage - min_voltage) / (max_voltage - min_voltage) * (max_soc - min_soc) + min_soc;
			soc = cell_err_buf[1].CELL_SOC;
		}

		normal_task_classification++;
		vTaskPrioritySet(TaskHandle2, task_random);
    	osDelay(100);
    }
}

void Datatx_TASK(void const * argument)
{
	while(1)
	{
		for(cellPosition=0; cellPosition<12; cellPosition++)
		{
			if(cellPosition > 0)
			{
				if(cell_buffer[cellPosition].CELL_VOLTAGE >= 4.086 && cell_buffer[cellPosition].CELL_VOLTAGE < 4.1)
				{
					cell_err_buf[cellPosition].CELL_VOLTAGE = Cell_Overcharge_Error;
				}
				else if(cell_buffer[cellPosition].CELL_VOLTAGE >= 4.1)
				{
					cell_err_buf[cellPosition].CELL_VOLTAGE = Cell_Overcharge_Emergency_Stop;
					HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_12);
				}
				if(cell_buffer[cellPosition].CELL_VOLTAGE <= 3.15 && cell_buffer[cellPosition].CELL_VOLTAGE > 2.9)
				{
					cell_err_buf[cellPosition].CELL_VOLTAGE = Cell_Overdischarge_Error;
				}
				else if(cell_buffer[cellPosition].CELL_VOLTAGE <= 2.9)
				{
					cell_err_buf[cellPosition].CELL_VOLTAGE = Cell_Overdischarge_Emergency_Stop;
					HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_13);
				}
				if(cell_buffer[cellPosition].CELL_VOLTAGE >= 4.086 && cell_buffer[cellPosition].CELL_VOLTAGE < 4.1)
				{
					cell_err_buf[cellPosition].CELL_VOLTAGE = Cell_Over_Voltage_Error;
				}
				else if(cell_buffer[cellPosition].CELL_VOLTAGE >= 4.1)
				{
					cell_err_buf[cellPosition].CELL_VOLTAGE = Cell_Over_Voltage_Emergency_Stop;
					HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_14);
				}
				if(cell_voltage_difference >= 300 && cell_voltage_difference < 500)
				{
					cell_err_buf[cellPosition].CELL_VOLTAGE_difference = Cell_Voltage_Unbalance_Error;
					HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_14);
				}
				else if(cell_voltage_difference >= 500)
				{
					cell_err_buf[cellPosition].CELL_VOLTAGE_difference = Cell_Voltage_Unbalance_Emergency_Stop;
					HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_15);
				}
				if(cell_temperature_difference >= 20 && cell_temperature_difference < 25 )
				{
					cell_err_buf[cellPosition].CELL_temperature_difference = Cell_Temperature_Unbalance_Error;
				}
				else if(cell_temperature_difference >= 25 )
				{
					cell_err_buf[cellPosition].CELL_temperature_difference = Cell_Temperature_Unbalance_Emergency_Stop;
					HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_12);
					HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_13);
				}
			}
		}
		//tray error
		for(trayPosition=0; trayPosition<17; trayPosition++)
		{
			if(trayPosition > 0)
			{
				tray_temperature_difference = (tray_rack_buffer[trayPosition].TRAY_MAX_CELL_temperature - tray_rack_buffer[trayPosition-1].TRAY_MIN_CELL_temperature);
				if(tray_temperature_difference >= 50 & tray_temperature_difference < 55)
				{

					tray_err_buf[trayPosition].TRAY_temperature_difference = Tray_Temperature_Unbalance_error;
				}
				else if(tray_temperature_difference >= 55)
				{
					tray_err_buf[trayPosition].TRAY_temperature_difference = Tray_Temperature_Unbalance_Emergency_Stop;
					HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_13);
					HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_14);
				}
			}
		}

		//rack error
		//float rack_current_difference = rack_buffer.RACK_MAX_CELL_CURRENT - rack_buffer.RACK_MIN_CELL_CURRENT;
		rack_current_difference = rack_buffer.RACK_MAX_CELL_CURRENT - rack_buffer.RACK_MIN_CELL_CURRENT;

		if(rack_current_difference >= 140 & rack_current_difference < 150)
		{
			rack_err_buf.RACK_current_difference = Rack_Current_Unbalance_Error;
		}
		else if(rack_current_difference >= 150)
		{
			rack_err_buf.RACK_current_difference = Rack_Current_Unbalance_Emergency_Stop;
			HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_14);
			HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_15);
		}

		for(cellErrorBufPosition=0; cellErrorBufPosition<12; cellErrorBufPosition++)
		{
			printf("%.2f ",cell_err_buf[cellErrorBufPosition].CELL_VOLTAGE);
			printf("%.2f ",cell_err_buf[cellErrorBufPosition].CELL_VOLTAGE_difference);
			printf("%.2f",cell_err_buf[cellErrorBufPosition].CELL_temperature_difference);
			printf("%.2f\n\n",cell_err_buf[cellErrorBufPosition].CELL_SOC);
		}
		for(trayErrorBufPosition=0; trayErrorBufPosition<17;trayErrorBufPosition++)
		{
			printf("%.2f\n\n",tray_err_buf[trayErrorBufPosition].TRAY_temperature_difference);

		}
		printf("%.2f\n\n",rack_err_buf.RACK_current_difference);

		normal_task_classification++;
		vTaskPrioritySet(TaskHandle3, task_random);
    	osDelay(100);
	}
}

void TxDataTrans()
{
	memset(tx_data, 0, sizeof(tx_data));

	tx_data[0] = cpuUsage >> 8 ;
	tx_data[1] = cpuUsage;
	tx_data[2] = (stack_value >> 8) & 0xFF;
	tx_data[3] = (stack_value & 0xFF);
	tx_data[4] = heapvalue_current >> 8;
	tx_data[5] = heapvalue_current;
	tx_data[6] = heapvalue_max >> 8;
	tx_data[7] = heapvalue_max;
	tx_data[8] = temp >> 8;
	tx_data[9] = temp;
	tx_data[10] = task_deadlock;
	tx_data[11] = task_classification;
	tx_data[12] = malloc_abnormal;

	HAL_UART_Transmit_DMA(&huart2, (uint8_t*)tx_data, sizeof(tx_data));
}

static cell_comparison Cell_Tray()
{
	Cell_create(cell_buffer);

	//tray create
	tray_buffer.TRAY_MIN_CELL_temperature = cell_buffer[0].CELL_temperature;
	tray_buffer.TRAY_MIN_CELL_temperature_POSITION = cell_buffer[0].CELL_ID;

	tray_buffer.TRAY_MAX_CELL_temperature = cell_buffer[0].CELL_temperature;
	tray_buffer.TRAY_MAX_CELL_temperature_POSITION = cell_buffer[0].CELL_ID;

	tray_buffer.TRAY_MIN_CELL_VOLTAGE = cell_buffer[0].CELL_VOLTAGE;
	tray_buffer.TRAY_MIN_CELL_VOLTAGE_POSITION = cell_buffer[0].CELL_ID;

	tray_buffer.TRAY_MAX_CELL_VOLTAGE = cell_buffer[0].CELL_VOLTAGE;
	tray_buffer.TRAY_MAX_CELL_VOLTAGE_POSITION = cell_buffer[0].CELL_ID;

	tray_buffer.TRAY_MIN_CELL_CURRENT = cell_buffer[0].CELL_CURRENT;
	tray_buffer.TRAY_MIN_CELL_CURRENT_POSITION = cell_buffer[0].CELL_ID;

	tray_buffer.TRAY_MAX_CELL_CURRENT = cell_buffer[0].CELL_CURRENT;
	tray_buffer.TRAY_MAX_CELL_CURRENT_POSITION = cell_buffer[0].CELL_ID;

	for(tray=1; tray<11; tray++)
	{
		if(tray_buffer.TRAY_MIN_CELL_temperature >= cell_buffer[tray].CELL_temperature)
		{
			tray_buffer.TRAY_MIN_CELL_temperature = cell_buffer[tray].CELL_temperature;
			tray_buffer.TRAY_MIN_CELL_temperature_POSITION = cell_buffer[tray].CELL_ID;
		}

		if(tray_buffer.TRAY_MAX_CELL_temperature <= cell_buffer[tray].CELL_temperature)
		{
			tray_buffer.TRAY_MAX_CELL_temperature = cell_buffer[tray].CELL_temperature;
			tray_buffer.TRAY_MAX_CELL_temperature_POSITION = cell_buffer[cell].CELL_ID;
		}

		if(tray_buffer.TRAY_MIN_CELL_VOLTAGE >= cell_buffer[tray].CELL_VOLTAGE)
		{
			tray_buffer.TRAY_MIN_CELL_VOLTAGE = cell_buffer[tray].CELL_VOLTAGE;
			tray_buffer.TRAY_MIN_CELL_VOLTAGE_POSITION = cell_buffer[cell].CELL_ID;
		}

		if(tray_buffer.TRAY_MAX_CELL_VOLTAGE <= cell_buffer[tray].CELL_VOLTAGE)
		{
			tray_buffer.TRAY_MAX_CELL_VOLTAGE = cell_buffer[tray].CELL_VOLTAGE;
			tray_buffer.TRAY_MAX_CELL_VOLTAGE_POSITION = cell_buffer[tray].CELL_ID;
		}
		if(tray_buffer.TRAY_MIN_CELL_CURRENT >= cell_buffer[tray].CELL_CURRENT)
		{
			tray_buffer.TRAY_MIN_CELL_CURRENT = cell_buffer[tray].CELL_CURRENT;
			tray_buffer.TRAY_MIN_CELL_CURRENT_POSITION = cell_buffer[tray].CELL_ID;
		}

		if(tray_buffer.TRAY_MAX_CELL_CURRENT <= cell_buffer[tray].CELL_CURRENT)
		{
			tray_buffer.TRAY_MAX_CELL_CURRENT = cell_buffer[tray].CELL_CURRENT;
			tray_buffer.TRAY_MAX_CELL_CURRENT_POSITION = cell_buffer[tray].CELL_ID;
		}
	}
	return tray_buffer;
}

static tray_comparison Tray_Rack()
{
	cell_comparison tray_rack_buffer[17]={0,};
	tray_comparison rack_buffer;

	tray_rack_buffer[0] = Cell_Tray();

	rack_buffer.RACK_MIN_CELL_temperature = tray_rack_buffer[0].TRAY_MIN_CELL_temperature;
	rack_buffer.RACK_MIN_CELL_temperature_POSITION = tray_rack_buffer[0].TRAY_MIN_CELL_temperature_POSITION;

	rack_buffer.RACK_MAX_CELL_temperature = tray_rack_buffer[0].TRAY_MAX_CELL_temperature;
	rack_buffer.RACK_MAX_CELL_temperature_POSITION = tray_rack_buffer[0].TRAY_MAX_CELL_temperature_POSITION;

	rack_buffer.RACK_MIN_CELL_VOLTAGE = tray_rack_buffer[0].TRAY_MIN_CELL_VOLTAGE;
	rack_buffer.RACK_MIN_CELL_VOLTAGE_POSITION = tray_rack_buffer[0].TRAY_MIN_CELL_VOLTAGE_POSITION;

	rack_buffer.RACK_MAX_CELL_VOLTAGE = tray_rack_buffer[0].TRAY_MAX_CELL_VOLTAGE;
	rack_buffer.RACK_MAX_CELL_VOLTAGE_POSITION = tray_rack_buffer[0].TRAY_MAX_CELL_VOLTAGE_POSITION;

	rack_buffer.RACK_MIN_CELL_CURRENT = tray_rack_buffer[0].TRAY_MIN_CELL_CURRENT;
	rack_buffer.RACK_MIN_CELL_CURRENT_POSITION = tray_rack_buffer[0].TRAY_MIN_CELL_CURRENT_POSITION;

	rack_buffer.RACK_MAX_CELL_CURRENT = tray_rack_buffer[0].TRAY_MAX_CELL_CURRENT;
	rack_buffer.RACK_MAX_CELL_CURRENT_POSITION = tray_rack_buffer[0].TRAY_MAX_CELL_CURRENT_POSITION;

	for(rack = 1; rack<16; rack++)
	{
		tray_rack_buffer[rack] = Cell_Tray();

		if(rack_buffer.RACK_MIN_CELL_temperature >= tray_rack_buffer[rack].TRAY_MIN_CELL_temperature)
		{
			rack_buffer.RACK_MIN_CELL_temperature = tray_rack_buffer[rack].TRAY_MIN_CELL_temperature;
			rack_buffer.RACK_MIN_CELL_temperature_POSITION = tray_rack_buffer[rack].TRAY_MIN_CELL_temperature_POSITION;
		}

		if(rack_buffer.RACK_MAX_CELL_temperature <= tray_rack_buffer[rack].TRAY_MAX_CELL_temperature)
		{
			rack_buffer.RACK_MAX_CELL_temperature = tray_rack_buffer[rack].TRAY_MAX_CELL_temperature;
			rack_buffer.RACK_MAX_CELL_temperature_POSITION = tray_rack_buffer[rack].TRAY_MAX_CELL_temperature_POSITION;
		}

		if(rack_buffer.RACK_MIN_CELL_VOLTAGE >= tray_rack_buffer[rack].TRAY_MIN_CELL_VOLTAGE)
		{
			rack_buffer.RACK_MIN_CELL_VOLTAGE = tray_rack_buffer[rack].TRAY_MIN_CELL_VOLTAGE;
			rack_buffer.RACK_MIN_CELL_VOLTAGE_POSITION = tray_rack_buffer[rack].TRAY_MIN_CELL_VOLTAGE_POSITION;
		}

		if(rack_buffer.RACK_MAX_CELL_VOLTAGE <= tray_rack_buffer[rack].TRAY_MAX_CELL_VOLTAGE)
		{
			rack_buffer.RACK_MAX_CELL_VOLTAGE = tray_rack_buffer[rack].TRAY_MAX_CELL_VOLTAGE;
			rack_buffer.RACK_MAX_CELL_VOLTAGE_POSITION = tray_rack_buffer[rack].TRAY_MAX_CELL_VOLTAGE_POSITION;
		}
		if(rack_buffer.RACK_MIN_CELL_CURRENT >= tray_rack_buffer[rack].TRAY_MIN_CELL_CURRENT)
		{
			rack_buffer.RACK_MIN_CELL_CURRENT = tray_rack_buffer[rack].TRAY_MIN_CELL_CURRENT;
			rack_buffer.RACK_MIN_CELL_CURRENT_POSITION = tray_rack_buffer[rack].TRAY_MIN_CELL_CURRENT_POSITION;
		}

		if(rack_buffer.RACK_MAX_CELL_CURRENT <= tray_rack_buffer[rack].TRAY_MAX_CELL_CURRENT)
		{
			rack_buffer.RACK_MAX_CELL_CURRENT = tray_rack_buffer[rack].TRAY_MAX_CELL_CURRENT;
			rack_buffer.RACK_MAX_CELL_CURRENT_POSITION = tray_rack_buffer[rack].TRAY_MAX_CELL_CURRENT_POSITION;
		}
	}
	return rack_buffer;
}

static void Cell_create(cell_info cell_buffer[])
{
	for(cell=0; cell<12; cell++)
	{
		cell_buffer[cell].CELL_ID = cell+1;
		cell_buffer[cell].CELL_temperature =(HAL_RNG_GetRandomNumber(&hrng)%100);
		cell_buffer[cell].CELL_VOLTAGE = (HAL_RNG_GetRandomNumber(&hrng)%5);
		cell_buffer[cell].CELL_CURRENT = (HAL_RNG_GetRandomNumber(&hrng)%300);
	}
}


static void MCU_Temperature()
{
	HAL_ADC_Start(&hadc1);
	HAL_ADC_PollForConversion(&hadc1, HAL_MAX_DELAY);

	adc = HAL_ADC_GetValue(&hadc1);
	temperature = adc * 3.3 / 0xfff;
	temperature = (temperature-0.76)/0.025 + 25.0;
	temp = temperature*10;
}
/*-----------------------------------------------------------*/

/* USER CODE BEGIN Header_StartMyTask1 */
/**
  * @brief  Function implementing the MyTask1 thread.
  * @param  argument: Not used
  * @retval None
  */
/* USER CODE END Header_StartMyTask1 */
void StartMyTask1(void const * argument)
{
  /* USER CODE BEGIN 5 */
  /* Infinite loop */

  osThreadDef(Task, Monitor_TASK, osPriorityRealtime, 0, 256);
  TaskHandle = osThreadCreate(osThread(Task), NULL);

  osThreadDef(Task1,Memory_TASK, osPriorityNormal, 0, 512);
  TaskHandle1 = osThreadCreate(osThread(Task1), NULL);

  osThreadDef(Task2, Calc_TASK, osPriorityNormal, 0, 256);
  TaskHandle2 = osThreadCreate(osThread(Task2), NULL);

  osThreadDef(Task3, Datatx_TASK, osPriorityNormal, 0, 256);
  TaskHandle3 = osThreadCreate(osThread(Task3), NULL);

  while(1)
  {
      osDelay(100);
  }
}
