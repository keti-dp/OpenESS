#include "main.h"
#include "cmsis_os.h"
#include <stdio.h> //기본 라이브러리
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "cpu_utils.h"
#include "FreeRTOS.h"
#include "task.h"

/* task's priority */
extern ADC_HandleTypeDef hadc1;
extern UART_HandleTypeDef huart2;
extern UART_HandleTypeDef huart4;

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

/* The task functions. */
static void MCU_Temperature(void);
static void TxDataTrans();
static cell_comparison Cell_Tray();
static tray_comparison Tray_Rack();
static void Cell_create(cell_info cell_buffer[]);
static void Memory_func(uint32_t memory_size);
static void Calc_func(uint32_t parameter);
static void GPIO_func(uint32_t parameter);
//static void measureTaskCPUUsage(void);

static void func1();
static void func2();
static void func3();
static void func4();
static void func5();
static void func6();
static void malloc_func1();
static void malloc_add();
static void malloc_free();
//TaskHandle_t TX_xHandle1, TX_xHandle2, TX_xHandle3;
extern volatile unsigned long ulHighFrequencyTimerTicks;
//extern volatile unsigned long task1_Percentage;

extern UBaseType_t stack_value;
extern uint32_t heapvalue_max;
extern uint32_t heapvalue_current;
extern RNG_HandleTypeDef hrng;
extern double cpu_usage;
extern uint32_t heapvalue_current;
extern uint32_t heapvalue_max;
extern uint8_t tx_data[32];

float adc, temperature;
const int maxTasksToCheck = 3;

uint32_t Deadlock_count=0;
uint32_t temp=0;
TaskStatus_t taskStatusArray[4];
uint32_t taskcpuarr[4]={0,};
uint32_t usage =0;
uint32_t Memory_TASK_priority =0;
uint32_t Calc_TASK_priority =0;
uint32_t Datatx_TASK_priority =0;
uint8_t task_random=0;
uint32_t random_arr=0;
uint16_t normal_task_classification = 0;
uint8_t  memory_func_class = 0;
uint8_t  calc_func_class = 0;
uint8_t  gpio_func_class = 0;
uint32_t normal_random_parameter=0;
uint16_t func_number=0;
cell_info cell_err_buf[12]={0,};
cell_comparison tray_err_buf[17]={0,};
tray_comparison rack_err_buf;
uint8_t task_classification = 0;
uint8_t task_deadlock=0;
uint32_t task_count=0;
uint16_t testusage=0;

cell_info cell_buffer[12]={0,};
cell_comparison tray_buffer;

cell_comparison tray_rack_buffer[17]={0,};
tray_comparison rack_buffer;

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

static float f32_Asw_PiController(float f32_ref, float f32_in, Pi_Errors enp_error, Pi_Gains en_K);

uint8_t *memory_heap0;
uint8_t *memory_heap1;
uint8_t *memory_heap2;
uint8_t *memory_heap3;
uint8_t *memory_heap4;
uint8_t *memory_heap5;
uint8_t *memory_heap6;
uint8_t *memory_heap7;
uint8_t *memory_heap8;
uint8_t *memory_heap9;
uint8_t *memory_heap10;
uint8_t *memory_heap11;

uint8_t *memory_ptr0;
uint8_t *memory_ptr1;
uint8_t *memory_ptr2;
uint8_t *memory_ptr3;
uint8_t *memory_ptr4;
uint8_t *memory_ptr5;
uint32_t malloc_random =0;
uint8_t start_Test=0;
uint8_t malloc_abnormal=0;

#define KIY
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

osThreadId TaskHandle;
osThreadId TaskHandle1;
osThreadId TaskHandle2;
osThreadId TaskHandle3;
osThreadId TaskHandle4;

static uint32_t func_count=0;

void Monitor_TASK(void const * argument)//monitor
{
    while(1)
    {
    	task_random =(HAL_RNG_GetRandomNumber(&hrng)%7);
        usage = osGetCPUUsage();
        heapvalue_current = xPortGetFreeHeapSize();
		heapvalue_max = xPortGetMinimumEverFreeHeapSize();
		MCU_Temperature();

		TxDataTrans();
		osDelay(100);
    }
}

void Memory_TASK(void const * argument)
{
	while(1)
	{
//		if(task_count < Task_Count_Num)
//		{
		    task_deadlock = 0;
		    malloc_random=(HAL_RNG_GetRandomNumber(&hrng)%1000);
			Cell_Tray();
			Tray_Rack();

			memory_heap0 = pvPortMalloc(sizeof(uint8_t));
			memory_heap1 = pvPortMalloc(sizeof(uint8_t));
			memory_heap2 = pvPortMalloc(sizeof(uint8_t));
#if 1
		 if(malloc_random < 995 && malloc_random > 980  && task_count > 25000)
		{
				malloc_abnormal = 1;
				malloc_add();
				func_count++;
				osDelay(30000);

				malloc_add();
				func_count++;
				osDelay(30000);

				osDelay(30000);

				malloc_free();
				func_count--;
				osDelay(30000);

				task_count=(HAL_RNG_GetRandomNumber(&hrng)%24000);
		}
		else if(malloc_abnormal == 1)
		{
			malloc_abnormal=0;

			for(int i=0; i<func_count;i++)
			{
				//malloc_free();
			}
		}

#endif


#if 0
		 //임의로 걸어놓은 것
		normal_random_parameter = (((HAL_RNG_GetRandomNumber(&hrng)%100)+1)*100);

		if(normal_task_classification < 50 )
		{
			task_classification = 0;

			if(normal_task_classification<10)
			{
				for(int j=0; j<10; j++)
				{
					Memory_func(normal_random_parameter);
					GPIO_func(normal_random_parameter);
				}
			}
			else if(normal_task_classification>=10 && normal_task_classification<20)
			{
				for(int j=0; j<13; j++)
				{
					Memory_func(normal_random_parameter);
					GPIO_func(normal_random_parameter);
				}
			}
			else if(normal_task_classification >=20 && normal_task_classification<30)
			{
				for(int j=0; j<16; j++)
				{
					Memory_func(normal_random_parameter);
					GPIO_func(normal_random_parameter);
				}
			}
			else if(normal_task_classification>=30 && normal_task_classification <32)
			{
				for(int k=0;k<5; k++)
				{
					Calc_func(normal_random_parameter);
				}
			}
			else if(normal_task_classification>=33 && normal_task_classification <50)
			{
				for(int j=0; j<16; j++)
				{
					Memory_func(normal_random_parameter);
					GPIO_func(normal_random_parameter);
				}
			}
			normal_task_classification++;
		}
		else if(normal_task_classification >= 50)
		{
			//task_classification = 0;
			task_classification = 1;

			if(normal_task_classification<60)
			{
				for(int j=0; j<28; j++)
				{
					Memory_func(normal_random_parameter);
					GPIO_func(normal_random_parameter);
				}
			}
			else if(normal_task_classification>=60 && normal_task_classification<70)
			{
				if((normal_task_classification%2)==0)
				{
					for(int j=0; j<36; j++)
					{
						Memory_func(normal_random_parameter);
						GPIO_func(normal_random_parameter);
					}
				}
			}
			else if(normal_task_classification >=70 && normal_task_classification<80)
			{
				for(int j=0; j<42; j++)
				{
					Memory_func(normal_random_parameter);
					GPIO_func(normal_random_parameter);
				}
			}
			else if(normal_task_classification>=80 && normal_task_classification <82)
			{
				for(int k=0;k<5; k++)
				{
					Calc_func(normal_random_parameter);
				}
			}
			else if(normal_task_classification>=83 && normal_task_classification <100)
			{
				for(int j=0; j<61; j++)
				{
					Memory_func(normal_random_parameter);
					GPIO_func(normal_random_parameter);
				}
			}
			normal_task_classification++;
			if(normal_task_classification == 100)
			{
				normal_task_classification=0;
			}
		}
#endif
#if 1
		if(malloc_random > Task_Count_Num && task_count > 25000)
		{
			task_deadlock = 1;
			for(int i=0; i<Deadlock_On; i++)
			{
				//usage = osGetCPUUsage();
				//nothing
			}
			task_count=(HAL_RNG_GetRandomNumber(&hrng)%24000);
		}
#endif
		normal_task_classification++;
		task_count++;

		vTaskPrioritySet(TaskHandle1, task_random);
		Memory_TASK_priority = uxTaskPriorityGet(TaskHandle1);

		vPortFree((uint8_t*)memory_heap0);
		vPortFree((uint8_t*)memory_heap1);
		vPortFree((uint8_t*)memory_heap2);
		osDelay(100);
	}

}
#if 1
static void Calc_TASK(void const * argument)
{
    while(1)
    {
 #if 1
    	normal_random_parameter=(HAL_RNG_GetRandomNumber(&hrng)%1000);

    	memory_heap3 = pvPortMalloc(sizeof(uint8_t));
    	memory_heap4 = pvPortMalloc(sizeof(uint8_t));
    	memory_heap5 = pvPortMalloc(sizeof(uint8_t));

    		task_deadlock = 0;
			if(normal_task_classification <= 25000)
			{
				task_classification = 0;
				for(int j=0; j<3000;j++)
				{
					for(int i=0; i<12;i++)
					{
							cell_voltage_difference = ((cell_buffer[i].CELL_VOLTAGE - cell_buffer[i-1].CELL_VOLTAGE) * 0.001);
							cell_temperature_difference = (cell_buffer[i].CELL_temperature - cell_buffer[i-1].CELL_temperature);
							current_voltage = cell_buffer[i].CELL_VOLTAGE;

							soc = (current_voltage - min_voltage) / (max_voltage - min_voltage) * (max_soc - min_soc) + min_soc;
							soc = cell_err_buf[i].CELL_SOC;
					}
				}
			}
			else if(normal_task_classification > 25000 &&normal_random_parameter > 900)
			{
				task_classification = 1;
				for(int h=0; h<10; h++)
				{
					for(int j=0; j<6000;j++)
					{
						for(int i=0; i<12;i++)
						{
								cell_voltage_difference = ((cell_buffer[i].CELL_VOLTAGE - cell_buffer[i-1].CELL_VOLTAGE) * 0.001);
								cell_temperature_difference = (cell_buffer[i].CELL_temperature - cell_buffer[i-1].CELL_temperature);
								current_voltage = cell_buffer[i].CELL_VOLTAGE;

								soc = (current_voltage - min_voltage) / (max_voltage - min_voltage) * (max_soc - min_soc) + min_soc;
								soc = cell_err_buf[i].CELL_SOC;
						}
					}
					//usage = osGetCPUUsage();
				}
				normal_task_classification = 0;
			}
#if 1
    	if(malloc_random >= Task_Count_Num && task_count > 25000)
		{
    		task_deadlock = 1;
    		for(int i=0; i<Deadlock_On; i++)
			{
    			//usage = osGetCPUUsage();
				//nothing
			}
    		task_count=(HAL_RNG_GetRandomNumber(&hrng)%24000);
		}
#endif
#if 0
		if(normal_task_classification < 50 )
		{
			task_classification = 0;

			if(normal_task_classification<10)
			{
				for(int j=0; j<10; j++)
				{
					Memory_func(normal_random_parameter);
					GPIO_func(normal_random_parameter);
				}
			}
			else if(normal_task_classification>=10 && normal_task_classification<20)
			{
				for(int j=0; j<13; j++)
				{
					Memory_func(normal_random_parameter);
					GPIO_func(normal_random_parameter);
				}
			}
			else if(normal_task_classification >=20 && normal_task_classification<30)
			{
				for(int j=0; j<16; j++)
				{
					Memory_func(normal_random_parameter);
					GPIO_func(normal_random_parameter);
				}
			}
			else if(normal_task_classification>=30 && normal_task_classification <32)
			{
				for(int k=0;k<5; k++)
				{
					Calc_func(normal_random_parameter);
				}
			}
			else if(normal_task_classification>=33 && normal_task_classification <50)
			{
				for(int j=0; j<16; j++)
				{
					Memory_func(normal_random_parameter);
					GPIO_func(normal_random_parameter);
				}
			}
			normal_task_classification++;
		}
		else if(normal_task_classification >= 50)
		{
			task_classification = 1;

			if(normal_task_classification<60)
			{
				for(int j=0; j<28; j++)
				{
					Memory_func(normal_random_parameter);
					GPIO_func(normal_random_parameter);
				}
			}
			else if(normal_task_classification>=60 && normal_task_classification<70)
			{
				if((normal_task_classification%2)==0)
				{
					for(int j=0; j<36; j++)
					{
						Memory_func(normal_random_parameter);
						GPIO_func(normal_random_parameter);
					}
				}
			}
			else if(normal_task_classification >=70 && normal_task_classification<80)
			{
				for(int j=0; j<42; j++)
				{
					Memory_func(normal_random_parameter);
					GPIO_func(normal_random_parameter);
				}
			}
			else if(normal_task_classification>=80 && normal_task_classification <82)
			{
				for(int k=0;k<5; k++)
				{
					Calc_func(normal_random_parameter);
				}
			}
			else if(normal_task_classification>=83 && normal_task_classification <100)
			{
				for(int j=0; j<61; j++)
				{
					Memory_func(normal_random_parameter);
					GPIO_func(normal_random_parameter);
				}
			}
			normal_task_classification++;
			if(normal_task_classification == 100)
			{
				normal_task_classification=0;
			}
		}


		normal_task_classification++;
		task_count++;
    	vTaskPrioritySet(TaskHandle2, task_random);
    	Calc_TASK_priority = uxTaskPriorityGet(TaskHandle2);
#endif
#endif
		vTaskPrioritySet(TaskHandle1, task_random);
		Memory_TASK_priority = uxTaskPriorityGet(TaskHandle2);
		task_count++;

		vPortFree((uint8_t*)memory_heap3);
		vPortFree((uint8_t*)memory_heap4);
		vPortFree((uint8_t*)memory_heap5);


    	osDelay(100);
    }
}
#endif
void Datatx_TASK(void const * argument)
{
	while(1)
	{

#if 1
	  	memory_heap6 = pvPortMalloc(sizeof(uint8_t));
		memory_heap7 = pvPortMalloc(sizeof(uint8_t));
		memory_heap8 = pvPortMalloc(sizeof(uint8_t));

		task_deadlock = 0;

		for(int i=0; i<12;i++)
		{
			if(i > 0)
			{
				if(cell_buffer[i].CELL_VOLTAGE >= 4.086 && cell_buffer[i].CELL_VOLTAGE < 4.1)
				{
					cell_err_buf[i].CELL_VOLTAGE = Cell_Overcharge_Error;
				}
				else if(cell_buffer[i].CELL_VOLTAGE >= 4.1)
				{
					cell_err_buf[i].CELL_VOLTAGE = Cell_Overcharge_Emergency_Stop;
					HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_12);
				}
				if(cell_buffer[i].CELL_VOLTAGE <= 3.15 && cell_buffer[i].CELL_VOLTAGE > 2.9)
				{
					cell_err_buf[i].CELL_VOLTAGE = Cell_Overdischarge_Error;
				}
				else if(cell_buffer[i].CELL_VOLTAGE <= 2.9)
				{
					cell_err_buf[i].CELL_VOLTAGE = Cell_Overdischarge_Emergency_Stop;
					HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_13);
				}
				if(cell_buffer[i].CELL_VOLTAGE >= 4.086 && cell_buffer[i].CELL_VOLTAGE < 4.1)
				{
					cell_err_buf[i].CELL_VOLTAGE = Cell_Over_Voltage_Error;
				}
				else if(cell_buffer[i].CELL_VOLTAGE >= 4.1)
				{
					cell_err_buf[i].CELL_VOLTAGE = Cell_Over_Voltage_Emergency_Stop;
					HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_14);
				}
				if(cell_voltage_difference >= 300 && cell_voltage_difference < 500)
				{
					cell_err_buf[i].CELL_VOLTAGE_difference = Cell_Voltage_Unbalance_Error;
					HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_14);
				}
				else if(cell_voltage_difference >= 500)
				{
					cell_err_buf[i].CELL_VOLTAGE_difference = Cell_Voltage_Unbalance_Emergency_Stop;
					HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_15);
				}
				if(cell_temperature_difference >= 20 && cell_temperature_difference < 25 )
				{
					cell_err_buf[i].CELL_temperature_difference = Cell_Temperature_Unbalance_Error;
				}
				else if(cell_temperature_difference >= 25 )
				{
					cell_err_buf[i].CELL_temperature_difference = Cell_Temperature_Unbalance_Emergency_Stop;
					HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_12);
					HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_13);
				}
			}
		}

			//tray error
			for(int i=0; i<17;i++)
			{
				if(i > 0)
				{
					tray_temperature_difference = (tray_rack_buffer[i].TRAY_MAX_CELL_temperature - tray_rack_buffer[i-1].TRAY_MIN_CELL_temperature);
					if(tray_temperature_difference >= 50 & tray_temperature_difference < 55)
					{

						tray_err_buf[i].TRAY_temperature_difference = Tray_Temperature_Unbalance_error;
					}
					else if(tray_temperature_difference >= 55)
					{
						tray_err_buf[i].TRAY_temperature_difference = Tray_Temperature_Unbalance_Emergency_Stop;
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

			for(int i=0; i<12;i++)
			{
				printf("%.2f ",cell_err_buf[i].CELL_VOLTAGE);
				printf("%.2f ",cell_err_buf[i].CELL_VOLTAGE_difference);
				printf("%.2f",cell_err_buf[i].CELL_temperature_difference);
				printf("%.2f\n\n",cell_err_buf[i].CELL_SOC);
			}
			for(int i=0; i<17;i++)
			{
				printf("%.2f\n\n",tray_err_buf[i].TRAY_temperature_difference);

			}
			printf("%.2f\n\n",rack_err_buf.RACK_current_difference);
#endif
//	    }
#if 1
    	if(malloc_random >= Task_Count_Num && task_count > 25000)
		{
    		task_deadlock = 1;
			for(int i=0; i<Deadlock_On; i++)
			{
				//usage = osGetCPUUsage();
				//nothing
			}
			task_count=(HAL_RNG_GetRandomNumber(&hrng)%24000);
		}
#endif
		normal_task_classification++;
		task_count++;
		vTaskPrioritySet(TaskHandle2, task_random);
		Calc_TASK_priority = uxTaskPriorityGet(TaskHandle3);

		vPortFree((uint8_t*)memory_heap6);
		vPortFree((uint8_t*)memory_heap7);
		vPortFree((uint8_t*)memory_heap8);
		osDelay(100);
	}

}

void TxDataTrans()
{
	memset(tx_data, 0, sizeof(tx_data));

	tx_data[0] = testusage >> 8 ;
	tx_data[1] = testusage;
	tx_data[2] = stack_value >> 8;
	tx_data[3] = stack_value;
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

	for(int i=1; i<11;i++)
	{
		if(tray_buffer.TRAY_MIN_CELL_temperature >= cell_buffer[i].CELL_temperature)
		{
			tray_buffer.TRAY_MIN_CELL_temperature = cell_buffer[i].CELL_temperature;
			tray_buffer.TRAY_MIN_CELL_temperature_POSITION = cell_buffer[i].CELL_ID;
		}

		if(tray_buffer.TRAY_MAX_CELL_temperature <= cell_buffer[i].CELL_temperature)
		{
			tray_buffer.TRAY_MAX_CELL_temperature = cell_buffer[i].CELL_temperature;
			tray_buffer.TRAY_MAX_CELL_temperature_POSITION = cell_buffer[i].CELL_ID;
		}

		if(tray_buffer.TRAY_MIN_CELL_VOLTAGE >= cell_buffer[i].CELL_VOLTAGE)
		{
			tray_buffer.TRAY_MIN_CELL_VOLTAGE = cell_buffer[i].CELL_VOLTAGE;
			tray_buffer.TRAY_MIN_CELL_VOLTAGE_POSITION = cell_buffer[i].CELL_ID;
		}

		if(tray_buffer.TRAY_MAX_CELL_VOLTAGE <= cell_buffer[i].CELL_VOLTAGE)
		{
			tray_buffer.TRAY_MAX_CELL_VOLTAGE = cell_buffer[i].CELL_VOLTAGE;
			tray_buffer.TRAY_MAX_CELL_VOLTAGE_POSITION = cell_buffer[i].CELL_ID;
		}
		if(tray_buffer.TRAY_MIN_CELL_CURRENT >= cell_buffer[i].CELL_CURRENT)
		{
			tray_buffer.TRAY_MIN_CELL_CURRENT = cell_buffer[i].CELL_CURRENT;
			tray_buffer.TRAY_MIN_CELL_CURRENT_POSITION = cell_buffer[i].CELL_ID;
		}

		if(tray_buffer.TRAY_MAX_CELL_CURRENT <= cell_buffer[i].CELL_CURRENT)
		{
			tray_buffer.TRAY_MAX_CELL_CURRENT = cell_buffer[i].CELL_CURRENT;
			tray_buffer.TRAY_MAX_CELL_CURRENT_POSITION = cell_buffer[i].CELL_ID;
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

	for(int i=1; i<16;i++)
	{
		tray_rack_buffer[i] = Cell_Tray();

		if(rack_buffer.RACK_MIN_CELL_temperature >= tray_rack_buffer[i].TRAY_MIN_CELL_temperature)
		{
			rack_buffer.RACK_MIN_CELL_temperature = tray_rack_buffer[i].TRAY_MIN_CELL_temperature;
			rack_buffer.RACK_MIN_CELL_temperature_POSITION = tray_rack_buffer[i].TRAY_MIN_CELL_temperature_POSITION;
		}

		if(rack_buffer.RACK_MAX_CELL_temperature <= tray_rack_buffer[i].TRAY_MAX_CELL_temperature)
		{
			rack_buffer.RACK_MAX_CELL_temperature = tray_rack_buffer[i].TRAY_MAX_CELL_temperature;
			rack_buffer.RACK_MAX_CELL_temperature_POSITION = tray_rack_buffer[i].TRAY_MAX_CELL_temperature_POSITION;
		}

		if(rack_buffer.RACK_MIN_CELL_VOLTAGE >= tray_rack_buffer[i].TRAY_MIN_CELL_VOLTAGE)
		{
			rack_buffer.RACK_MIN_CELL_VOLTAGE = tray_rack_buffer[i].TRAY_MIN_CELL_VOLTAGE;
			rack_buffer.RACK_MIN_CELL_VOLTAGE_POSITION = tray_rack_buffer[i].TRAY_MIN_CELL_VOLTAGE_POSITION;
		}

		if(rack_buffer.RACK_MAX_CELL_VOLTAGE <= tray_rack_buffer[i].TRAY_MAX_CELL_VOLTAGE)
		{
			rack_buffer.RACK_MAX_CELL_VOLTAGE = tray_rack_buffer[i].TRAY_MAX_CELL_VOLTAGE;
			rack_buffer.RACK_MAX_CELL_VOLTAGE_POSITION = tray_rack_buffer[i].TRAY_MAX_CELL_VOLTAGE_POSITION;
		}
		if(rack_buffer.RACK_MIN_CELL_CURRENT >= tray_rack_buffer[i].TRAY_MIN_CELL_CURRENT)
		{
			rack_buffer.RACK_MIN_CELL_CURRENT = tray_rack_buffer[i].TRAY_MIN_CELL_CURRENT;
			rack_buffer.RACK_MIN_CELL_CURRENT_POSITION = tray_rack_buffer[i].TRAY_MIN_CELL_CURRENT_POSITION;
		}

		if(rack_buffer.RACK_MAX_CELL_CURRENT <= tray_rack_buffer[i].TRAY_MAX_CELL_CURRENT)
		{
			rack_buffer.RACK_MAX_CELL_CURRENT = tray_rack_buffer[i].TRAY_MAX_CELL_CURRENT;
			rack_buffer.RACK_MAX_CELL_CURRENT_POSITION = tray_rack_buffer[i].TRAY_MAX_CELL_CURRENT_POSITION;
		}
	}
	return rack_buffer;
}

static void malloc_func1()
{

//double free test
#if 0
	uint8_t *memory_ptr0;
    static uint16_t	malloc_memoryflag=0;

	if(malloc_memoryflag == 0)
	{
		memory_ptr0 = pvPortMalloc(sizeof(uint8_t));

		heapvalue_current = xPortGetFreeHeapSize();
	}
	else if(malloc_memoryflag == 1000)
	{
		vPortFree((uint8_t*)memory_ptr0);
		heapvalue_current = xPortGetFreeHeapSize();
	}
	else if(malloc_memoryflag == 2500)
	{
		vPortFree((uint8_t*)memory_ptr0);
		heapvalue_current = xPortGetFreeHeapSize();
		memory_flag1=0;
	}
	memory_flag1++;
#endif

//Memory_Free_On_Stack
#if 0
	uint8_t memory_ptr0[100]={0,};
    static uint16_t	malloc_memoryflag=0;

    if(malloc_memoryflag == 1000)
	{
		vPortFree((uint8_t*)memory_ptr0);
		heapvalue_current = xPortGetFreeHeapSize();
	}
    malloc_memoryflag++;
#endif
//Mismatched_Memory
#if 0
	uint8_t memory_ptr0;
	static uint16_t	malloc_memoryflag=0;

	if(malloc_memoryflag == 1000)
	{
		delete((uint8_t*)memory_ptr0);
		heapvalue_current = xPortGetFreeHeapSize();
	}
	malloc_memoryflag++;
#endif

}
static void malloc_add()
{
//	if(malloc_memoryflag == 0)
//	{
		memory_ptr0 = pvPortMalloc(sizeof(uint8_t));
		heapvalue_current = xPortGetFreeHeapSize();

		for(int i=0; i<1000000; i++)
		{

		}
		memory_ptr1 = pvPortMalloc(sizeof(uint8_t));
		heapvalue_current = xPortGetFreeHeapSize();

		for(int i=0; i<1000000; i++)
		{

		}
		memory_ptr2 = pvPortMalloc(sizeof(uint8_t));
		heapvalue_current = xPortGetFreeHeapSize();

		for(int i=0; i<1000000; i++)
		{

		}
		memory_ptr3 = pvPortMalloc(sizeof(uint8_t));
		heapvalue_current = xPortGetFreeHeapSize();

		for(int i=0; i<1000000; i++)
		{

		}
		memory_ptr4 = pvPortMalloc(sizeof(uint8_t));
		heapvalue_current = xPortGetFreeHeapSize();
		for(int i=0; i<1000000; i++)
		{

		}
	//}
}
static void malloc_free()
{
	vPortFree((uint8_t*)memory_ptr0);
	heapvalue_current = xPortGetFreeHeapSize();
	for(int i=0; i<1000000; i++)
	{

	}
	vPortFree((uint8_t*)memory_ptr1);
	heapvalue_current = xPortGetFreeHeapSize();
	for(int i=0; i<1000000; i++)
	{

	}
	vPortFree((uint8_t*)memory_ptr2);
	heapvalue_current = xPortGetFreeHeapSize();
	for(int i=0; i<1000000; i++)
	{

	}
	vPortFree((uint8_t*)memory_ptr3);
	heapvalue_current = xPortGetFreeHeapSize();
	for(int i=0; i<1000000; i++)
	{

	}
	vPortFree((uint8_t*)memory_ptr4);
	heapvalue_current = xPortGetFreeHeapSize();
	for(int i=0; i<1000000; i++)
	{

	}
}
static void func1()
{
	memory_heap0 = pvPortMalloc(sizeof(uint8_t));
	heapvalue_current = xPortGetFreeHeapSize();
	//TxDataTrans();
	for(int i=0; i<1000000;i++)
	{

	}
	func2();
}
static void func2()
{
	memory_heap1 = pvPortMalloc(sizeof(uint8_t));
	heapvalue_current = xPortGetFreeHeapSize();
	//TxDataTrans();

	vPortFree((uint8_t*)memory_heap0);
	heapvalue_current = xPortGetFreeHeapSize();
	//TxDataTrans();

	func3();
}
static void func3()
{
	memory_heap2 = pvPortMalloc(sizeof(uint8_t));
	heapvalue_current = xPortGetFreeHeapSize();
	//TxDataTrans();

	vPortFree((uint8_t*)memory_heap1);
	heapvalue_current = xPortGetFreeHeapSize();
	//TxDataTrans();

	//func4();
}
static void func4()
{
	memory_heap3 = pvPortMalloc(sizeof(uint8_t));
	heapvalue_current = xPortGetFreeHeapSize();
	//TxDataTrans();

	vPortFree((uint8_t*)memory_heap2);
	heapvalue_current = xPortGetFreeHeapSize();
	//TxDataTrans();

	func5();
}
static void func5()
{
	memory_heap4 = pvPortMalloc(sizeof(uint8_t));
	heapvalue_current = xPortGetFreeHeapSize();
	//TxDataTrans();

	vPortFree((uint8_t*)memory_heap3);
	heapvalue_current = xPortGetFreeHeapSize();
	//TxDataTrans();

	//func6();
}
static void func6()
{
	vPortFree((uint8_t*)memory_heap4);
	heapvalue_current = xPortGetFreeHeapSize();
	TxDataTrans();

	memory_heap5 = pvPortMalloc(sizeof(uint8_t));
	heapvalue_current = xPortGetFreeHeapSize();
	TxDataTrans();

	vPortFree((uint8_t*)memory_heap5);
	heapvalue_current = xPortGetFreeHeapSize();
	TxDataTrans();
}
static void Cell_create(cell_info cell_buffer[])
{
	for(int i=0; i<12;i++)
	{
		cell_buffer[i].CELL_ID = i+1;
		cell_buffer[i].CELL_temperature =(HAL_RNG_GetRandomNumber(&hrng)%100);
		cell_buffer[i].CELL_VOLTAGE = (HAL_RNG_GetRandomNumber(&hrng)%5);
		cell_buffer[i].CELL_CURRENT = (HAL_RNG_GetRandomNumber(&hrng)%300);
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

static float f32_Asw_PiController(float f32_ref, float f32_in, Pi_Errors enp_error, Pi_Gains en_K)
{
	float out = 0.f;
	en_K.P = HAL_RNG_GetRandomNumber(&hrng);
	en_K.I = HAL_RNG_GetRandomNumber(&hrng);

	enp_error.error = f32_ref - f32_in;
	enp_error.sum += enp_error.error;
	out = (en_K.P * enp_error.error) + (en_K.I * enp_error.sum);

	return out;
}

void Memory_func(uint32_t memory_size)
{
	func_number = 1;
	static uint8_t memory_flag = 0;
	static uint8_t *memory_ptr;

	if(memory_func_class == 0)
	{
		if(memory_flag == 0)
		{
			memory_ptr = pvPortMalloc(sizeof(uint8_t)*memory_size);
			memory_flag = 1;
		}
		else if(memory_flag == 1)
		{
			vPortFree((uint8_t*)memory_ptr);
			memory_flag = 0;
		}
	}
}
void Calc_func(uint32_t calc_parameter)
{

    static uint32_t KIY_test_cnt2 = 0;
    static uint32_t KIY_test_cnt3 = 0;
	float random_test=0.f;

	if(calc_func_class == 0)
	{
		random_test = ((HAL_RNG_GetRandomNumber(&hrng)%3)+1)*10;
		random_arr = (HAL_RNG_GetRandomNumber(&hrng)%20);

		volatile float arr[random_arr];

		for(int i=0; i<random_arr; i++)
		{
			arr[i] = random_arr;
		}

		for(int i=0;i<calc_parameter;i++)
		{
			f32_Asw_PiController(random_test,(random_test+1),error_test,gain_test);
		}
	}
}

void GPIO_func(uint32_t gpio_parameter)
{
	func_number = 3;

	if(gpio_func_class == 0)
	{
		for(int i=0;i<gpio_parameter;i++)
		{
			HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_12);
			HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_13);
			HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_14);
			HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_15);
		}
	}
}

#if 0
void measureTaskCPUUsage()
{
    static const TickType_t measurementPeriod = pdMS_TO_TICKS(1000);
    static TickType_t lastMeasurementTime = 0;
    //unsigned portBASE_TYPE uxArraySize1 = 4;

    unsigned portBASE_TYPE uxArraySize1=uxTaskGetNumberOfTasks();

    TaskStatus_t taskStatusArray[uxArraySize1];

    if (xTaskGetTickCount() - lastMeasurementTime >= measurementPeriod)
    {
        lastMeasurementTime = xTaskGetTickCount();

        uint32_t totalRunTime;
        const unsigned portBASE_TYPE actualTasks = uxTaskGetSystemState(&taskStatusArray[0], uxArraySize1, &totalRunTime);//

        for (unsigned portBASE_TYPE i = 0; i < actualTasks; i++)
        {
            const TaskStatus_t *taskStatus = &taskStatusArray[i];

            uint32_t taskRunTime = taskStatus->ulRunTimeCounter;
            uint32_t taskRunCount = taskStatus->ulRunTimeCounter / portTICK_PERIOD_MS;

            float cpuUsage = (float)(taskRunTime * 100) / (totalRunTime * taskRunCount);

            //taskcpuarr[i]=(float)(taskRunTime * 100) / (totalRunTime * taskRunCount);
        }
    }
}
#endif



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

  osThreadDef(KiyTask, Monitor_TASK, osPriorityHigh, 0, 256);
  TaskHandle = osThreadCreate(osThread(KiyTask), NULL);

  osThreadDef(KiyTask1,Memory_TASK, osPriorityNormal, 0, 512);
  TaskHandle1 = osThreadCreate(osThread(KiyTask1), NULL);

#if 1
  osThreadDef(KiyTask2, Calc_TASK, osPriorityNormal, 0, 256);
  TaskHandle2 = osThreadCreate(osThread(KiyTask2), NULL);

  osThreadDef(KiyTask3, Datatx_TASK, osPriorityNormal, 0, 256);
  TaskHandle3 = osThreadCreate(osThread(KiyTask3), NULL);
#if 0
  osThreadDef(KiyTask4, abnormal_TASK, osPriorityNormal, 0, 150);
  TaskHandle4 = osThreadCreate(osThread(KiyTask4), NULL);
#endif

#endif

  while(1)
  {
      osDelay(100);
  }
}
