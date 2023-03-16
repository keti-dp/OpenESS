// 더미 데이터 이상치 판단 시나리오

#include <stdio.h>
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"

/* task's priority */
#define TASK_MAIN3_PRIO	20
#define TASK_5_PRIO		13
#define TASK_6_PRIO		14

/* The task functions. */
void TaskMain3( void *pvParameters );
void Task5( void *pvParameters );
void Task6( void *pvParameters);

TaskHandle_t xHandleMain3, xHandle5, xHandle6;

/* ...........................................................................
 *
 * 메시지큐 & 사용자 정의 블럭 정의
 * ===================
 */
QueueHandle_t qid3;

typedef struct tag_qBuffer
{

	float RACK_MIN_CELL_TEMPERATURE;
	float RACK_MIN_CELL_TEMPERATURE_POSITION; //1~40
	float RACK_MAX_CELL_TEMPERATURE;
	float RACK_MAX_CELL_TEMPERATURE_POSITION; //1~40

	float RACK_MIN_CELL_VOLTAGE;
	float RACK_MIN_CELL_VOLTAGE_POSITION; //1~240
	float RACK_MAX_CELL_VOLTAGE;
	float RACK_MAX_CELL_VOLTAGE_POSITION; //1~240

	float RACK_VOLTAGE;
	float RACK_CURRENT;
	float RACK_SOC;//0~100

}qBuffer;

#define QUEUE_LENGTH	5
#define QUEUE_ITEM_SIZE sizeof(qBuffer)

/*-----------------------------------------------------------*/

void Scenario_3( void )
{
	//prvSetupHardware();
#ifdef CMSIS_OS
	osThreadDef(defaultTask, TaskMain, osPriorityNormal, 0, 256);
	defaultTaskHandle = osThreadCreate(osThread(defaultTask), NULL);
#else

	xTaskCreate(	(TaskFunction_t)TaskMain3,
					"TaskMain3",
					1024,
					NULL,
					TASK_MAIN3_PRIO,
					&xHandleMain3 );
#endif
}
/*-----------------------------------------------------------*/

void TaskMain3( void *pvParameters )
{
	const char *pcTaskName = "TaskMain3";

//create a Queue
#if 1
	qid3 = xQueueCreate(QUEUE_LENGTH, QUEUE_ITEM_SIZE);
if (qid3 == NULL)
	printf("xQueueCreate error found\n");
#endif

xTaskCreate((TaskFunction_t)Task5,
				"Task5",
				1026,
				NULL,
				TASK_5_PRIO,
				&xHandle5);

xTaskCreate((TaskFunction_t)Task6,
				"Task6",
				1026,
				NULL,
				TASK_6_PRIO,
				&xHandle6);

	printf( "\n**************SCENARIO_3**************\n");

	/* Print out the name of this task. */
	printf( "%s is running\r\n", pcTaskName );

	/* delete self task */
	/* Print out the name of this task. */
	printf( "%s is deleted\r\n\n", pcTaskName );

	vTaskDelete (xHandleMain3);
}

/*-----------------------------------------------------------*/

void Task5( void *pvParameters )
{
	const char *pcTaskName = "Task5";

	qBuffer RxBuffer[20];
	qBuffer Outlier[20];
	int i;
	int index = 0;

	vTaskDelay (pdMS_TO_TICKS (1500));
	printf( "%s is running\n\n", pcTaskName );

	for(i=0; i<sizeof(RxBuffer)/sizeof(qBuffer); i++)
	{
		if(xQueueReceive( qid3, &RxBuffer[i],portMAX_DELAY) == pdPASS)
		{
			//SOC 이상
			if(RxBuffer[i].RACK_SOC >= 90)
			{
				Outlier[index].RACK_MIN_CELL_TEMPERATURE = RxBuffer[i].RACK_MIN_CELL_TEMPERATURE;
				Outlier[index].RACK_MIN_CELL_TEMPERATURE_POSITION = RxBuffer[i].RACK_MIN_CELL_TEMPERATURE_POSITION;
				Outlier[index].RACK_MAX_CELL_TEMPERATURE = RxBuffer[i].RACK_MAX_CELL_TEMPERATURE;
				Outlier[index].RACK_MAX_CELL_TEMPERATURE_POSITION = RxBuffer[i].RACK_MAX_CELL_TEMPERATURE_POSITION;
				Outlier[index].RACK_MIN_CELL_VOLTAGE = RxBuffer[i].RACK_MIN_CELL_VOLTAGE;
				Outlier[index].RACK_MIN_CELL_VOLTAGE_POSITION = RxBuffer[i].RACK_MIN_CELL_VOLTAGE_POSITION;
				Outlier[index].RACK_MAX_CELL_VOLTAGE = RxBuffer[i].RACK_MAX_CELL_VOLTAGE;
				Outlier[index].RACK_MAX_CELL_VOLTAGE_POSITION = RxBuffer[i].RACK_MAX_CELL_VOLTAGE_POSITION;
				Outlier[index].RACK_VOLTAGE = RxBuffer[i].RACK_VOLTAGE;
				Outlier[index].RACK_CURRENT = RxBuffer[i].RACK_CURRENT;
				Outlier[index].RACK_SOC = RxBuffer[i].RACK_SOC;

				printf("[Task5] %d : SOC_ERROR \%.3f\n",index,Outlier[index].RACK_SOC);
				index++;
				continue;
			}

			//과충전
			if(RxBuffer[i].RACK_MAX_CELL_VOLTAGE > 4.086)
			{
				Outlier[index].RACK_MIN_CELL_TEMPERATURE = RxBuffer[i].RACK_MIN_CELL_TEMPERATURE;
				Outlier[index].RACK_MIN_CELL_TEMPERATURE_POSITION = RxBuffer[i].RACK_MIN_CELL_TEMPERATURE_POSITION;
				Outlier[index].RACK_MAX_CELL_TEMPERATURE = RxBuffer[i].RACK_MAX_CELL_TEMPERATURE;
				Outlier[index].RACK_MAX_CELL_TEMPERATURE_POSITION = RxBuffer[i].RACK_MAX_CELL_TEMPERATURE_POSITION;
				Outlier[index].RACK_MIN_CELL_VOLTAGE = RxBuffer[i].RACK_MIN_CELL_VOLTAGE;
				Outlier[index].RACK_MIN_CELL_VOLTAGE_POSITION = RxBuffer[i].RACK_MIN_CELL_VOLTAGE_POSITION;
				Outlier[index].RACK_MAX_CELL_VOLTAGE = RxBuffer[i].RACK_MAX_CELL_VOLTAGE;
				Outlier[index].RACK_MAX_CELL_VOLTAGE_POSITION = RxBuffer[i].RACK_MAX_CELL_VOLTAGE_POSITION;
				Outlier[index].RACK_VOLTAGE = RxBuffer[i].RACK_VOLTAGE;
				Outlier[index].RACK_CURRENT = RxBuffer[i].RACK_CURRENT;
				Outlier[index].RACK_SOC = RxBuffer[i].RACK_SOC;

				printf("[Task5] %d : OVER_CHARGE \%.3f\n",index,Outlier[index].RACK_MAX_CELL_VOLTAGE);
				index++;
				continue;
			}

			//과방전
			if(RxBuffer[i].RACK_MIN_CELL_VOLTAGE < 3.15)
			{
				Outlier[index].RACK_MIN_CELL_TEMPERATURE = RxBuffer[i].RACK_MIN_CELL_TEMPERATURE;
				Outlier[index].RACK_MIN_CELL_TEMPERATURE_POSITION = RxBuffer[i].RACK_MIN_CELL_TEMPERATURE_POSITION;
				Outlier[index].RACK_MAX_CELL_TEMPERATURE = RxBuffer[i].RACK_MAX_CELL_TEMPERATURE;
				Outlier[index].RACK_MAX_CELL_TEMPERATURE_POSITION = RxBuffer[i].RACK_MAX_CELL_TEMPERATURE_POSITION;
				Outlier[index].RACK_MIN_CELL_VOLTAGE = RxBuffer[i].RACK_MIN_CELL_VOLTAGE;
				Outlier[index].RACK_MIN_CELL_VOLTAGE_POSITION = RxBuffer[i].RACK_MIN_CELL_VOLTAGE_POSITION;
				Outlier[index].RACK_MAX_CELL_VOLTAGE = RxBuffer[i].RACK_MAX_CELL_VOLTAGE;
				Outlier[index].RACK_MAX_CELL_VOLTAGE_POSITION = RxBuffer[i].RACK_MAX_CELL_VOLTAGE_POSITION;
				Outlier[index].RACK_VOLTAGE = RxBuffer[i].RACK_VOLTAGE;
				Outlier[index].RACK_CURRENT = RxBuffer[i].RACK_CURRENT;
				Outlier[index].RACK_SOC = RxBuffer[i].RACK_SOC;

				printf("[Task5] %d : OVER_DISCHARGE  \%.3f\n",index,Outlier[index].RACK_MIN_CELL_VOLTAGE);
				index++;
				continue;
			}

			//과전압
			if(RxBuffer[i].RACK_MAX_CELL_VOLTAGE > 4.086 || RxBuffer[i].RACK_MIN_CELL_VOLTAGE > 4.086)
			{
				Outlier[index].RACK_MIN_CELL_TEMPERATURE = RxBuffer[i].RACK_MIN_CELL_TEMPERATURE;
				Outlier[index].RACK_MIN_CELL_TEMPERATURE_POSITION = RxBuffer[i].RACK_MIN_CELL_TEMPERATURE_POSITION;
				Outlier[index].RACK_MAX_CELL_TEMPERATURE = RxBuffer[i].RACK_MAX_CELL_TEMPERATURE;
				Outlier[index].RACK_MAX_CELL_TEMPERATURE_POSITION = RxBuffer[i].RACK_MAX_CELL_TEMPERATURE_POSITION;
				Outlier[index].RACK_MIN_CELL_VOLTAGE = RxBuffer[i].RACK_MIN_CELL_VOLTAGE;
				Outlier[index].RACK_MIN_CELL_VOLTAGE_POSITION = RxBuffer[i].RACK_MIN_CELL_VOLTAGE_POSITION;
				Outlier[index].RACK_MAX_CELL_VOLTAGE = RxBuffer[i].RACK_MAX_CELL_VOLTAGE;
				Outlier[index].RACK_MAX_CELL_VOLTAGE_POSITION = RxBuffer[i].RACK_MAX_CELL_VOLTAGE_POSITION;
				Outlier[index].RACK_VOLTAGE = RxBuffer[i].RACK_VOLTAGE;
				Outlier[index].RACK_CURRENT = RxBuffer[i].RACK_CURRENT;
				Outlier[index].RACK_SOC = RxBuffer[i].RACK_SOC;

				printf("[Task5] %d : OVER_VOLTAGE %.3f\n",index,Outlier[index].RACK_MIN_CELL_VOLTAGE);
				printf("[Task5] %d : OVER_VOLTAGE %.3f\n\n",index,Outlier[index].RACK_MAX_CELL_VOLTAGE);
				index++;
				continue;
			}
			else
			{
				printf("[Task5] %d : Outlier not found\n",index);
				index++;
			}
		}
		else
		{
			printf("xQueueReceive error found\ n");
		}
	}
	vTaskDelete (xHandle5);
}

/*-----------------------------------------------------------*/

void Task6( void *pvParameters)
{
	const char *pcTaskName = "Task6";

	//이상데이터 넣은 것
	qBuffer TxBuffer[20]=
	{
		{22.2,34,23.6,2,3.177,28,3.273,109,660.3,0,92}, //SOC 이상
		{20.5,34,23.6,2,3.177,28,4.087,109,660.3,0,0}, //MAX_CELL_VOL 이상 과충전
		{20.6,34,23.6,2,3.1,28,3.273,109,660.2,0,0}, //MIN_CELL_VOL 이상 과방전
		{20.7,34,23.6,2,4.7,28,3.273,109,660.2,0,0}, //MIN_CELL_VOL 이상 과전압
		{20.8,34,23.6,2,3.176,28,3.273,109,660.2,0,0},
		{20.9,34,23.6,2,3.177,28,3.273,109,660.2,0,0},
		{21.2,34,23.6,2,3.177,28,3.273,109,660.2,0,0},
		{24.2,34,23.6,2,3.177,28,3.273,109,660.2,0,0},
		{25.2,34,23.6,2,3.177,28,3.273,109,660.2,0,0},
		{22.2,34,23.6,2,3.176,28,3.273,109,660.2,0,0},
		{20.2,34,23.6,2,3.177,28,3.273,109,660.3,0,0},
		{20.5,34,23.6,2,3.177,28,3.273,109,660.3,0,0},
		{20.6,34,23.6,2,3.177,28,3.273,109,660.2,0,0},
		{20.7,34,23.6,2,3.177,28,3.273,109,660.2,0,0},
		{20.8,34,23.6,2,3.176,28,3.273,109,660.2,0,0},
		{10.3,34,23.6,2,3.176,28,3.273,109,660.2,0,0},
		{22.2,34,23.6,2,3.177,28,3.273,109,660.3,0,0},
		{20.5,34,23.6,2,3.177,28,3.273,109,660.3,0,0},
		{20.6,34,23.6,2,3.177,28,3.273,109,660.2,0,0},
		{20.7,34,23.6,2,3.177,28,3.273,109,660.2,0,0}
	};

	BaseType_t p;
	int i;

	vTaskDelay (pdMS_TO_TICKS (1000));
	printf( "%s is running\n\n", pcTaskName );

	for(i=0; i<sizeof(TxBuffer)/sizeof(qBuffer); i++)
	{
		//post a message to TASK qpend
		p=xQueueSendToBack(qid3, &TxBuffer[i],portMAX_DELAY);
		//printf("[Task6]set message %d\n",i); send test
		if (p != pdPASS)
		{
			printf("xQueueSendToBack error found\n");
		}
	}
	vTaskDelete (xHandle6);
}
/*-----------------------------------------------------------*/
