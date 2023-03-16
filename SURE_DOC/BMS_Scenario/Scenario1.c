// STACK OVERFLOW Scenario 1

#include <stdio.h>
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"

/* task's priority */
#define TASK_MAIN1_PRIO	20
#define TASK_1_PRIO		17
#define TASK_2_PRIO		18


/* The task functions. */
void TaskMain1( void *pvParameters );
void Task1( void *pvParameters );
void Task2(void *pvParameters);

TaskHandle_t xHandleMain1, xHandle1, xHandle2;

/* ...........................................................................
 *
 * 메시지큐 & 사용자 정의 블럭 정의
 * ===================
 */
QueueHandle_t qid1;

typedef struct tag_qBuffer {

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

void Scenario_1(void)
{
	//prvSetupHardware();
#ifdef CMSIS_OS
	osThreadDef(defaultTask, TaskMain, osPriorityNormal, 0, 256);
	defaultTaskHandle = osThreadCreate(osThread(defaultTask), NULL);
#else

	xTaskCreate(	(TaskFunction_t)TaskMain1,
					"TaskMain1",
					512,
					NULL,
					TASK_MAIN1_PRIO,
					&xHandleMain1);
#endif
}
/*-----------------------------------------------------------*/

void TaskMain1( void *pvParameters )
{
	const char *pcTaskName = "TaskMain1";

//create a Queue

#if 1
	qid1 = xQueueCreate(QUEUE_LENGTH, QUEUE_ITEM_SIZE);
if (qid1 == NULL)
	printf("xQueueCreate error found\n");
#endif

xTaskCreate((TaskFunction_t)Task1,
				"Task1",
				317,
				NULL,
				TASK_1_PRIO,
				&xHandle1 );


xTaskCreate((TaskFunction_t)Task2,
				"Task2",
				512,
				NULL, //(void*)Param,
				TASK_2_PRIO,
				&xHandle2 );

	printf( "\n**************SCENARIO_1**************\n");

	/* Print out the name of this task. */
	printf( "%s is running\r\n", pcTaskName );

	/* delete self task */
	/* Print out the name of this task. */
	printf( "%s is deleted\r\n\n", pcTaskName );

	vTaskDelete (xHandleMain1);
}
/*-----------------------------------------------------------*/

void Task1( void *pvParameters )
{
	const char *pcTaskName = "Task1";
	qBuffer RxBuffer[20];
	int i;

	vTaskDelay (pdMS_TO_TICKS (1000));
	printf( "%s is running\n", pcTaskName );

	for(i=0; i<sizeof(RxBuffer)/sizeof(qBuffer); i++)
	{
		if(xQueueReceive( qid1, &RxBuffer[i],portMAX_DELAY) == pdPASS)
		{
			printf("[Task1]get message \"%d\"\n",i);
			printf("RACK_MIN_CELL_TEMPERATURE : \%.3f\n",RxBuffer[i].RACK_MIN_CELL_TEMPERATURE);
			printf("RACK_MIN_CELL_TEMPERATURE_POSITION : \%.3f\n",RxBuffer[i].RACK_MIN_CELL_TEMPERATURE_POSITION);
			printf("RACK_MAX_CELL_TEMPERATURE : \%.3f\n",RxBuffer[i].RACK_MAX_CELL_TEMPERATURE);
			printf("RACK_MAX_CELL_TEMPERATURE_POSITION : \%.3f\n",RxBuffer[i].RACK_MAX_CELL_TEMPERATURE_POSITION);
			printf("RACK_MIN_CELL_VOLTAGE : \%.3f\n",RxBuffer[i].RACK_MIN_CELL_VOLTAGE);
			printf("RACK_MIN_CELL_VOLTAGE_POSITION : \%.3f\n",RxBuffer[i].RACK_MIN_CELL_VOLTAGE_POSITION);
			printf("RACK_MAX_CELL_VOLTAGE : \%.3f\n",RxBuffer[i].RACK_MAX_CELL_VOLTAGE);
			printf("RACK_MAX_CELL_VOLTAGE_POSITION : \%.3f\n",RxBuffer[i].RACK_MAX_CELL_VOLTAGE_POSITION);
			printf("RACK_VOLTAGE : \%.3f\n",RxBuffer[i].RACK_VOLTAGE);
			printf("RACK_CURRENT : \%.3f\n",RxBuffer[i].RACK_CURRENT);
			printf("RACK_SOC : \%.3f\n\n",RxBuffer[i].RACK_SOC);
		}
		else
		{
			printf("xQueueReceive error found\ n");
		}
	}
	vTaskDelete (NULL);
}

/*-----------------------------------------------------------*/

void Task2( void *pvParameters )
{
	const char *pcTaskName = "Task2";
	qBuffer TxBuffer[20]=
	{
		{20.2,34,23.6,2,3.177,28,3.273,109,660.3,0,0},
		{20.2,34,23.6,2,3.177,28,3.273,109,660.3,0,0},
		{20.2,34,23.6,2,3.177,28,3.273,109,660.2,0,0},
		{20.2,34,23.6,2,3.177,28,3.273,109,660.2,0,0},
		{20.2,34,23.6,2,3.176,28,3.273,109,660.2,0,0},
		{20.2,34,23.6,2,3.177,28,3.273,109,660.2,0,0},
		{20.2,34,23.6,2,3.177,28,3.273,109,660.2,0,0},
		{20.2,34,23.6,2,3.177,28,3.273,109,660.2,0,0},
		{20.2,34,23.6,2,3.177,28,3.273,109,660.2,0,0},
		{20.2,34,23.6,2,3.176,28,3.273,109,660.2,0,0},
		{20.2,34,23.6,2,3.176,28,3.273,109,660.2,0,0},
		{20.2,34,23.6,2,3.177,28,3.273,109,660.2,0,0},
		{20.2,34,23.6,2,3.177,28,3.273,109,660.2,0,0},
		{20.2,34,23.6,2,3.177,28,3.273,109,660.2,0,0},
		{20.2,34,23.6,2,3.177,28,3.273,109,660.2,0,0},
		{20.2,34,23.6,2,3.177,28,3.273,109,660.2,0,0},
		{20.2,34,23.6,2,3.177,28,3.273,109,660.2,0,0},
		{20.2,34,23.6,2,3.177,28,3.273,109,660.2,0,0},
		{20.2,34,23.6,2,3.177,28,3.273,109,660.2,0,0},
		{20.2,34,23.6,2,3.177,28,3.273,109,660.2,0,0}
	};

	BaseType_t p;
	int i;

	vTaskDelay (pdMS_TO_TICKS (1000));
	printf( "%s is running\n", pcTaskName );

	for(i=0; i<sizeof(TxBuffer)/sizeof(qBuffer); i++)
	{
		//post a message to TASK qpend
		p=xQueueSendToBack(qid1, &TxBuffer[i],portMAX_DELAY);

		if (p != pdPASS)
		{
			printf("xQueueSendToBack error found\n");
		}
	}
	vTaskDelete (xHandle2);
}
/*-----------------------------------------------------------*/


