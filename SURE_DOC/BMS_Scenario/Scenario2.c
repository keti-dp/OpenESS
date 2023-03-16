// TX,RX BUFFER OVERFLOW Scenario 2

#include <stdio.h>
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"

/* task's priority */
#define TASK_MAIN2_PRIO	20
#define TASK_3_PRIO		15
#define TASK_4_PRIO		16


/* The task functions. */
void TaskMain2( void *pvParameters );
void Task3( void *pvParameters );
void Task4(void *pvParameters );

TaskHandle_t xHandleMain2, xHandle3, xHandle4;

/* ...........................................................................
 *
 * 메시지큐 & 사용자 정의 블럭 정의
 * ========================
*/
QueueHandle_t qid2;

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

void Scenario_2(void)
{
	//prvSetupHardware();
#ifdef CMSIS_OS
	osThreadDef(defaultTask, TaskMain, osPriorityNormal, 0, 256);
	defaultTaskHandle = osThreadCreate(osThread(defaultTask), NULL);
#else

	xTaskCreate(	(TaskFunction_t)TaskMain2,
					"TaskMain2",
					512,
					NULL,
					TASK_MAIN2_PRIO,
					&xHandleMain2 );
#endif
}
/*-----------------------------------------------------------*/

void TaskMain2( void *pvParameters )
{
	const char *pcTaskName = "TaskMain2";

//create a Queue

#if 1
	qid2 = xQueueCreate(QUEUE_LENGTH, QUEUE_ITEM_SIZE);
if (qid2 == NULL)
	printf("xQueueCreate error found\n");
#endif

xTaskCreate((TaskFunction_t)Task3,
				"Task3",
				512,
				NULL,
				TASK_3_PRIO,
				&xHandle3 );


xTaskCreate((TaskFunction_t)Task4,
				"Task4",
				512,
				NULL, //(void*)Param,
				TASK_4_PRIO,
				&xHandle4 );

	printf( "\n**************SCENARIO_2**************\n");

	/* Print out the name of this task. */
	printf( "%s is running\r\n", pcTaskName );

	/* delete self task */
	/* Print out the name of this task. */
	printf( "%s is deleted\r\n\n", pcTaskName );

	vTaskDelete (xHandleMain2);
}
/*-----------------------------------------------------------*/

void Task3( void *pvParameters )
{
	const char *pcTaskName = "Task3";
	qBuffer RxBuffer[15];
	int i;

	vTaskDelay (pdMS_TO_TICKS (1500));
	printf( "%s is running\n\n", pcTaskName );

	for(i=0; i<sizeof(RxBuffer)/sizeof(qBuffer); i++)
	{
		if(xQueueReceive( qid2, &RxBuffer[i],portMAX_DELAY) == pdPASS)
		{
			printf("[Task3]get message \"%d\"\n",i);
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

void Task4( void *pvParameters)
{
	const char *pcTaskName = "Task4";
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
	};;


	BaseType_t p;
	int i;

	vTaskDelay (pdMS_TO_TICKS (1000));
	printf( "%s is running\n", pcTaskName );

	for(i=0; sizeof(TxBuffer)/sizeof(qBuffer); i++)
	{
		//post a message to TASK qpend
		p=xQueueSendToBack(qid2, &TxBuffer[i],portMAX_DELAY);
		//printf("[Task4]set message %d\n",i); send test
		if (p != pdPASS)
		{
			printf("xQueueSendToBack error found\n");
		}
	}
	vTaskDelete (xHandle4);
}
/*-----------------------------------------------------------*/


