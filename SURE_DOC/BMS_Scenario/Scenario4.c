// 데이터 무한 루프 생성 Scenario 4

#include <stdio.h>
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"

/* task's priority */
#define TASK_MAIN4_PRIO	20
#define TASK_7_PRIO		15
#define TASK_8_PRIO		16


/* The task functions. */
void TaskMain4( void *pvParameters );
void Task7( void *pvParameters );
void Task8(void *pvParameters );

TaskHandle_t xHandleMain4, xHandle7, xHandle8;

/* ...........................................................................
 *
 * 메시지큐 & 사용자 정의 블럭 정의
 * ========================
*/
QueueHandle_t qid4;

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

void Scenario_4(void)
{
	//prvSetupHardware();
#ifdef CMSIS_OS
	osThreadDef(defaultTask, TaskMain, osPriorityNormal, 0, 256);
	defaultTaskHandle = osThreadCreate(osThread(defaultTask), NULL);
#else

	xTaskCreate(	(TaskFunction_t)TaskMain4,
					"TaskMain4",
					512,
					NULL,
					TASK_MAIN4_PRIO,
					&xHandleMain4 );
#endif
}
/*-----------------------------------------------------------*/

void TaskMain4( void *pvParameters )
{
	const char *pcTaskName = "TaskMain4";

//create a Queue

#if 1
	qid4 = xQueueCreate(QUEUE_LENGTH, QUEUE_ITEM_SIZE);
if (qid4 == NULL)
	printf("xQueueCreate error found\n");
#endif

xTaskCreate((TaskFunction_t)Task7,
				"Task7",
				512,
				NULL,
				TASK_7_PRIO,
				&xHandle7 );


xTaskCreate((TaskFunction_t)Task8,
				"Task8",
				512,
				NULL, //(void*)Param,
				TASK_8_PRIO,
				&xHandle8 );

	printf( "\n**************SCENARIO_4**************\n");

	/* Print out the name of this task. */
	printf( "%s is running\r\n", pcTaskName );

	/* delete self task */
	/* Print out the name of this task. */
	printf( "%s is deleted\r\n\n", pcTaskName );

	vTaskDelete (xHandleMain4);
}
/*-----------------------------------------------------------*/

void Task7( void *pvParameters )
{
	const char *pcTaskName = "Task7";
	qBuffer RxBuffer[20];
	int i=0;
	int index_count=0;

	vTaskDelay (pdMS_TO_TICKS (1500));
	printf( "%s is running\n\n", pcTaskName );

	for(;;)
	{
		if(xQueueReceive( qid4, &RxBuffer[i],portMAX_DELAY) == pdPASS)
		{
			printf("[Task7]get message \"%d\"\n",index_count);
			printf("RACK_MIN_CELL_TEMPERATURE : \%.3f\n",RxBuffer[i].RACK_MIN_CELL_TEMPERATURE);
			printf("RACK_MIN_CELL_TEMPERATURE_POSITION : \%.3f\n",RxBuffer[i].RACK_MIN_CELL_TEMPERATURE_POSITION);
			printf("RACK_MAX_CELL_TEMPERATURE : \%.3f\n",RxBuffer[i].RACK_MAX_CELL_TEMPERATURE);
			printf("RACK_MAX_CELL_TEMPERATURE_POSITION : \%.3f\n",RxBuffer[i].RACK_MAX_CELL_TEMPERATURE_POSITION);
			printf("RACK_MIN_CELL_VOLTAGE : \%.3f\0n",RxBuffer[i].RACK_MIN_CELL_VOLTAGE);
			printf("RACK_MIN_CELL_VOLTAGE_POSITION : \%.3f\n",RxBuffer[i].RACK_MIN_CELL_VOLTAGE_POSITION);
			printf("RACK_MAX_CELL_VOLTAGE : \%.3f\n",RxBuffer[i].RACK_MAX_CELL_VOLTAGE);
			printf("RACK_MAX_CELL_VOLTAGE_POSITION : \%.3f\n",RxBuffer[i].RACK_MAX_CELL_VOLTAGE_POSITION);
			printf("RACK_VOLTAGE : \%.3f\n",RxBuffer[i].RACK_VOLTAGE);
			printf("RACK_CURRENT : \%.3f\n",RxBuffer[i].RACK_CURRENT);
			printf("RACK_SOC : \%.3f\n\n",RxBuffer[i].RACK_SOC);

			i++;
			i = i%(sizeof(RxBuffer)/sizeof(qBuffer));
			index_count++;
		}
		else
		{
			printf("xQueueReceive error found\ n");
		}
	}
}

/*-----------------------------------------------------------*/

void Task8( void *pvParameters)
{
	const char *pcTaskName = "Task8";
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
	int i=0;

	vTaskDelay (pdMS_TO_TICKS (1000));
	printf( "%s is running\n", pcTaskName );

	for(;;)
	{
		//post a message to TASK qpend
		p=xQueueSendToBack(qid4, &TxBuffer[i],portMAX_DELAY);
		i++;
		//printf("[Task4]set message %d\n",i); send test
		//if (p != pdPASS)
		//{
		//	printf("xQueueSendToBack error found\n");
		//}
	}
	vTaskDelete (xHandle8);
}
/*-----------------------------------------------------------*/


