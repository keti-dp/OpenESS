#include <stdint.h>

// stack base pointer
stack_base = (unsigned)&CSTACK$$Base;
// stack limit pointer
stack_limit = (unsigned)&CSTACK$$Limit;
// total stack size
stack_size = (unsigned)&CSTACK$$Limit - (unsigned)&CSTACK$$Base;

uint32_t max_runtime = 0u;
uint32_t max_stackuse = 0u;


/**
 * @brief   stack usage 측정 함수
 * @details 현재 MCU에서의 Stack Point를 받아 Stack Usage Percent 반환
 * @return  uint8_t stack_percent
 **/

uint8_t get_stack_use(void)
{
  uint32_t stack_pt = __get_SP(); 
  uint32_t stack_use = (stack_limit - stack_pt);
  uint8_t stack_percent = (uint32_t)(stack_use / stack_size * 100u);
  
  if(stack_percent > max_stackuse)
  {
    max_stackuse = stack_percent;
  }

  return (uint8_t)stack_percent;
}

/**
 * @brief   mcu temperature 측정 함수
 * @details ADC로부터 현재 MCU의 온도 데이터를 받아 변환
 * @return  float temerature
 **/
float get_mcu_temp(void)
{
  HAL_ADC_Start(&hadc1);
  while(HAL_ADC_PollForConversion(&hadc1, 1000) != HAL_OK);
  uint16_t adc_val = HAL_ADC_GetValue(&hadc1);

  float temperature = adc_val * 3300;
  temperature /= 0xfff;                   // Reading in mV
  temperature /= 1000.0f;                 // Reading in Volts
  temperature -= 0.76f;                   // Subtract the reference voltage at 25℃
  temperature /= 0.0025;                  // Divide by slope 2.5mV
  temperature += 25.0;                    // Add the 25℃

  return temperature;
}

/**
 * @brief   BMS Runtime 측정 시작 함수
 * @details BMS Runtime 측정을 위한 타이머 시작함수. 
 *          반드시 loop 시작 시 호출되어야 하며 stop_measure_runtime() 와 함께 사용하여야 한다.
 * @return  uint32_t start_time
 **/
uint32_t start_measure_runtime(void)
{
  start_timer();
  uint32_t start_time = get_timer();
  return start_time;
}

/**
 * @brief   BMS Runtime 측정 종료 함수
 * @details BMS Runtime 측정 종료 및 계산 함수. 
 *          반드시 loop 종료 시 호출되어야 하며 start_measure_runtime() 와 함께 사용하여야 한다.
 * @param   uint32_t start_time
 * @return  uint32_t runtime
 **/
uint32_t stop_measure_runtime(uint32_t start_time)
{
  uint32_t end_time = get_timer();
  uint32_t runtime = end_time - start_time;
  stop_timer();  
  if(runtime > max_runtime){
      max_runtime = runtime;
  }
  return runtime;
}



// 호출 예제
int main(void)
{
  while(1){
    // runtime 측정 시작
    uint32_t start_time = start_measure_runtime();

    // BMS 기능 함수
    func1();
    func2();
    func3();
      ;
      ;
    
    // 동작 코드에 삽입 
    uint8_t stack_usage = get_stack_use();
    float mcu_temp = get_mcu_temp();
    uint32_t runtime = stop_measure_runtime(start_time);

  }
}