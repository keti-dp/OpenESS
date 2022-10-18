#include <stdint.h>
#include <stddef.h>

// stack base pointer 및 limit pointer
stack_base = (unsigned)&CSTACK$$Base;
stack_limit = (unsigned)&CSTACK$$Limit;
stack_size = (unsigned)&CSTACK$$Limit - (unsigned)&CSTACK$$Base;

int main(void)
{
  while(1){
      func1();
      func2();
      func3();
        ;
        ;
      
      // 동작 코드에 삽입 
      uint8_t stack_usage = get_stack_use();
      float mcu_temp = get_mcu_temp();

  }
}

uint8_t get_stack_use(void)
{
  uint32_t stack_pt = __get_SP(); 
  uint32_t stack_use = (stack_limit - stack_pt);
  uint8_t stack_percent = (uint32_t)(stack_use / stack_size * 100u);
  
  if(stack_percent > max_stackuse)
  {
    max_stackuse = stack_percent;
  }

  return (uint8_t)max_stackuse;
}

uint16_t get_mcu_temp(void)
{
  HAL_ADC_Start(&hadc1);
  while(HAL_ADC_PollForConversion(&hadc1, 1000) != HAL_OK);
  uint16_t adc_val = HAL_ADC_GetValue(&hadc1);

  float temperature = adc_val * 3300;
  temperature /= 0xfff;   //Reading in mV
  temperature /= 1000.0f; //Reading in Volts
  temperature -= 0.76f;   // Subtract the reference voltage at 25℃
  temperature /= 0.0025;  // Divide by slope 2.5mV
  temperature += 25.0;    // Add the 25℃

  return temperature;
}

uint32_t get_runtime(void){

}
