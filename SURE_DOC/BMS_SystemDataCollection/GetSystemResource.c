#include <stdint.h>

// stack base pointer 및 limit pointer
stack_base = (unsigned)&CSTACK$$Base;
stack_limit = (unsigned)&CSTACK$$Limit;
stack_size = (unsigned)&CSTACK$$Limit - (unsigned)&CSTACK$$Base;

int main(void){
  while(1){

      func1();
      func2();
      func3();
        ;
        ;
      
      // 동작 코드에 삽입 
      get_stack_use();

  }
}

uint32_t get_stack_use(void){
  uint32_t stack_pt = __get_SP(); 
  uint32_t stack_use = (stack_limit - stack_pt);
  uint8_t stack_Percent = (uint32_t)(stack_use / stack_size * 100u);
  
  if(stack_Percent > max_stackuse)
  {
    max_stackuse = stack_Percent;
  }
  return max_stackuse
}
