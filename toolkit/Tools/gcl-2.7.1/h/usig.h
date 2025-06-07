typedef void (*handler_function_type)(int,siginfo_t *,void *);

EXTER handler_function_type our_signal_handler[32];

   
#define signal_mask(n)  (1 << (n))
   
   
     
   
   
