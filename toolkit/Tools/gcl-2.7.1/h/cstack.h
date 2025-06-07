#if SIZEOF_LONG == 4

#if defined(__PPC__)
#define SET_STACK_POINTER "addi %%r1,%0,0\n\t"
#elif defined(__m68k__)
#define SET_STACK_POINTER "movel %0,%%sp\n\t"
#elif defined(__i386__)
#define SET_STACK_POINTER "mov %0,%%esp\n\t"
#elif defined(__ILP32__) && defined(__x86_64__)
#define SET_STACK_POINTER "mov %0,%%esp\n\t"
#elif defined(__arm__)
#define SET_STACK_POINTER "mov %%sp,%0\n\t"
#elif defined(__hppa__)
#define SET_STACK_POINTER "copy %0,%%sp\n\t"
#elif defined(__SH4__)
#define SET_STACK_POINTER "mov %0,r15\n\t"
#endif

#define FIXED_STACK (1UL<<23)/*FIXME configure?*/
#if defined(__SH4__)/*FIXME is this just due to qemu?*/
#define CTOP (void *)0x80000000
#define SS FIXED_STACK
#elif defined(__gnu_hurd__)
#define CTOP (void *)0xc0000000
#define SS FIXED_STACK
#define MAP_GROWSDOWN 0
#define MAP_STACK 0
#else
#define CTOP (void *)0xc0000000/*FIXME configure?*/
#define SS getpagesize()
#endif

#ifdef SET_STACK_POINTER
{
  void *p,*p1,*b,*s;
  int a,f=MAP_FIXED|MAP_PRIVATE|MAP_ANON|MAP_STACK;

  p=alloca(1);
  p1=alloca(1);
  b=CTOP-(p1<p ? SS : FIXED_STACK);
  a=p1<p ? p-p1 : p1-p;
  a<<=2;
  s=p1<p ? CTOP-a : b+a;
  if (p1<p) f|=MAP_GROWSDOWN;

  if (p > CTOP || p < b) {
    if (mmap(b,SS,PROT_READ|PROT_WRITE|PROT_EXEC,f,-1,0)!=(void *)-1) {
      stack_map_base=b;
      asm volatile (SET_STACK_POINTER::"r" (s):"memory");
      if (p1>p)
	mmap(CTOP,getpagesize(),PROT_NONE,f,-1,0);/*guard page*/
    }
  }
}
#endif
#endif
