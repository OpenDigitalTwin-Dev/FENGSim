/* Copyright (C) 2024 Camm Maguire */
#define NO_PRELINK_UNEXEC_DIVERSION

#include "include.h"

#if !defined(__MINGW32__) && !defined(__CYGWIN__)
extern FILE *stdin __attribute__((weak));
extern FILE *stderr __attribute__((weak));
extern FILE *stdout __attribute__((weak));

#ifdef USE_READLINE

#if defined(RL_COMPLETION_ENTRY_FUNCTION_TYPE_FUNCTION)
extern Function		*rl_completion_entry_function __attribute__((weak));
#elif defined(RL_COMPLETION_ENTRY_FUNCTION_TYPE_RL_COMPENTRY_FUNC_T)
extern rl_compentry_func_t *rl_completion_entry_function __attribute__((weak));
#else
#error Unknown rl_completion_entry_function return type
#endif

#if defined(RL_READLINE_NAME_TYPE_CHAR)
extern char		*rl_readline_name __attribute__((weak));
#elif defined(RL_READLINE_NAME_TYPE_CONST_CHAR)
extern const char *rl_readline_name __attribute__((weak));
#else
#error Unknown rl_readline_name return type
#endif

#endif
#endif

void
prelink_init(void) {
  
  my_stdin=stdin;
  my_stdout=stdout;
  my_stderr=stderr;
#ifdef USE_READLINE
  my_rl_completion_entry_function_ptr=(void *)&rl_completion_entry_function;
  my_rl_readline_name_ptr=(void *)&rl_readline_name;
#endif

}

