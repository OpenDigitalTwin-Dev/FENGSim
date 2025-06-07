/* prelink support for gcl images:
   if GCL references variables (as opposed to functions) defined in
   external shared libraries, ld will place COPY relocations in
   .rela.dyn pointing to a location in .bss for these references.
   Unexec will later incorporate this into a second .data section,
   causing prelink to fail.  While one might prelink the raw images,
   which would then be inherited by the saved images, this is not
   convenient as part of the build process, so here we isolate the
   problematic references and compile as position independent code,
   changing the COPY reloc to some form of GOT.
 */
#ifdef NO_PRELINK_UNEXEC_DIVERSION
#define PRELINK_EXTER
#else
#define PRELINK_EXTER extern

#undef stdin
#define stdin my_stdin
#undef stdout
#define stdout my_stdout
#undef stderr
#define stderr my_stderr

#endif

PRELINK_EXTER FILE *my_stdin;
PRELINK_EXTER FILE *my_stdout;
PRELINK_EXTER FILE *my_stderr;

#ifdef USE_READLINE
PRELINK_EXTER rl_compentry_func_t **my_rl_completion_entry_function_ptr;
PRELINK_EXTER const char **my_rl_readline_name_ptr;
#endif
