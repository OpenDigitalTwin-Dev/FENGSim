/* unexec for GNU Emacs on Windows NT.
   Copyright (C) 1994 Free Software Foundation, Inc.
   Copyright (C) 2024 Camm Maguire

This file is part of GNU Emacs.

GNU Emacs is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2, or (at your option)
any later version.

GNU Emacs is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING.  If not, write to
the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

   Geoff Voelker (voelker@cs.washington.edu)                         8-12-94
*/

/* #include "gclincl.h" */

#ifndef UNIXSAVE
#include <config.h>
#endif
/* in case the include of config.h defined it */
#undef va_start
#include <stdio.h>
#include <fcntl.h>
#include <time.h>
#include <windows.h>

#ifdef _GNU_H_WINDOWS_H 
#include "cyglacks.h"
#endif

/* Include relevant definitions from IMAGEHLP.H, which can be found
   in \\win32sdk\mstools\samples\image\include\imagehlp.h. */



PIMAGE_NT_HEADERS
(__stdcall * pfnCheckSumMappedFile) (LPVOID BaseAddress,
				    DWORD FileLength,
				    LPDWORD HeaderSum,
				    LPDWORD CheckSum);



#include <stdio.h>

#include "ntheap.h"

/* Info for keeping track of our heap.  */
unsigned char *data_region_base = UNINIT_PTR;
unsigned char *data_region_end = UNINIT_PTR;
unsigned char *real_data_region_end = UNINIT_PTR;
unsigned long  data_region_size = UNINIT_LONG;
unsigned long  reserved_heap_size = UNINIT_LONG;

extern BOOL ctrl_c_handler (unsigned long type);

extern char my_begdata[];
extern char my_edata[];
extern char my_begbss[];
extern char my_endbss[];
extern char *my_begbss_static;
extern char *my_endbss_static;

#include "ntheap.h"

enum {
  HEAP_UNINITIALIZED = 1,
  HEAP_UNLOADED,
  HEAP_LOADED
};

/* Basically, our "initialized" flag.  */
int heap_state = HEAP_UNINITIALIZED;

/* So we can find our heap in the file to recreate it.  */
unsigned long heap_index_in_executable = UNINIT_LONG;

static void get_section_info (file_data *p_file);
static void copy_executable_and_dump_data_section (file_data *, file_data *);
static void dump_bss_and_heap (file_data *p_infile, file_data *p_outfile);

/* Cached info about the .data section in the executable.  */
PUCHAR data_start_va = UNINIT_PTR;
DWORD  data_start_file = UNINIT_LONG;
DWORD  data_size = UNINIT_LONG;

/* Cached info about the .bss section in the executable.  */
PUCHAR bss_start = UNINIT_PTR;
DWORD  bss_size = UNINIT_LONG;

void recreate_heap1()
{
  char executable_path[MAX_PATH];
  
  if (heap_state == HEAP_UNLOADED) { 
  if (GetModuleFileName (NULL, executable_path, MAX_PATH) == 0) 
    {
      printf ("Failed to find path for executable.\n");
      do_gcl_abort();
    }
    recreate_heap (executable_path);
  }
  heap_state = HEAP_LOADED;

}


#ifdef HAVE_NTGUI
HINSTANCE hinst = NULL;
HINSTANCE hprevinst = NULL;
LPSTR lpCmdLine = "";
int nCmdShow = 0;
#endif /* HAVE_NTGUI */

#ifndef UNIXSAVE
/* Startup code for running on NT.  When we are running as the dumped
   version, we need to bootstrap our heap and .bss section into our
   address space before we can actually hand off control to the startup
   code supplied by NT (primarily because that code relies upon malloc ()).  */
void
_start (void)
{
  extern void mainCRTStartup (void);

#if 0
  /* Give us a way to debug problems with crashes on startup when
     running under the MSVC profiler. */
  if (GetEnvironmentVariable ("EMACS_DEBUG", NULL, 0) > 0)
    DebugBreak ();
#endif

  /* Cache system info, e.g., the NT page size.  */
  cache_system_info ();

  /* If we're a dumped version of emacs then we need to recreate
     our heap and play tricks with our .bss section.  Do this before
     start up.  (WARNING:  Do not put any code before this section
     that relies upon malloc () and runs in the dumped version.  It
     won't work.)  */
  if (heap_state == HEAP_UNLOADED) 
    {
      char executable_path[MAX_PATH];

      if (GetModuleFileName (NULL, executable_path, MAX_PATH) == 0) 
	{
	  printf ("Failed to find path for executable.\n");
	  do_gcl_abort();
	}

#if 1
      /* To allow profiling, make sure executable_path names the .exe
	 file, not the ._xe file created by the profiler which contains
	 extra code that makes the stored exe offsets incorrect.  (This
	 will not be necessary when unexec properly extends the .bss (or
	 .data as appropriate) section to include the dumped bss data,
	 and dumps the heap into a proper section of its own.)  */
      {
	char * p = strrchr (executable_path, '.');
	if (p && p[1] == '_')
	  p[1] = 'e';
      }

      /* Using HiProf profiler, exe name is different still. */
      {
	char * p = strrchr (executable_path, '\\');
	strcpy (p, "\\emacs.exe");
      }
#endif

      recreate_heap (executable_path);
      heap_state = HEAP_LOADED;
    }
  else
    {
      /* Grab our malloc arena space now, before CRT starts up. */
      sbrk (0);
    }

  /* The default behavior is to treat files as binary and patch up
     text files appropriately, in accordance with the MSDOS code.  */
  _fmode = O_BINARY;

  /* This prevents ctrl-c's in shells running while we're suspended from
     having us exit.  */
  SetConsoleCtrlHandler ((PHANDLER_ROUTINE) ctrl_c_handler, TRUE);

  /* Invoke the NT CRT startup routine now that our housecleaning
     is finished.  */
#ifdef HAVE_NTGUI
  /* determine WinMain args like crt0.c does */
  hinst = GetModuleHandle(NULL);
  lpCmdLine = GetCommandLine();
  nCmdShow = SW_SHOWDEFAULT;
#endif
  mainCRTStartup ();
}
#endif /* UNIXSAVE */

#ifdef __CYGWIN__
#include <sys/cygwin.h>
#endif

/* Dump out .data and .bss sections into a new executable.  */
void
unexec (char *new_name, char *old_name, void *start_data, void *start_bss,
	void *entry_address)
{
#ifdef __CYGWIN__
  static file_data in_file, out_file;
  char out_filename[MAX_PATH], in_filename[MAX_PATH];
  char filename[MAX_PATH];
  unsigned long size;
  char *ptr;

  fflush (stdin);
  /* copy_stdin = *stdin; */
    setvbuf(stdin,0,_IONBF,0);
    setvbuf(stdout,0,_IONBF,0);
    
  /* stdin->_data->__sdidinit = 0;
   */
  

  if (!get_allocation_unit())
    cache_system_info ();
  
  /* Make sure that the input and output filenames have the
     ".exe" extension...patch them up if they don't.  */
  ptr = old_name + strlen (old_name) - 4;
  strcpy(filename, old_name);
  strcat(filename, (strcmp (ptr, ".exe") && strcmp (ptr, ".EXE"))?".exe":"");
  cygwin_conv_path(CCP_POSIX_TO_WIN_A,filename,in_filename,sizeof(in_filename));
  ptr = new_name + strlen (new_name) - 4;
  strcpy(filename, new_name);
  strcat(filename, (strcmp (ptr, ".exe") && strcmp (ptr, ".EXE"))?".exe":"");
  cygwin_conv_path(CCP_POSIX_TO_WIN_A,filename,out_filename,sizeof(out_filename));
#else 
  static file_data in_file, out_file;
  char out_filename[MAX_PATH], in_filename[MAX_PATH];
  unsigned long size;
  char *ptr;

  fflush (stdin);
  /* copy_stdin = *stdin; */
    setvbuf(stdin,0,_IONBF,0);
    setvbuf(stdout,0,_IONBF,0);
    
  /* stdin->_data->__sdidinit = 0;
   */
  

  if (!get_allocation_unit())
    cache_system_info ();
  
  /* Make sure that the input and output filenames have the
     ".exe" extension...patch them up if they don't.  */
  strcpy (in_filename, old_name);
  ptr = in_filename + strlen (in_filename) - 4;
  if  (strcmp (ptr, ".exe") && strcmp (ptr, ".EXE")  )
    strcat (in_filename, ".exe");

  strcpy (out_filename, new_name);
  ptr = out_filename + strlen (out_filename) - 4;
  if (strcmp (ptr, ".exe") && strcmp (ptr, ".EXE")  )
    strcat (out_filename, ".exe");
#endif
  /* printf ("Dumping from %s\n", in_filename); */
  /* printf ("          to %s\n", out_filename); */

  /* We need to round off our heap to NT's allocation unit (64KB).  */
  round_heap (get_allocation_unit ());

  /* Open the undumped executable file.  */
  if (!open_input_file (&in_file, in_filename))
    {
      printf ("Failed to open %s (%u)...bailing.\n",
	      in_filename, (unsigned)GetLastError ());
      do_gcl_abort();
    }

  /* Get the interesting section info, like start and size of .bss...  */
  get_section_info (&in_file);

  /* The size of the dumped executable is the size of the original
     executable plus the size of the heap and the size of the .bss section.  */
  if (heap_index_in_executable==UNINIT_LONG)
    heap_index_in_executable = (unsigned long)
      round_to_next ((unsigned char *) in_file.size, get_allocation_unit ());
  /* from lisp we know what to use */
#ifdef IN_UNIXSAVE
  data_region_end = round_to_next((unsigned char *)core_end,0x10000);
  real_data_region_end = data_region_end;
#endif  
  size = heap_index_in_executable + get_committed_heap_size () + bss_size;
  if (!open_output_file (&out_file, out_filename, size))
    {
      printf ("Failed to open %s (%u)...bailing.\n",
	      out_filename, (unsigned)GetLastError ());
      do_gcl_abort();
    }

  /* Set the flag (before dumping).  */
  heap_state = HEAP_UNLOADED;

  copy_executable_and_dump_data_section (&in_file, &out_file);
  dump_bss_and_heap (&in_file, &out_file);

  /* Patch up header fields; profiler is picky about this. */

  {
    PIMAGE_DOS_HEADER dos_header;
    PIMAGE_NT_HEADERS nt_header;
    HANDLE hImagehelp = LoadLibrary ("imagehlp.dll");
    DWORD  headersum;
    DWORD  checksum;

    dos_header = (PIMAGE_DOS_HEADER) out_file.file_base;
    nt_header = (PIMAGE_NT_HEADERS) ((char *) dos_header + dos_header->e_lfanew);

 
    nt_header->OptionalHeader.SizeOfStackReserve=0x800000;
    /* nt_header->OptionalHeader.SizeOfHeapReserve=0x80000000; */
    /* nt_header->OptionalHeader.SizeOfHeapCommit=0x80000000; */
   
    nt_header->OptionalHeader.CheckSum = 0;
//    nt_header->FileHeader.TimeDateStamp = time (NULL);
//    dos_header->e_cp = size / 512;
//    nt_header->OptionalHeader.SizeOfImage = size;

    pfnCheckSumMappedFile = (void *) GetProcAddress (hImagehelp, "CheckSumMappedFile");
    if (pfnCheckSumMappedFile)
      {
//	nt_header->FileHeader.TimeDateStamp = time (NULL);
	pfnCheckSumMappedFile (out_file.file_base,
			       out_file.size,
			       &headersum,
			       &checksum);
	nt_header->OptionalHeader.CheckSum = checksum;
      }
    FreeLibrary (hImagehelp);
  }

  close_file_data (&in_file);
  close_file_data (&out_file);
}


/* File handling.  */


int
open_input_file (file_data *p_file, char *filename)
{
  HANDLE file;
  HANDLE file_mapping;
  void  *file_base;
  DWORD size, upper_size;

  file = CreateFile (filename, GENERIC_READ, FILE_SHARE_READ, NULL,
		     OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
  if (file == INVALID_HANDLE_VALUE)
    return FALSE;

  size = GetFileSize (file, &upper_size);
  file_mapping = CreateFileMapping (file, NULL, PAGE_READONLY, 
				    0, size, NULL);
  if (!file_mapping) 
    return FALSE;

  file_base = MapViewOfFile (file_mapping, FILE_MAP_READ, 0, 0, size);
  if (file_base == 0) 
    return FALSE;

  p_file->name = filename;
  p_file->size = size;
  p_file->file = file;
  p_file->file_mapping = file_mapping;
  p_file->file_base = file_base;

  return TRUE;
}

int
open_output_file (file_data *p_file, char *filename, unsigned long size)
{
  HANDLE file;
  HANDLE file_mapping;
  void  *file_base;

  file = CreateFile (filename, GENERIC_READ | GENERIC_WRITE, 0, NULL,
		     CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0);
  if (file == INVALID_HANDLE_VALUE) 
    return FALSE;

  file_mapping = CreateFileMapping (file, NULL, PAGE_READWRITE, 
				    0, size, NULL);
  if (!file_mapping) 
    return FALSE;
  
  file_base = MapViewOfFile (file_mapping, FILE_MAP_WRITE, 0, 0, size);
  if (file_base == 0) 
    return FALSE;
  
  p_file->name = filename;
  p_file->size = size;
  p_file->file = file;
  p_file->file_mapping = file_mapping;
  p_file->file_base = file_base;

  return TRUE;
}

/* Close the system structures associated with the given file.  */
void
close_file_data (file_data *p_file)
{
    UnmapViewOfFile (p_file->file_base);
    CloseHandle (p_file->file_mapping);
    CloseHandle (p_file->file);
}


/* Routines to manipulate NT executable file sections.  */

#ifdef SEPARATE_BSS_SECTION
static void
get_bss_info_from_map_file (file_data *p_infile, PUCHAR *p_bss_start, 
			    DWORD *p_bss_size)
{
  int n, start, len;
  char map_filename[MAX_PATH];
  char buffer[256];
  FILE *map;

  /* Overwrite the .exe extension on the executable file name with
     the .map extension.  */
  strcpy (map_filename, p_infile->name);
  n = strlen (map_filename) - 3;
  strcpy (&map_filename[n], "map");

  map = fopen (map_filename, "r");
  if (!map)
    {
      printf ("Failed to open map file %s, error %d...bailing out.\n",
	      map_filename, GetLastError ());
      do_gcl_abort();
    }

  while (fgets (buffer, sizeof (buffer), map))
    {
      if (!(strstr (buffer, ".bss") && strstr (buffer, "DATA")))
	continue;
      n = sscanf (buffer, " %*d:%x %x", &start, &len);
      if (n != 2)
	{
	  printf ("Failed to scan the .bss section line:\n%s", buffer);
	  do_gcl_abort();
	}
      break;
    }
  *p_bss_start = (PUCHAR) start;
  *p_bss_size = (DWORD) len;
}
#endif

unsigned long
get_section_size (PIMAGE_SECTION_HEADER p_section)
{
  /* The true section size, before rounding.  Some linkers swap the
     meaning of these two values.  */
  return min (p_section->SizeOfRawData,
	      p_section->Misc.VirtualSize);
}

/* Return pointer to section header for named section. */
IMAGE_SECTION_HEADER *
find_section (char * name, IMAGE_NT_HEADERS * nt_header)
{
  PIMAGE_SECTION_HEADER section;
  int i;

  section = IMAGE_FIRST_SECTION (nt_header);

  for (i = 0; i < nt_header->FileHeader.NumberOfSections; i++)
    {
      if (strcmp ((char *)section->Name, name) == 0)
	return section;
      section++;
    }
  return NULL;
}

/* Return pointer to section header for section containing the given
   relative virtual address. */
IMAGE_SECTION_HEADER *
rva_to_section (DWORD rva, IMAGE_NT_HEADERS * nt_header)
{
  PIMAGE_SECTION_HEADER section;
  int i;

  section = IMAGE_FIRST_SECTION (nt_header);

  for (i = 0; i < nt_header->FileHeader.NumberOfSections; i++)
    {
      if (rva >= section->VirtualAddress &&
	  rva < section->VirtualAddress + section->SizeOfRawData)
	return section;
      section++;
    }
  return NULL;
}


/* Flip through the executable and cache the info necessary for dumping.  */
static void
get_section_info (file_data *p_infile)
{
  PIMAGE_DOS_HEADER dos_header;
  PIMAGE_NT_HEADERS nt_header;
  PIMAGE_SECTION_HEADER section, data_section;
  unsigned char *ptr;
  int i;
  
  dos_header = (PIMAGE_DOS_HEADER) p_infile->file_base;
  if (dos_header->e_magic != IMAGE_DOS_SIGNATURE) 
    {
      printf ("Unknown EXE header in %s...bailing.\n", p_infile->name);
      do_gcl_abort();
    }
  nt_header = (PIMAGE_NT_HEADERS) (((unsigned long) dos_header) + 
				   dos_header->e_lfanew);
  if (nt_header == NULL) 
    {
      printf ("Failed to find IMAGE_NT_HEADER in %s...bailing.\n", 
	     p_infile->name);
      do_gcl_abort();
    }

  /* Check the NT header signature ...  */
  if (nt_header->Signature != IMAGE_NT_SIGNATURE) 
    {
      printf ("Invalid IMAGE_NT_SIGNATURE 0x%x in %s...bailing.\n",
	      (int)nt_header->Signature, p_infile->name);
    }

  /* Flip through the sections for .data and .bss ...  */
  section = (PIMAGE_SECTION_HEADER) IMAGE_FIRST_SECTION (nt_header);
  for (i = 0; i < nt_header->FileHeader.NumberOfSections; i++) 
    {
#ifdef SEPARATE_BSS_SECTION
      if (!strcmp (section->Name, ".bss")) 
	{
	  /* The .bss section.  */
	  ptr = (char *) nt_header->OptionalHeader.ImageBase +
	    section->VirtualAddress;
	  bss_start = ptr;
	  bss_size = get_section_size (section);
	}
#endif
#if 0
      if (!strcmp (section->Name, ".data")) 
	{
	  /* From lastfile.c  */
	  extern char my_edata[];

	  /* The .data section.  */
	  data_section = section;
	  ptr = (char *) nt_header->OptionalHeader.ImageBase +
	    section->VirtualAddress;
	  data_start_va = ptr;
	  data_start_file = section->PointerToRawData;

	  /* We want to only write Emacs data back to the executable,
	     not any of the library data (if library data is included,
	     then a dumped Emacs won't run on system versions other
	     than the one Emacs was dumped on).  */
	  data_size = my_edata - data_start_va;
	}
#else
#ifdef emacs
 #define DATA_SECTION "EMDATA"
#else
#define DATA_SECTION ".data"
#endif      
      if (!strcmp ((char *)section->Name, DATA_SECTION)) 
	{
	  /* The Emacs initialized data section.  */
	  data_section = section;
	  ptr = (unsigned char *) nt_header->OptionalHeader.ImageBase +
	    section->VirtualAddress;
	  data_start_va = ptr;
	  data_start_file = section->PointerToRawData;

	  /* Write back the full section.  */
	  data_size = get_section_size (section);
	}
#endif
      section++;
    }

#ifdef SEPARATE_BSS_SECTION
  if (bss_start == UNINIT_PTR && bss_size == UNINIT_LONG)
    {
      /* Starting with MSVC 4.0, the .bss section has been eliminated
	 and appended virtually to the end of the .data section.  Our
	 only hint about where the .bss section starts in the address
	 comes from the SizeOfRawData field in the .data section
	 header.  Unfortunately, this field is only approximate, as it
	 is a rounded number and is typically rounded just beyond the
	 start of the .bss section.  To find the start and size of the
	 .bss section exactly, we have to peek into the map file.  */
      get_bss_info_from_map_file (p_infile, &ptr, &bss_size);
      bss_start = ptr + nt_header->OptionalHeader.ImageBase
	+ data_section->VirtualAddress;
    }
#else
/* As noted in lastfile.c, the Alpha (but not the Intel) MSVC linker
   globally segregates all static and public bss data (ie. across all
   linked modules, not just per module), so we must take both static and
   public bss areas into account to determine the true extent of the bss
   area used by Emacs.

   To be strictly correct, we should dump the static and public bss
   areas used by Emacs separately if non-overlapping (since otherwise we
   are dumping bss data belonging to system libraries, eg. the static
   bss system data on the Alpha).  However, in practice this doesn't
   seem to matter, since presumably the system libraries always
   reinitialize their bss variables.  */
  bss_start = (unsigned char *)min (my_begbss, my_begbss_static);
  bss_size = max ((char *)my_endbss, (char *) my_endbss_static) - (char *) bss_start;

#endif
}


/* The dump routines.  */

static void
copy_executable_and_dump_data_section (file_data *p_infile, 
				       file_data *p_outfile)
{
  unsigned char *data_file, *data_va;
  unsigned long size, index;
  
  /* Get a pointer to where the raw data should go in the executable file.  */
  data_file = (unsigned char *) p_outfile->file_base + data_start_file;

  /* Get a pointer to the raw data in our address space.  */
  data_va = data_start_va;
    
  size = (unsigned long) data_file - (unsigned long) p_outfile->file_base;
  /* printf ("Copying executable up to data section...\n"); */
  /* printf ("\t0x%08x Offset in input file.\n", 0); */
  /* printf ("\t0x%08x Offset in output file.\n", 0); */
  /* printf ("\t0x%08lx Size in bytes.\n", size); */
  memcpy (p_outfile->file_base, p_infile->file_base, size);
  
  size = data_size;
  /* printf ("Dumping .data section...\n"); */
  /* printf ("\t0x%p Address in process.\n", data_va); */
  /* printf ("\t0x%08x Offset in output file.\n",  */
  /* 	  data_file - p_outfile->file_base); */
  /* printf ("\t0x%08lx Size in bytes.\n", size); */
  memcpy (data_file, data_va, size);
  
  index = (unsigned long) data_file + size - (unsigned long) p_outfile->file_base;
  size = p_infile->size - index;
  /* printf ("Copying rest of executable...\n"); */
  /* printf ("\t0x%08lx Offset in input file.\n", index); */
  /* printf ("\t0x%08lx Offset in output file.\n", index); */
  /* printf ("\t0x%08lx Size in bytes.\n", size); */
  memcpy ((char *) p_outfile->file_base + index, 
	  (char *) p_infile->file_base + index, size);
}

static void
dump_bss_and_heap (file_data *p_infile, file_data *p_outfile)
{
    unsigned char *heap_data, *bss_data;
    unsigned long size, index;

    /* printf ("Dumping heap into executable...\n"); */

    index = heap_index_in_executable;
    size = get_committed_heap_size ();
    heap_data = get_heap_start ();

    /* printf ("\t0x%p Heap start in process.\n", heap_data); */
    /* printf ("\t0x%08lx Heap offset in executable.\n", index); */
    /* printf ("\t0x%08lx Heap size in bytes.\n", size); */

    memcpy ((PUCHAR) p_outfile->file_base + index, heap_data, size);

    /* printf ("Dumping .bss into executable...\n"); */
    
    index += size;
    size = bss_size;
    bss_data = bss_start;
    
    /* printf ("\t0x%p BSS start in process.\n", bss_data); */
    /* printf ("\t0x%08lx BSS offset in executable.\n", index); */
    /* printf ("\t0x%08lx BSS size in bytes.\n", size); */
    memcpy ((char *) p_outfile->file_base + index, bss_data, size);
}


/* Reload and remap routines.  */


/* Load the dumped .bss section into the .bss area of our address space.  */
void
read_in_bss (char *filename)
{
  HANDLE file;
  DWORD index, n_read;
  int    i;

  file = CreateFile (filename, GENERIC_READ, FILE_SHARE_READ, NULL,
		     OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
  if (file == INVALID_HANDLE_VALUE) 
    {
      i = GetLastError ();
      do_gcl_abort();
    }

  /* Seek to where the .bss section is tucked away after the heap...  */
  index = heap_index_in_executable + get_committed_heap_size ();
  if (SetFilePointer (file, index, NULL, FILE_BEGIN) == 0xFFFFFFFF) 
    {
      i = GetLastError ();
      do_gcl_abort();
    }

  
  /* Ok, read in the saved .bss section and initialize all 
     uninitialized variables.  */
  if (!ReadFile (file, bss_start, bss_size, &n_read, (void *)NULL))
    {
      i = GetLastError ();
      do_gcl_abort();
    }

  CloseHandle (file);
}

/* Map the heap dumped into the executable file into our address space.  */
void 
map_in_heap (char *filename)
{
  HANDLE file;
  HANDLE file_mapping;
  void  *file_base;
  DWORD size, upper_size, n_read;
  int    i;

  file = CreateFile (filename, GENERIC_READ, FILE_SHARE_READ, NULL,
		     OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
  if (file == INVALID_HANDLE_VALUE) 
    {
      i = GetLastError ();
      do_gcl_abort();
    }
  
  size = GetFileSize (file, &upper_size);
  file_mapping = CreateFileMapping (file, NULL, PAGE_WRITECOPY, 
				    0, size, NULL);
  if (!file_mapping) 
    {
      i = GetLastError ();
      do_gcl_abort();
    }
    
  size = get_committed_heap_size ();
  file_base = MapViewOfFileEx (file_mapping, FILE_MAP_ALL_ACCESS, 0,
			       heap_index_in_executable, size,
			       get_heap_start ());
  if (file_base != 0) 
    {
      return;
    }

  /* If we don't succeed with the mapping, then copy from the 
     data into the heap.  */

  CloseHandle (file_mapping);

  if (VirtualAlloc (get_heap_start (), get_committed_heap_size (),
		    MEM_COMMIT, PAGE_EXECUTE_READWRITE) == NULL)
    {
      i = GetLastError ();
      do_gcl_abort();
    }

  /* Seek to the location of the heap data in the executable.  */
  i = heap_index_in_executable;
  if (SetFilePointer (file, i, NULL, FILE_BEGIN) == 0xFFFFFFFF)
    {
      i = GetLastError ();
      do_gcl_abort();
    }

  /* Read in the data.  */
  if (!ReadFile (file, get_heap_start (), 
		 get_committed_heap_size (), &n_read, (void *)NULL))
    {
      i = GetLastError ();
      do_gcl_abort();
    }

  CloseHandle (file);
}

/* ntheap.c */
/* Heap management routines for GNU Emacs on Windows NT.
   Copyright (C) 1994 Free Software Foundation, Inc.

This file is part of GNU Emacs.

GNU Emacs is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2, or (at your option)
any later version.

GNU Emacs is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GNU Emacs; see the file COPYING.  If not, write to
the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
Boston, MA 02111-1307, USA.

   Geoff Voelker (voelker@cs.washington.edu)			     7-29-94
*/
/*

*/
/* #include "lisp.h" */  /* for VALMASK */
#define VALMASK -1
/* try for 500 MB of address space */
#define VALBITS 29

/* This gives us the page size and the size of the allocation unit on NT.  */
SYSTEM_INFO sysinfo_cache;
unsigned long syspage_mask = 0;

/* These are defined to get Emacs to compile, but are not used.  */
int edata;
int etext;

/* The major and minor versions of NT.  */
int nt_major_version;
int nt_minor_version;

/* Distinguish between Windows NT and Windows 95.  */
int os_subtype;

/* Cache information describing the NT system for later use.  */
void
cache_system_info (void)
{
  union 
    {
      struct info 
	{
	  char  major;
	  char  minor;
	  short platform;
	} info;
      DWORD data;
    } version;

  /* Cache the version of the operating system.  */
  version.data = GetVersion ();
  nt_major_version = version.info.major;
  nt_minor_version = version.info.minor;

  if (version.info.platform & 0x8000)
    os_subtype = OS_WIN95;
  else
    os_subtype = OS_NT;

  /* Cache page size, allocation unit, processor type, etc.  */
  GetSystemInfo (&sysinfo_cache);
  syspage_mask = sysinfo_cache.dwPageSize - 1;
}

/* Round ADDRESS up to be aligned with ALIGN.  */
unsigned char *
round_to_next (unsigned char *address, unsigned long align)
{
  unsigned long tmp;

  tmp = (unsigned long) address;
  tmp = (tmp + align - 1) / align;

  return (unsigned char *) (tmp * align);
}


/* The start of the data segment.  */
unsigned char *
get_data_start (void)
{
  return data_region_base;
}

/* The end of the data segment.  */
unsigned char *
get_data_end (void)
{
  return data_region_end;
}

void *
probe_base(void *base,unsigned long try,unsigned long inc,unsigned long c) {
  void *r;
  if (!(r=VirtualAlloc(base,try,MEM_RESERVE,PAGE_NOACCESS)))
    return probe_base(base+inc,try,inc,c+1);
  VirtualFree (r, 0, MEM_RELEASE);
  return !c || inc<2 ? base : probe_base(base-inc,try,inc>>1,c+1);
}

unsigned long
probe_heap_size(void *base,unsigned long try,unsigned long inc) {
  void *r;
  if (!(r=VirtualAlloc(base,try,MEM_RESERVE,PAGE_NOACCESS)))
    return inc<2 ? try-inc : probe_heap_size(base,try-inc,inc>>1);
  VirtualFree (r, 0, MEM_RELEASE);
  return probe_heap_size(base,try+inc,inc);
}

static char *
allocate_heap (void)
{
  /* The base address for our GNU malloc heap is chosen in conjuction
     with the link settings for temacs.exe which control the stack size,
     the initial default process heap size and the executable image base
     address.  The link settings and the malloc heap base below must all
     correspond; the relationship between these values depends on how NT
     and Win95 arrange the virtual address space for a process (and on
     the size of the code and data segments in temacs.exe).

     The most important thing is to make base address for the executable
     image high enough to leave enough room between it and the 4MB floor
     of the process address space on Win95 for the primary thread stack,
     the process default heap, and other assorted odds and ends
     (eg. environment strings, private system dll memory etc) that are
     allocated before temacs has a chance to grab its malloc arena.  The
     malloc heap base can then be set several MB higher than the
     executable image base, leaving enough room for the code and data
     segments.

     Because some parts of Emacs can use rather a lot of stack space
     (for instance, the regular expression routines can potentially
     allocate several MB of stack space) we allow 8MB for the stack.

     Allowing 1MB for the default process heap, and 1MB for odds and
     ends, we can base the executable at 16MB and still have a generous
     safety margin.  At the moment, the executable has about 810KB of
     code (for x86) and about 550KB of data - on RISC platforms the code
     size could be roughly double, so if we allow 4MB for the executable
     we will have plenty of room for expansion.

     Thus we would like to set the malloc heap base to 20MB.  However,
     Win95 refuses to allocate the heap starting at this address, so we
     set the base to 27MB to make it happy.  Since Emacs now leaves
     28 bits available for pointers, this lets us use the remainder of
     the region below the 256MB line for our malloc arena - 229MB is
     still a pretty decent arena to play in!  */

  void *base,*ptr;
  unsigned long min=PAGESIZE,inc=(1UL<<31);

#if defined(__CYGWIN__)
  ptr=my_endbss;
#else
  ptr=(void *)0x5000000;
#endif
  base=probe_base(ptr,min,(unsigned long)my_endbss,0);
  reserved_heap_size=probe_heap_size(base,inc+min,inc);
  ptr = VirtualAlloc ((void *) base,get_reserved_heap_size (),MEM_RESERVE,PAGE_NOACCESS);
  /* printf("probe results: %lu at %p\n",reserved_heap_size,ptr); */

  DBEGIN = (DBEGIN_TY) ptr;

  return ptr;

}

/* Emulate Unix sbrk.  */
void *
sbrk (ptrdiff_t increment)
{
  void *result;
  long size = (long) increment;
  
  /* Allocate our heap if we haven't done so already.  */
  if (data_region_base == UNINIT_PTR) 
    {
      data_region_base = (unsigned char *)allocate_heap ();
      if (!data_region_base)
	return NULL;

      /* Ensure that the addresses don't use the upper tag bits since
	 the Lisp type goes there.  */
      if (((unsigned long) data_region_base & ~VALMASK) != 0) 
	{
	  printf ("Error: The heap was allocated in upper memory.\n");
	  do_gcl_abort();
	}

      data_region_end = data_region_base;
      real_data_region_end = data_region_end;
      data_region_size = get_reserved_heap_size ();
    }
  
  result = data_region_end;
  
  /* If size is negative, shrink the heap by decommitting pages.  */
  if (size < 0) 
    {
      int new_size;
      unsigned char *new_data_region_end;

      size = -size;

      /* Sanity checks.  */
      if ((data_region_end - size) < data_region_base)
	return NULL;

      /* We can only decommit full pages, so allow for 
	 partial deallocation [cga].  */
      new_data_region_end = (data_region_end - size);
      new_data_region_end = (unsigned char *)
	((long) (new_data_region_end + syspage_mask) & ~syspage_mask);
      new_size = real_data_region_end - new_data_region_end;
      real_data_region_end = new_data_region_end;
      if (new_size > 0) 
	{
	  /* Decommit size bytes from the end of the heap.  */
	  if (!VirtualFree (real_data_region_end, new_size, MEM_DECOMMIT))
	    return NULL;
 	}

      data_region_end -= size;
    } 
  /* If size is positive, grow the heap by committing reserved pages.  */
  else if (size > 0) 
    {
      /* Sanity checks.  */
      if ((data_region_end + size) >
	  (data_region_base + get_reserved_heap_size ()))
	return NULL;

      /* Commit more of our heap. */
      if (VirtualAlloc (data_region_end, size, MEM_COMMIT,
			PAGE_EXECUTE_READWRITE) == NULL)
	return NULL;
      data_region_end += size;

      /* We really only commit full pages, so record where
	 the real end of committed memory is [cga].  */
      real_data_region_end = (unsigned char *)
	  ((long) (data_region_end + syspage_mask) & ~syspage_mask);
    }
  
  return result;
}

#ifdef __CYGWIN__
/* Emulate Unix getpagesize.  */
int getpagesize (void) { return 4096; }
#endif

/* Recreate the heap from the data that was dumped to the executable.
   EXECUTABLE_PATH tells us where to find the executable.  */
void
recreate_heap (char *executable_path) {

  unsigned char *tmp;

  /* First reserve the upper part of our heap.  (We reserve first
     because there have been problems in the past where doing the
     mapping first has loaded DLLs into the VA space of our heap.)  */
  tmp = VirtualAlloc ((void *) get_heap_start (),
		      get_reserved_heap_size (),
		      MEM_RESERVE,
		      PAGE_NOACCESS);
  if (!tmp)
    do_gcl_abort();

  /* We read in the data for the .bss section from the executable
     first and map in the heap from the executable second to prevent
     any funny interactions between file I/O and file mapping.  */

  read_in_bss (executable_path);

  map_in_heap (executable_path);

  /* Update system version information to match current system.  */
  cache_system_info ();

}

/* Round the heap up to the given alignment.  */
void
round_heap (unsigned long align)
{
  unsigned long needs_to_be;
  unsigned long need_to_alloc;
  
  needs_to_be = (unsigned long) round_to_next (get_heap_end (), align);
  need_to_alloc = needs_to_be - (unsigned long) get_heap_end ();
  
  if (need_to_alloc) 
    sbrk (need_to_alloc);
}

#if (_MSC_VER >= 1000)

/* MSVC 4.2 invokes these functions from mainCRTStartup to initialize
   a heap via HeapCreate.  They are normally defined by the runtime,
   but we override them here so that the unnecessary HeapCreate call
   is not performed.  */

int __cdecl
_heap_init (void)
{
  /* Stepping through the assembly indicates that mainCRTStartup is
     expecting a nonzero success return value.  */
  return 1;
}

void __cdecl
_heap_term (void)

#endif



#ifdef UNIXSAVE
BOOL ctrl_c_handler (unsigned long type)
{
  extern void sigint(void);
  sigint();
  return 0;

}
#include "save.c"
#endif
