/*
 Copyright (C) 1994 M. Hagiya, W. Schelter, T. Yuasa
 Copyright (C) 2024 Camm Maguire

This file is part of GNU Common Lisp, herein referred to as GCL

GCL is free software; you can redistribute it and/or modify it under
the terms of the GNU LIBRARY GENERAL PUBLIC LICENSE as published by
the Free Software Foundation; either version 2, or (at your option)
any later version.

GCL is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Library General Public 
License for more details.

You should have received a copy of the GNU Library General Public License 
along with GCL; see the file COPYING.  If not, write to the Free Software
Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.

*/

/*
	unixtime.c
*/

#define IN_UNIXTIME

#include <unistd.h>

#include "include.h"
#include <sys/types.h>
#ifdef UNIX
/* all we want from this is HZ the number of clock ticks per second
which is usually 60 maybe 100 or something else. */
#undef PAGESIZE
#ifndef NO_SYS_PARAM_H
#include <sys/param.h>
#endif
#endif

#ifndef HZ
/* #define HZ 60 */
#define HZ 100
#endif

/* #define HZ1 (HZ > 100 ? 100 : HZ) */
#define HZ1 HZ

#ifdef USE_ATT_TIME
#  undef BSD
#  define ATT
#endif

#if defined __MINGW32__ || !defined NO_SYSTEM_TIME_ZONE

#  ifdef __MINGW32__
#    include <windows.h>
#    include <time.h>
#    include <sys/timeb.h>

static struct timeb t0;
int usleep1 ( unsigned int microseconds );
#undef usleep
#define usleep(x) usleep1(x)

#  endif

#endif /* __MINGW32__ or  !defined NO_SYSTEM_TIME_ZONE */

#ifdef BSD
#include <time.h>
#include <sys/timeb.h>
#ifndef NO_SYS_TIMES_H
#include <sys/times.h>
#endif
#include <sys/time.h>
/* static struct timeb beginning; */
#endif

#ifdef ATT
#include <sys/times.h>
static long beginning;
#endif

int
runtime(void)
{

#ifdef USE_INTERNAL_REAL_TIME_FOR_RUNTIME

#  ifdef __MINGW32__    
    struct timeb t;
    if ( t0.time == 0 ) {
        ftime(&t0);
    }
    ftime ( &t );
    return ( ( t.time - t0.time ) * HZ1 + ( (t.millitm) * HZ1 ) / 1000 );
#  else
#  error Need to return runtime without generating a fixnum (else GBC(t_fixnum) will loop)
#  endif
    
#else	
	{
	  struct tms buf;
	  times(&buf);
	  return(buf.tms_utime);
	}
#endif
}

object
unix_time_to_universal_time(int i)
{
	object x;
	vs_mark;

	vs_push(make_fixnum(24*60*60));
	vs_push(make_fixnum(70*365+17));
	x = number_times(vs_top[-1], vs_top[-2]);
	vs_push(x);
	vs_push(make_fixnum(i));
	x = number_plus(vs_top[-1], vs_top[-2]);
	vs_reset;
	return(x);
}

DEFUN("GET-UNIVERSAL-TIME",object,fLget_universal_time,LISP
   ,0,0,NONE,OO,OO,OO,OO,(void),"")
{
	/* 0 args */
	RETURN1(unix_time_to_universal_time(time(0)));
}

LFD(Lsleep)(void) {

  useconds_t um=-1,ul=um/1000000;
  double d;

  check_arg(1);
  check_type_or_rational_float(&vs_base[0]);
  if (number_minusp(vs_base[0]) == TRUE)
    FEerror("~S is not a non-negative number.", 1, vs_base[0]);
  d=number_to_double(vs_base[0]);
  d=d<1 ? 0 : d;
  usleep(d>ul ? um : d*1000000);
  vs_top = vs_base;
  vs_push(Cnil);

}

DEFUNM("GET-INTERNAL-RUN-TIMES",object,fSget_internal_run_times,SI,0,0,NONE,OO,OO,OO,OO,(),"") {

  object *base=vs_top;

#ifdef USE_INTERNAL_REAL_TIME_FOR_RUNTIME
  RETURN2(fLget_internal_real_time(),small_fixnum(0));
#else
  struct tms buf;
  fixnum vals=(fixnum)fcall.valp;
  
  times(&buf);
  RETURN4(make_fixnum(buf.tms_utime),make_fixnum(buf.tms_cutime),make_fixnum(buf.tms_stime),make_fixnum(buf.tms_cstime));
  
#endif	
  
}

DEFUN("GET-INTERNAL-RUN-TIME",object,fLget_internal_run_time,LISP
	   ,0,0,NONE,OO,OO,OO,OO,(void),"") {
  object x=(fcall.valp=0,FFN(fSget_internal_run_times)());
  RETURN1(x);
}


DEFUN("GETTIMEOFDAY",object,fSgettimeofday,SI,0,0,NONE,OO,OO,OO,OO,(void),"Return time with maximum resolution") {
#ifdef __MINGW32__
  LARGE_INTEGER uu,ticks;
  if (QueryPerformanceFrequency(&ticks)) {
    QueryPerformanceCounter(&uu);
    return make_longfloat((longfloat)uu.QuadPart/ticks.QuadPart);
  } else {
    FEerror("microsecond timing not available",0);
    return Cnil;
  }
#endif  
#ifdef BSD
  struct timeval tzp;
  gettimeofday(&tzp,0);
  return make_longfloat((longfloat)tzp.tv_sec+1.0e-6*tzp.tv_usec);
#endif
#ifdef ATT
  return make_longfloat((longfloat)time(0));
#endif
}
#ifdef STATIC_FUNCTION_POINTERS
object
fSgettimeofday() {
  return FFN(fSgettimeofday)();
}
#endif


DEFUN("GET-INTERNAL-REAL-TIME",object,fLget_internal_real_time,LISP,0,0,NONE,OO,OO,OO,OO,(void),"Run time relative to beginning")
     
{
#ifdef __MINGW32__
    struct timeb t;
    if ( t0.time == 0 ) {
        ftime ( &t0 );
    }
    ftime(&t);
    return ( make_fixnum ( ( t.time - t0.time ) * HZ1 + ( (t.millitm) * HZ1 ) / 1000 ) );
#endif  
#ifdef BSD
	static struct timeval begin_tzp;
	struct timeval tzp;
	if (begin_tzp.tv_sec==0)
	  gettimeofday(&begin_tzp,0);
	gettimeofday(&tzp,0);
/* the value returned will be relative to the first time this is called,
   plus the fraction of a second.  We must make it relative, so this
   will only wrap if the process lasts longer than 818 days
   */
	return make_fixnum(((tzp.tv_sec-begin_tzp.tv_sec)*HZ1
			    + ((tzp.tv_usec)*HZ1)/1000000));
#endif
#ifdef ATT
	return make_fixnum((time(0) - beginning)*HZ1);
#endif
}


void
gcl_init_unixtime(void) {
#ifdef ATT
  beginning = time(0);
#endif
#if defined __MINGW32__
  ftime(&t0);
#endif        
  
  make_constant("INTERNAL-TIME-UNITS-PER-SECOND", make_fixnum(HZ1));
  make_function("SLEEP", Lsleep);

}

#ifdef __MINGW32__
int usleep1 ( unsigned int microseconds )
{
    unsigned int milliseconds = microseconds / 1000;
    return ( SleepEx ( milliseconds, TRUE ) );
}

#endif


DEFUN("CURRENT-TIMEZONE",object,fScurrent_timezone,SI,0,0,NONE,IO,OO,OO,OO,(void),"") {

#if defined(__MINGW32__)

  TIME_ZONE_INFORMATION tzi;
  DWORD TZResult;
  
  TZResult = GetTimeZoneInformation ( &tzi );
  
  /* Now UTC = (local time + bias), in units of minutes, so */
  /*fprintf ( stderr, "Bias = %ld\n", tzi.Bias );*/
  return (object)((tzi.Bias+tzi.DaylightBias)/60);
  
#elif defined NO_SYSTEM_TIME_ZONE
  return (object)0;
#elif defined __CYGWIN__
  struct tm gt,lt;
  fixnum _t=time(0);
  gmtime_r(&_t, &gt);
  localtime_r(&_t, &lt);
  return (object)(long)(gt.tm_hour-lt.tm_hour+24*(gt.tm_yday!=lt.tm_yday ? (gt.tm_year>lt.tm_year||gt.tm_yday>lt.tm_yday ? 1 : -1) : 0));
#else
  time_t _t=time(0);
  return (object)(-localtime(&_t)->tm_gmtoff/3600);
#endif
}

DEFUN("CURRENT-DSTP",object,fScurrent_dstp,SI,0,0,NONE,OO,OO,OO,OO,(void),"") {

#if defined(__MINGW32__)

  return Cnil;

#elif defined NO_SYSTEM_TIME_ZONE /*solaris*/
  return Cnil;
#else
  time_t _t=time(0);
  return localtime(&_t)->tm_isdst > 0 ? Ct : Cnil;
#endif
}

static object
time_t_to_object(time_t l) {
  object x=new_bignum();

  mpz_set_si(MP(x),l>>32);
  mpz_mul_2exp(MP(x),MP(x),32);
  mpz_add_ui(MP(x),MP(x),l&((1ULL<<32)-1));
  return normalize_big(x);

}

static time_t
object_to_time_t(object x) {

  switch(type_of(x)) {
  case t_fixnum:
    return fix(x);
  case t_bignum:
    {
      time_t h;
      mpz_set_si(MP(big_fixnum3),1);
      mpz_mul_2exp(MP(big_fixnum3),MP(big_fixnum3),31);
      mpz_fdiv_qr(MP(big_fixnum1),MP(big_fixnum2),MP(x),MP(big_fixnum3));
      massert(mpz_fits_slong_p(MP(big_fixnum1)));
      massert(mpz_fits_slong_p(MP(big_fixnum2)));
      h=mpz_get_si(MP(big_fixnum1));
      h<<=31;
      h+=mpz_get_si(MP(big_fixnum2));
      return h;
    }
  default:
    TYPE_ERROR(x,sLinteger);
  }

}

DEFUNM("LOCALTIME",object,fSlocaltime,SI,1,1,NONE,OO,OO,OO,OO,(object t),"") {

  fixnum vals=(fixnum)fcall.valp;
  object *base=vs_top;

#if defined NO_SYSTEM_TIME_ZONE /*solaris*/
  return Cnil;
#else

  time_t i=object_to_time_t(t);
  struct tm *lt;
  object zn;
#if defined(__MINGW32__)
  struct tm *gt;
  fixnum gmt_hour;
  massert(gt=gmtime(&i));
  gmt_hour=gt->tm_hour;
#endif

  massert(lt=localtime(&i));
  zn=make_simple_string(lt->tm_zone);

  RETURN(11,object,
	 make_fixnum(lt->tm_sec),
	 (
	  RV(make_fixnum(lt->tm_min)),
	  RV(make_fixnum(lt->tm_hour)),
	  RV(make_fixnum(lt->tm_mday)),
	  RV(make_fixnum(lt->tm_mon)),
	  RV(make_fixnum(lt->tm_year)),
	  RV(make_fixnum(lt->tm_wday)),
	  RV(make_fixnum(lt->tm_yday)),
	  RV(make_fixnum(lt->tm_isdst)),
#if defined(__MINGW32__)
	  RV(make_fixnum((lt->tm_hour-gmt_hour)*3600)),
	  RV(Cnil)
#else
	  RV(make_fixnum(lt->tm_gmtoff)),
	  RV(zn)/*make_simple_string(lt->tm_zone)*/
#endif
	  ));
#endif
}


DEFUNM("GMTIME",object,fSgmtime,SI,1,1,NONE,OO,OO,OO,OO,(object t),"") {

  fixnum vals=(fixnum)fcall.valp;
  object *base=vs_top;

#if defined NO_SYSTEM_TIME_ZONE /*solaris*/
  return Cnil;
#else

  time_t i=object_to_time_t(t);
  struct tm *gt;
  object zn;

  massert(gt=gmtime(&i));
  zn=make_simple_string(gt->tm_zone);

  RETURN(11,object,
	 make_fixnum(gt->tm_sec),
	 (
	  RV(make_fixnum(gt->tm_min)),
	  RV(make_fixnum(gt->tm_hour)),
	  RV(make_fixnum(gt->tm_mday)),
	  RV(make_fixnum(gt->tm_mon)),
	  RV(make_fixnum(gt->tm_year)),
	  RV(make_fixnum(gt->tm_wday)),
	  RV(make_fixnum(gt->tm_yday)),
	  RV(make_fixnum(gt->tm_isdst)),
#if defined(__MINGW32__)
	  RV(make_fixnum(0)),
	  RV(Cnil)
#else
	  RV(make_fixnum(gt->tm_gmtoff)),
	  RV(zn)/*make_simple_string(gt->tm_zone)*/
#endif
	  ));
#endif
}


DEFUNM("MKTIME",object,fSmktime,SI,7,7,NONE,OI,II,II,II,
       (fixnum s,fixnum n,fixnum h,fixnum d,fixnum m,fixnum y,fixnum isdst),"") {

  struct tm lt;
  time_t t;
  fixnum vals=(fixnum)fcall.valp;
  object *base=vs_top;

  lt.tm_sec=s;
  lt.tm_min=n;
  lt.tm_hour=h;
  lt.tm_mday=d;
  lt.tm_mon=m;
  lt.tm_year=y;
  lt.tm_isdst=isdst;

  massert((t=mktime(&lt))!=-1);
  RETURN(2,object,time_t_to_object(t),(RV(make_fixnum(lt.tm_isdst))));

}

