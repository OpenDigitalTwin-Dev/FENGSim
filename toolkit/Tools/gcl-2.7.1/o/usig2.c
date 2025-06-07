/*
 Copyright (C) 1994  W. Schelter
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


#ifndef IN_UNIXINT
#define NEED_MP_H
#include <unistd.h>
#include "include.h"

static void
invoke_handler(int,int);


#ifndef USIG2
#include <signal.h>
#include "usig.h"
/*  #include "arith.h" */
#endif
#endif

#ifdef USIG2
#include USIG2
#else



/* these sstructure pointers would need their structures provided...
   so we just call them void */
void * sfaslp;

#ifdef CMAC
EXTER
unsigned long s4_neg_int[4],small_neg_int[3],small_pos_int[3];
#endif

 
/* 
   We have two mechanisms for protecting against interrupts.  1] We have a
   facility for delaying certain signals during critical regions of code.
   This facility will involve BEGIN_NO_INTERRUPT and END_NO_INTERRUPT

*/   

handler_function_type our_signal_handler[32];

struct save_for_interrupt {

  object free1[t_end];
  object free2[t_end];
  object altfree1[t_end];
  object altfree2[t_end];
  union lispunion buf[t_end];
  struct call_data fcall;
  object  *vs_top,vs_topVAL,*vs_base;
  struct bds_bd  *bds_top,bds_topVAL;
  struct  invocation_history *ihs_top,ihs_topVAL;
  char *token_bufp;
  char token_buf [4*INITIAL_TOKEN_LENGTH];
  int token_st_dim;
  /* for storing the XS objects in te usig2_aux.c */
  void *save_objects[75];
   
};


/* note these are the reverse of the ones in unixint.c
   ... uggghhh*/


#undef SS1
#undef RS1
#define SS1(a,b) a =  b ;
#define RS1(a,b) b = a ;

           /* save objects in save_objects list  */   


 
char signals_handled [] = {SIGINT,SIGUSR2,SIGUSR1,SIGIO,SIGALRM,
#ifdef OTHER_SIGNALS_HANDLED			   
			   OTHER_SIGNALS_HANDLED
#endif			   
			   0};

/* * in_signal_handler:   if not zero indicates we are running inside a signal
     handler, which may have been invoked at a random intruction, and so
     it is not safe to do a relocatable gc.   

   * signals_pending:   if (signals_pending & signal_mask(signo)) then this
     signo 's handler is waiting to be run.

   * signals_allowed:  indicates the state we think we were in when
      checking to invoke a signal.  Values:
      
      sig_none:    definitely dont run handler
      sig_normal:  In principle `ok', but if desiring maximum safety dont run here.
      sig_safe:    safe point to run a function (eg make_cons,...)
      sig_at_read: interrupting the getc function in read.  Should be safe.


      unwind (used by throw,return etc) resets this to sig_normal just as it
      does the longjmp.


   If we invoke signal handling routines at a storage
   allocation pt, it is completely safe:  we should save
   some of the globals, but the freelists etc dont need
   to be saved.   pass: sig_safe to raise_pending.

   If we invoke it at end of a No interrupts
   region, then it we must look at whether these were nested.
   We should probably have two endings for END_NO_INTERRUPTS,
   one for when we want to raise, and one for where we are sure
   we are at safe place.  pass sig_use_signals_allowed_value
   
   If we invoke a handler when at
   signals_allowed == sig_at_read, then we are safe.
   */


#define XX sig_safe
/* min safety level required for invoking a given signal handler  */
char safety_required[]={XX,XX,XX,XX,XX,XX,XX,XX,
			XX,XX,XX,XX,XX,XX,XX,XX,
			XX,XX,XX,XX,XX,XX,XX,XX,
			XX,XX,XX,XX,XX,XX,XX,XX};

void
gcl_init_safety(void)
{ safety_required[SIGINT]=sig_try_to_delay;
  safety_required[SIGALRM]=sig_normal;
  safety_required[SIGUSR1]=sig_normal;
}
  
DEFUN("SIGNAL-SAFETY-REQUIRED",object,sSsignal_safety_required,SI,2,2,
	  NONE,OI,IO,OO,OO,(fixnum signo,fixnum safety),
      "Set the safety level required for handling SIGNO to SAFETY, or if \
SAFETY is negative just return the current safety level for that \
signal number.  Value of 1 means allow interrupt at any place not \
specifically marked in the code as bad, and value of 2 means allow it \
only in very SAFE places.")

{ if (signo > sizeof(safety_required))
    {FEerror("Illegal signo:~a.",1,make_fixnum(signo));}
  if (safety >=0) safety_required[signo] = safety;
  return small_fixnum(safety_required[signo]) ;
}
     

void
main_signal_handler(int signo,siginfo_t *a,void *b)
{  int allowed = signals_allowed;
#ifdef NEED_TO_REINSTALL_SIGNALS
       signal(signo,main_signal_handler);
#endif
    if (allowed >= safety_required[signo])
     { signals_allowed = sig_none;
       
       if (signo == SIGUSR1 ||
	   signo == SIGIO)
	 { unblock_sigusr_sigio();}
	   
       invoke_handler(signo,allowed);
       signals_allowed = allowed;
      }
   else {
     signals_pending |= signal_mask(signo);
     alarm(1);}
   return;

 }

static void before_interrupt(struct save_for_interrupt *p, int allowed);
static void after_interrupt(struct save_for_interrupt *p, int allowed);

/* caller saves and restores the global signals_allowed; */
static void
invoke_handler(int signo, int allowed)
{struct save_for_interrupt buf;
 before_interrupt(&buf,allowed);
 signals_pending &= ~(signal_mask(signo));
 {int prev_in_handler = in_signal_handler;
  in_signal_handler |= (allowed <= sig_normal ? 1 : 0);
  signals_allowed = allowed;
  our_signal_handler[signo](signo,0,0);
  signals_allowed = 0;
  in_signal_handler = prev_in_handler;
  after_interrupt(&buf,allowed); 
}}

int tok_leng;

static void
before_interrupt(struct save_for_interrupt *p,int allowed) {

 /* all this must be run in no interrupts mode */
 if ( allowed < sig_safe) {

   int i;

   /* save tht tops of the free stacks */
   for (i=0;i<t_end;i++) {

     struct typemanager *ad = &tm_table[i];

     if ((p->free1[i]=ad->tm_free)) {

       void *beg=p->free1[i];
       object x=beg;
       int amt=ad->tm_size;

       memcpy(p->buf+i,beg,amt);
       bzero(beg+sizeof(struct freelist),amt-sizeof(struct freelist));/*FIXME t_free -> struct freelist **/
       ad->tm_nfree--;
       make_unfree(x);

       
       if ((p->free2[i]=OBJ_LINK(p->free1[i]))) {

	 beg=p->free2[i];
	 x=beg;
	 bzero(beg+sizeof(struct freelist),amt-sizeof(struct freelist));
	 ad->tm_nfree--;
	 make_unfree(x);

	 ad->tm_free=OBJ_LINK(p->free2[i]);

       } else

	 ad->tm_free=p->free2[i];

     }
   }
 }
   
 p->fcall=fcall;
 p->vs_top=vs_top;
 p->vs_topVAL=*vs_top;
 p->vs_base=vs_base;
 p->bds_top=bds_top;
 p->bds_topVAL=*bds_top;
 p->ihs_top=ihs_top;
 p->ihs_topVAL=*ihs_top;

 { 

   void **pp=p->save_objects;

#undef XS
#undef XSI
#define XS(a) *pp++ = (void *) (a);
#define XSI(a) *pp++ = (void *)(long)(a);

#include "usig2_aux.c"

   if (pp-p->save_objects>=(sizeof(p->save_objects)/sizeof(void *)))
     gcl_abort();

 }

 p->token_st_dim=tok_leng+1;
 if (token->st.st_dim<p->token_st_dim)
   p->token_st_dim=token->st.st_dim;
 if (p->token_st_dim<sizeof(p->token_buf))
   p->token_bufp=p->token_buf;
 else 
   p->token_bufp=(void *)ZALLOCA(p->token_st_dim);

 memcpy(p->token_bufp,token->st.st_self,p->token_st_dim);
  
}

static void
after_interrupt(struct save_for_interrupt *p,int allowed) {
  
  /* all this must be run in no interrupts mode */
  if ( allowed < sig_safe) {

    int i;
     for(i=0; i < t_end ; i++) {

       struct typemanager *ad = &tm_table[i];
       object current_fl = ad->tm_free;

       if ((ad->tm_free=p->free1[i])) { 

	 void *beg=p->free1[i];
	 object x=beg;
	 int amt=ad->tm_size;

	 if (is_marked_or_free(x)) error("should not be free");
	 memcpy(beg,p->buf+i,amt);

	 if ((p->free1[i]=p->free2[i])) { 

	   x=p->free2[i];
	   if (is_marked_or_free(x)) error("should not be free");

	   make_free(x);
	   F_LINK(F_LINK(ad->tm_free))=(long)current_fl;
	   ad->tm_nfree+=2;

	 } else
	   ad->tm_nfree=1;
       } else
	 ad->tm_nfree =0;
     }
  }

  fcall=p->fcall;
  vs_top=p->vs_top;
  *vs_top=p->vs_topVAL;
  vs_base=p->vs_base;
  bds_top=p->bds_top;
  *bds_top=p->bds_topVAL;
  ihs_top=p->ihs_top;
  *ihs_top=p->ihs_topVAL;

  { 

    void **pp=p->save_objects;

#undef XS
#undef XSI
#define XS(a) a=(void *)(*pp++)
#define XSI(a) {union {void *v;long l;} u; u.v=*pp++;a=u.l;}

#include "usig2_aux.c"

  }

  memcpy(token->st.st_self,p->token_bufp,p->token_st_dim);

}


/* claim the following version of make_cons can be interrupted at any line
   and is suitable for inlining.
*/

/* static object */
/* MakeCons(object a, object b) */
/* { struct typemanager*ad = &tm_table[t_cons]; */
/*   object new = (object) ad->tm_free; */
/*   if (new == 0) */
/*     { new = alloc_object(t_cons); */
/*       new->c.c_car = a; */
/*       goto END; */
/*     } */
      
/*   new->c.c_car=a; */
  /* interrupt here and before_interrupt will copy new->c into the
     C stack, so that a will be protected */
/*   new->c.t=t_cons; */
/*   new->c.m= 0; */
  /*  Make interrupt copy new out to the stack and then zero new.
      That way new is certainly gc valid, and its contents are protected.
      So the above three operations can occur in any order.
      */

/*   { object tem  = OBJ_LINK(new); */
    /*
      interrupt here and we see that before_interrupt must save the top of the
      free list AND the second thing on the Free list.  That way we will be ok
      here and an interrupt here could not affect tem.  It is possible that tem
      == 0, yet a gc happened in between.  An interrupt here when tem = 0 would
      mean the free list needs to be collected again by second gc.
      */
/*     ad->tm_free = tem; */
/*   } */
  /* Whew:  we got it safely off so interrupts can't hurt us now.  */
/*   ad->tm_nfree --; */
  /* interrupt here and the cdr field will point to a f_link which is
     a 'free' and so gc valid.   b is still protected since
     it is in the stack or a regiseter, and a is protected since it is
     in new, and new is not free
     */
/*  END: */
/*   new->c.c_cdr=b; */
/*   return new; */
/* } */


/* COND is the condition where this is raised.
   Might be sig_safe (eg at cons). */
   
void
raise_pending_signals(int cond)
{unsigned int allowed = signals_allowed ;
 if (cond == sig_use_signals_allowed_value)
 if (cond == sig_none  || interrupt_enable ==0) return ;
 
 
 AGAIN:
 { unsigned int pending = signals_pending;
   char *p = signals_handled;
   if (pending)
     while(*p)
       { if (signal_mask(*p) & pending
	     && cond >= safety_required[(unsigned char)*p])
	   {
	     signals_pending &= ~(signal_mask(*p));
	     if (*p == SIGALRM && cond >= sig_safe)
	       { alarm(0);}
	     else
	       invoke_handler(*p,cond);
	     goto AGAIN;
	   }
	   p++;
	 }
   signals_allowed = allowed; 
   return;
 }}


DEFUN("ALLOW-SIGNAL",object,fSallow_signal,SI,1,1,NONE,OI,OO,OO,OO,(fixnum n),
      "Install the default signal handler on signal N")

{

 signals_allowed |= signal_mask(n);
 unblock_signals(n,n);
 /* sys v ?? just restore the signal ?? */
 if (our_signal_handler[n])
   {gcl_signal(n,our_signal_handler[n]);
    return make_fixnum(1);
  }
 else
   return make_fixnum(0);
}



#endif
