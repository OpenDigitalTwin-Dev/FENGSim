/* Xutil-2.c           Hiep Huu Nguyen                         27 Aug 92 */

/* ; Copyright (c) 1994 Hiep Huu Nguyen and The University of Texas at Austin.

; See the files gnu.license and dec.copyright .

; This program is free software; you can redistribute it and/or modify
; it under the terms of the GNU General Public License as published by
; the Free Software Foundation; either version 1, or (at your option)
; any later version.

; This program is distributed in the hope that it will be useful,
; but WITHOUT ANY WARRANTY; without even the implied warranty of
; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
; GNU General Public License for more details.

; You should have received a copy of the GNU General Public License
; along with this program; if not, write to the Free Software
; Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

; Some of the files that interface to the Xlib are adapted from DEC/MIT files.
; See the file dec.copyright for details. */

#include <X11/Xlib.h>
#include <X11/Xresource.h>
#include <X11/keysym.h>

int IsKeypadKey(keysym) int keysym; { 
 return  (((unsigned)(keysym) >= XK_KP_Space) && ((unsigned)(keysym) <= XK_KP_Equal));}

int IsCursorKey(keysym) int keysym; { 
  return (((unsigned)(keysym) >= XK_Home)     && ((unsigned)(keysym) <  XK_Select));}

int IsPFKey(keysym) int keysym; { 
  return (((unsigned)(keysym) >= XK_KP_F1)     && ((unsigned)(keysym) <= XK_KP_F4));}

int IsFunctionKey(keysym) int keysym; { 
  return (((unsigned)(keysym) >= XK_F1)       && ((unsigned)(keysym) <= XK_F35));}

int IsMiscFunctionKey(keysym) int keysym; { 
  return (((unsigned)(keysym) >= XK_Select)   && ((unsigned)(keysym) <  XK_KP_Space));}

int IsModifierKey(keysym) int keysym; { 
  return (((unsigned)(keysym) >= XK_Shift_L)  && ((unsigned)(keysym) <= XK_Hyper_R));}

int XUniqueContext() 
{
      	return( ((int)XrmUniqueQuark()) );
}

int XStringToContext(string) 
	char *string; 
{
   	return( (int)XrmStringToQuark(string) );
}

