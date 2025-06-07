/* Events.c           Hiep Huu Nguyen            27 Jun 06 */

/*; Copyright (c) 1994 Hiep Huu Nguyen and The University of Texas at Austin.
; Copyright (c) 2024 Camm Maguire
; edited 27 Aug 92; 12 Aug 2002; 23 Jun 06 by GSN; 27 Jun 06 by GSN
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

#include <stdlib.h>
#include <X11/Xlib.h>

/********* XKeyEvent functions *****/

long  make_XKeyEvent (){
          return ((long) calloc(1, sizeof(XKeyEvent)));
}

int  XKeyEvent_same_screen(i)
XKeyEvent* i;
{
          return(i->same_screen);
}

void set_XKeyEvent_same_screen(i, j)
XKeyEvent* i;
int j;
{
          i->same_screen = j;
}

int  XKeyEvent_keycode(i)
XKeyEvent* i;
{
          return(i->keycode);
}

void set_XKeyEvent_keycode(i, j)
XKeyEvent* i;
int j;
{
          i->keycode = j;
}

int  XKeyEvent_state(i)
XKeyEvent* i;
{
          return(i->state);
}

void set_XKeyEvent_state(i, j)
XKeyEvent* i;
int j;
{
          i->state = j;
}

int  XKeyEvent_y_root(i)
XKeyEvent* i;
{
          return(i->y_root);
}

void set_XKeyEvent_y_root(i, j)
XKeyEvent* i;
int j;
{
          i->y_root = j;
}

int  XKeyEvent_x_root(i)
XKeyEvent* i;
{
          return(i->x_root);
}

void set_XKeyEvent_x_root(i, j)
XKeyEvent* i;
int j;
{
          i->x_root = j;
}

int  XKeyEvent_y(i)
XKeyEvent* i;
{
          return(i->y);
}

void set_XKeyEvent_y(i, j)
XKeyEvent* i;
int j;
{
          i->y = j;
}

int  XKeyEvent_x(i)
XKeyEvent* i;
{
          return(i->x);
}

void set_XKeyEvent_x(i, j)
XKeyEvent* i;
int j;
{
          i->x = j;
}

int  XKeyEvent_time(i)
XKeyEvent* i;
{
          return(i->time);
}

void set_XKeyEvent_time(i, j)
XKeyEvent* i;
int j;
{
          i->time = j;
}

int  XKeyEvent_subwindow(i)
XKeyEvent* i;
{
          return(i->subwindow);
}

void set_XKeyEvent_subwindow(i, j)
XKeyEvent* i;
int j;
{
          i->subwindow = j;
}

int  XKeyEvent_root(i)
XKeyEvent* i;
{
          return(i->root);
}

void set_XKeyEvent_root(i, j)
XKeyEvent* i;
int j;
{
          i->root = j;
}

int  XKeyEvent_window(i)
XKeyEvent* i;
{
          return(i->window);
}

void set_XKeyEvent_window(i, j)
XKeyEvent* i;
int j;
{
          i->window = j;
}

long  XKeyEvent_display(i)
XKeyEvent* i;
{
          return((long) i->display);
}

void set_XKeyEvent_display(i, j)
XKeyEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XKeyEvent_send_event(i)
XKeyEvent* i;
{
          return(i->send_event);
}

void set_XKeyEvent_send_event(i, j)
XKeyEvent* i;
int j;
{
          i->send_event = j;
}

int  XKeyEvent_serial(i)
XKeyEvent* i;
{
          return(i->serial);
}

void set_XKeyEvent_serial(i, j)
XKeyEvent* i;
int j;
{
          i->serial = j;
}

int  XKeyEvent_type(i)
XKeyEvent* i;
{
          return(i->type);
}

void set_XKeyEvent_type(i, j)
XKeyEvent* i;
int j;
{
          i->type = j;
}


/********* XButtonEvent functions *****/

long  make_XButtonEvent (){
          return ((long) calloc(1, sizeof(XButtonEvent)));
}

int  XButtonEvent_same_screen(i)
XButtonEvent* i;
{
          return(i->same_screen);
}

void set_XButtonEvent_same_screen(i, j)
XButtonEvent* i;
int j;
{
          i->same_screen = j;
}

int  XButtonEvent_button(i)
XButtonEvent* i;
{
          return(i->button);
}

void set_XButtonEvent_button(i, j)
XButtonEvent* i;
int j;
{
          i->button = j;
}

int  XButtonEvent_state(i)
XButtonEvent* i;
{
          return(i->state);
}

void set_XButtonEvent_state(i, j)
XButtonEvent* i;
int j;
{
          i->state = j;
}

int  XButtonEvent_y_root(i)
XButtonEvent* i;
{
          return(i->y_root);
}

void set_XButtonEvent_y_root(i, j)
XButtonEvent* i;
int j;
{
          i->y_root = j;
}

int  XButtonEvent_x_root(i)
XButtonEvent* i;
{
          return(i->x_root);
}

void set_XButtonEvent_x_root(i, j)
XButtonEvent* i;
int j;
{
          i->x_root = j;
}

int  XButtonEvent_y(i)
XButtonEvent* i;
{
          return(i->y);
}

void set_XButtonEvent_y(i, j)
XButtonEvent* i;
int j;
{
          i->y = j;
}

int  XButtonEvent_x(i)
XButtonEvent* i;
{
          return(i->x);
}

void set_XButtonEvent_x(i, j)
XButtonEvent* i;
int j;
{
          i->x = j;
}

int  XButtonEvent_time(i)
XButtonEvent* i;
{
          return(i->time);
}

void set_XButtonEvent_time(i, j)
XButtonEvent* i;
int j;
{
          i->time = j;
}

int  XButtonEvent_subwindow(i)
XButtonEvent* i;
{
          return(i->subwindow);
}

void set_XButtonEvent_subwindow(i, j)
XButtonEvent* i;
int j;
{
          i->subwindow = j;
}

int  XButtonEvent_root(i)
XButtonEvent* i;
{
          return(i->root);
}

void set_XButtonEvent_root(i, j)
XButtonEvent* i;
int j;
{
          i->root = j;
}

int  XButtonEvent_window(i)
XButtonEvent* i;
{
          return(i->window);
}

void set_XButtonEvent_window(i, j)
XButtonEvent* i;
int j;
{
          i->window = j;
}

long  XButtonEvent_display(i)
XButtonEvent* i;
{
          return((long) i->display);
}

void set_XButtonEvent_display(i, j)
XButtonEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XButtonEvent_send_event(i)
XButtonEvent* i;
{
          return(i->send_event);
}

void set_XButtonEvent_send_event(i, j)
XButtonEvent* i;
int j;
{
          i->send_event = j;
}

int  XButtonEvent_serial(i)
XButtonEvent* i;
{
          return(i->serial);
}

void set_XButtonEvent_serial(i, j)
XButtonEvent* i;
int j;
{
          i->serial = j;
}

int  XButtonEvent_type(i)
XButtonEvent* i;
{
          return(i->type);
}

void set_XButtonEvent_type(i, j)
XButtonEvent* i;
int j;
{
          i->type = j;
}


/********* XMotionEvent functions *****/

long  make_XMotionEvent (){
          return ((long) calloc(1, sizeof(XMotionEvent)));
}

int  XMotionEvent_same_screen(i)
XMotionEvent* i;
{
          return(i->same_screen);
}

void set_XMotionEvent_same_screen(i, j)
XMotionEvent* i;
int j;
{
          i->same_screen = j;
}

char XMotionEvent_is_hint(i)
XMotionEvent* i;
{
          return(i->is_hint);
}

void set_XMotionEvent_is_hint(i, j)
XMotionEvent* i;
char j;
{
          i->is_hint = j;
}

int  XMotionEvent_state(i)
XMotionEvent* i;
{
          return(i->state);
}

void set_XMotionEvent_state(i, j)
XMotionEvent* i;
int j;
{
          i->state = j;
}

int  XMotionEvent_y_root(i)
XMotionEvent* i;
{
          return(i->y_root);
}

void set_XMotionEvent_y_root(i, j)
XMotionEvent* i;
int j;
{
          i->y_root = j;
}

int  XMotionEvent_x_root(i)
XMotionEvent* i;
{
          return(i->x_root);
}

void set_XMotionEvent_x_root(i, j)
XMotionEvent* i;
int j;
{
          i->x_root = j;
}

int  XMotionEvent_y(i)
XMotionEvent* i;
{
          return(i->y);
}

void set_XMotionEvent_y(i, j)
XMotionEvent* i;
int j;
{
          i->y = j;
}

int  XMotionEvent_x(i)
XMotionEvent* i;
{
          return(i->x);
}

void set_XMotionEvent_x(i, j)
XMotionEvent* i;
int j;
{
          i->x = j;
}

int  XMotionEvent_time(i)
XMotionEvent* i;
{
          return(i->time);
}

void set_XMotionEvent_time(i, j)
XMotionEvent* i;
int j;
{
          i->time = j;
}

int  XMotionEvent_subwindow(i)
XMotionEvent* i;
{
          return(i->subwindow);
}

void set_XMotionEvent_subwindow(i, j)
XMotionEvent* i;
int j;
{
          i->subwindow = j;
}

int  XMotionEvent_root(i)
XMotionEvent* i;
{
          return(i->root);
}

void set_XMotionEvent_root(i, j)
XMotionEvent* i;
int j;
{
          i->root = j;
}

int  XMotionEvent_window(i)
XMotionEvent* i;
{
          return(i->window);
}

void set_XMotionEvent_window(i, j)
XMotionEvent* i;
int j;
{
          i->window = j;
}

long  XMotionEvent_display(i)
XMotionEvent* i;
{
          return((long) i->display);
}

void set_XMotionEvent_display(i, j)
XMotionEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XMotionEvent_send_event(i)
XMotionEvent* i;
{
          return(i->send_event);
}

void set_XMotionEvent_send_event(i, j)
XMotionEvent* i;
int j;
{
          i->send_event = j;
}

int  XMotionEvent_serial(i)
XMotionEvent* i;
{
          return(i->serial);
}

void set_XMotionEvent_serial(i, j)
XMotionEvent* i;
int j;
{
          i->serial = j;
}

int  XMotionEvent_type(i)
XMotionEvent* i;
{
          return(i->type);
}

void set_XMotionEvent_type(i, j)
XMotionEvent* i;
int j;
{
          i->type = j;
}


/********* XCrossingEvent functions *****/

long  make_XCrossingEvent (){
          return ((long) calloc(1, sizeof(XCrossingEvent)));
}

int  XCrossingEvent_state(i)
XCrossingEvent* i;
{
          return(i->state);
}

void set_XCrossingEvent_state(i, j)
XCrossingEvent* i;
int j;
{
          i->state = j;
}

int  XCrossingEvent_focus(i)
XCrossingEvent* i;
{
          return(i->focus);
}

void set_XCrossingEvent_focus(i, j)
XCrossingEvent* i;
int j;
{
          i->focus = j;
}

int  XCrossingEvent_same_screen(i)
XCrossingEvent* i;
{
          return(i->same_screen);
}

void set_XCrossingEvent_same_screen(i, j)
XCrossingEvent* i;
int j;
{
          i->same_screen = j;
}

int  XCrossingEvent_detail(i)
XCrossingEvent* i;
{
          return(i->detail);
}

void set_XCrossingEvent_detail(i, j)
XCrossingEvent* i;
int j;
{
          i->detail = j;
}

int  XCrossingEvent_mode(i)
XCrossingEvent* i;
{
          return(i->mode);
}

void set_XCrossingEvent_mode(i, j)
XCrossingEvent* i;
int j;
{
          i->mode = j;
}

int  XCrossingEvent_y_root(i)
XCrossingEvent* i;
{
          return(i->y_root);
}

void set_XCrossingEvent_y_root(i, j)
XCrossingEvent* i;
int j;
{
          i->y_root = j;
}

int  XCrossingEvent_x_root(i)
XCrossingEvent* i;
{
          return(i->x_root);
}

void set_XCrossingEvent_x_root(i, j)
XCrossingEvent* i;
int j;
{
          i->x_root = j;
}

int  XCrossingEvent_y(i)
XCrossingEvent* i;
{
          return(i->y);
}

void set_XCrossingEvent_y(i, j)
XCrossingEvent* i;
int j;
{
          i->y = j;
}

int  XCrossingEvent_x(i)
XCrossingEvent* i;
{
          return(i->x);
}

void set_XCrossingEvent_x(i, j)
XCrossingEvent* i;
int j;
{
          i->x = j;
}

int  XCrossingEvent_time(i)
XCrossingEvent* i;
{
          return(i->time);
}

void set_XCrossingEvent_time(i, j)
XCrossingEvent* i;
int j;
{
          i->time = j;
}

int  XCrossingEvent_subwindow(i)
XCrossingEvent* i;
{
          return(i->subwindow);
}

void set_XCrossingEvent_subwindow(i, j)
XCrossingEvent* i;
int j;
{
          i->subwindow = j;
}

int  XCrossingEvent_root(i)
XCrossingEvent* i;
{
          return(i->root);
}

void set_XCrossingEvent_root(i, j)
XCrossingEvent* i;
int j;
{
          i->root = j;
}

int  XCrossingEvent_window(i)
XCrossingEvent* i;
{
          return(i->window);
}

void set_XCrossingEvent_window(i, j)
XCrossingEvent* i;
int j;
{
          i->window = j;
}

long  XCrossingEvent_display(i)
XCrossingEvent* i;
{
          return((long) i->display);
}

void set_XCrossingEvent_display(i, j)
XCrossingEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XCrossingEvent_send_event(i)
XCrossingEvent* i;
{
          return(i->send_event);
}

void set_XCrossingEvent_send_event(i, j)
XCrossingEvent* i;
int j;
{
          i->send_event = j;
}

int  XCrossingEvent_serial(i)
XCrossingEvent* i;
{
          return(i->serial);
}

void set_XCrossingEvent_serial(i, j)
XCrossingEvent* i;
int j;
{
          i->serial = j;
}

int  XCrossingEvent_type(i)
XCrossingEvent* i;
{
          return(i->type);
}

void set_XCrossingEvent_type(i, j)
XCrossingEvent* i;
int j;
{
          i->type = j;
}


/********* XFocusChangeEvent functions *****/

long  make_XFocusChangeEvent (){
          return ((long) calloc(1, sizeof(XFocusChangeEvent)));
}

int  XFocusChangeEvent_detail(i)
XFocusChangeEvent* i;
{
          return(i->detail);
}

void set_XFocusChangeEvent_detail(i, j)
XFocusChangeEvent* i;
int j;
{
          i->detail = j;
}

int  XFocusChangeEvent_mode(i)
XFocusChangeEvent* i;
{
          return(i->mode);
}

void set_XFocusChangeEvent_mode(i, j)
XFocusChangeEvent* i;
int j;
{
          i->mode = j;
}

int  XFocusChangeEvent_window(i)
XFocusChangeEvent* i;
{
          return(i->window);
}

void set_XFocusChangeEvent_window(i, j)
XFocusChangeEvent* i;
int j;
{
          i->window = j;
}

long  XFocusChangeEvent_display(i)
XFocusChangeEvent* i;
{
          return((long) i->display);
}

void set_XFocusChangeEvent_display(i, j)
XFocusChangeEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XFocusChangeEvent_send_event(i)
XFocusChangeEvent* i;
{
          return(i->send_event);
}

void set_XFocusChangeEvent_send_event(i, j)
XFocusChangeEvent* i;
int j;
{
          i->send_event = j;
}

int  XFocusChangeEvent_serial(i)
XFocusChangeEvent* i;
{
          return(i->serial);
}

void set_XFocusChangeEvent_serial(i, j)
XFocusChangeEvent* i;
int j;
{
          i->serial = j;
}

int  XFocusChangeEvent_type(i)
XFocusChangeEvent* i;
{
          return(i->type);
}

void set_XFocusChangeEvent_type(i, j)
XFocusChangeEvent* i;
int j;
{
          i->type = j;
}


/********* XKeymapEvent functions *****/

long  make_XKeymapEvent (){
          return ((long) calloc(1, sizeof(XKeymapEvent)));
}

char* XKeymapEvent_key_vector(i)
XKeymapEvent* i;
{
          return(i->key_vector);
}
int  XKeymapEvent_window(i)
XKeymapEvent* i;
{
          return(i->window);
}

void set_XKeymapEvent_window(i, j)
XKeymapEvent* i;
int j;
{
          i->window = j;
}

long  XKeymapEvent_display(i)
XKeymapEvent* i;
{
          return((long) i->display);
}

void set_XKeymapEvent_display(i, j)
XKeymapEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XKeymapEvent_send_event(i)
XKeymapEvent* i;
{
          return(i->send_event);
}

void set_XKeymapEvent_send_event(i, j)
XKeymapEvent* i;
int j;
{
          i->send_event = j;
}

int  XKeymapEvent_serial(i)
XKeymapEvent* i;
{
          return(i->serial);
}

void set_XKeymapEvent_serial(i, j)
XKeymapEvent* i;
int j;
{
          i->serial = j;
}

int  XKeymapEvent_type(i)
XKeymapEvent* i;
{
          return(i->type);
}

void set_XKeymapEvent_type(i, j)
XKeymapEvent* i;
int j;
{
          i->type = j;
}


/********* XExposeEvent functions *****/

long  make_XExposeEvent (){
          return ((long) calloc(1, sizeof(XExposeEvent)));
}

int  XExposeEvent_count(i)
XExposeEvent* i;
{
          return(i->count);
}

void set_XExposeEvent_count(i, j)
XExposeEvent* i;
int j;
{
          i->count = j;
}

int  XExposeEvent_height(i)
XExposeEvent* i;
{
          return(i->height);
}

void set_XExposeEvent_height(i, j)
XExposeEvent* i;
int j;
{
          i->height = j;
}

int  XExposeEvent_width(i)
XExposeEvent* i;
{
          return(i->width);
}

void set_XExposeEvent_width(i, j)
XExposeEvent* i;
int j;
{
          i->width = j;
}

int  XExposeEvent_y(i)
XExposeEvent* i;
{
          return(i->y);
}

void set_XExposeEvent_y(i, j)
XExposeEvent* i;
int j;
{
          i->y = j;
}

int  XExposeEvent_x(i)
XExposeEvent* i;
{
          return(i->x);
}

void set_XExposeEvent_x(i, j)
XExposeEvent* i;
int j;
{
          i->x = j;
}

int  XExposeEvent_window(i)
XExposeEvent* i;
{
          return(i->window);
}

void set_XExposeEvent_window(i, j)
XExposeEvent* i;
int j;
{
          i->window = j;
}

long  XExposeEvent_display(i)
XExposeEvent* i;
{
          return((long) i->display);
}

void set_XExposeEvent_display(i, j)
XExposeEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XExposeEvent_send_event(i)
XExposeEvent* i;
{
          return(i->send_event);
}

void set_XExposeEvent_send_event(i, j)
XExposeEvent* i;
int j;
{
          i->send_event = j;
}

int  XExposeEvent_serial(i)
XExposeEvent* i;
{
          return(i->serial);
}

void set_XExposeEvent_serial(i, j)
XExposeEvent* i;
int j;
{
          i->serial = j;
}

int  XExposeEvent_type(i)
XExposeEvent* i;
{
          return(i->type);
}

void set_XExposeEvent_type(i, j)
XExposeEvent* i;
int j;
{
          i->type = j;
}


/********* XGraphicsExposeEvent functions *****/

long  make_XGraphicsExposeEvent (){
          return ((long) calloc(1, sizeof(XGraphicsExposeEvent)));
}

int  XGraphicsExposeEvent_minor_code(i)
XGraphicsExposeEvent* i;
{
          return(i->minor_code);
}

void set_XGraphicsExposeEvent_minor_code(i, j)
XGraphicsExposeEvent* i;
int j;
{
          i->minor_code = j;
}

int  XGraphicsExposeEvent_major_code(i)
XGraphicsExposeEvent* i;
{
          return(i->major_code);
}

void set_XGraphicsExposeEvent_major_code(i, j)
XGraphicsExposeEvent* i;
int j;
{
          i->major_code = j;
}

int  XGraphicsExposeEvent_count(i)
XGraphicsExposeEvent* i;
{
          return(i->count);
}

void set_XGraphicsExposeEvent_count(i, j)
XGraphicsExposeEvent* i;
int j;
{
          i->count = j;
}

int  XGraphicsExposeEvent_height(i)
XGraphicsExposeEvent* i;
{
          return(i->height);
}

void set_XGraphicsExposeEvent_height(i, j)
XGraphicsExposeEvent* i;
int j;
{
          i->height = j;
}

int  XGraphicsExposeEvent_width(i)
XGraphicsExposeEvent* i;
{
          return(i->width);
}

void set_XGraphicsExposeEvent_width(i, j)
XGraphicsExposeEvent* i;
int j;
{
          i->width = j;
}

int  XGraphicsExposeEvent_y(i)
XGraphicsExposeEvent* i;
{
          return(i->y);
}

void set_XGraphicsExposeEvent_y(i, j)
XGraphicsExposeEvent* i;
int j;
{
          i->y = j;
}

int  XGraphicsExposeEvent_x(i)
XGraphicsExposeEvent* i;
{
          return(i->x);
}

void set_XGraphicsExposeEvent_x(i, j)
XGraphicsExposeEvent* i;
int j;
{
          i->x = j;
}

Drawable XGraphicsExposeEvent_drawable(i)
XGraphicsExposeEvent* i;
{
          return(i->drawable);
}

void set_XGraphicsExposeEvent_drawable(i, j)
XGraphicsExposeEvent* i;
Drawable j;
{
          i->drawable = j;
}

long  XGraphicsExposeEvent_display(i)
XGraphicsExposeEvent* i;
{
          return((long) i->display);
}

void set_XGraphicsExposeEvent_display(i, j)
XGraphicsExposeEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XGraphicsExposeEvent_send_event(i)
XGraphicsExposeEvent* i;
{
          return(i->send_event);
}

void set_XGraphicsExposeEvent_send_event(i, j)
XGraphicsExposeEvent* i;
int j;
{
          i->send_event = j;
}

int  XGraphicsExposeEvent_serial(i)
XGraphicsExposeEvent* i;
{
          return(i->serial);
}

void set_XGraphicsExposeEvent_serial(i, j)
XGraphicsExposeEvent* i;
int j;
{
          i->serial = j;
}

int  XGraphicsExposeEvent_type(i)
XGraphicsExposeEvent* i;
{
          return(i->type);
}

void set_XGraphicsExposeEvent_type(i, j)
XGraphicsExposeEvent* i;
int j;
{
          i->type = j;
}


/********* XNoExposeEvent functions *****/

long  make_XNoExposeEvent (){
          return ((long) calloc(1, sizeof(XNoExposeEvent)));
}

int  XNoExposeEvent_minor_code(i)
XNoExposeEvent* i;
{
          return(i->minor_code);
}

void set_XNoExposeEvent_minor_code(i, j)
XNoExposeEvent* i;
int j;
{
          i->minor_code = j;
}

int  XNoExposeEvent_major_code(i)
XNoExposeEvent* i;
{
          return(i->major_code);
}

void set_XNoExposeEvent_major_code(i, j)
XNoExposeEvent* i;
int j;
{
          i->major_code = j;
}

Drawable XNoExposeEvent_drawable(i)
XNoExposeEvent* i;
{
          return(i->drawable);
}

void set_XNoExposeEvent_drawable(i, j)
XNoExposeEvent* i;
Drawable j;
{
          i->drawable = j;
}

long  XNoExposeEvent_display(i)
XNoExposeEvent* i;
{
          return((long) i->display);
}

void set_XNoExposeEvent_display(i, j)
XNoExposeEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XNoExposeEvent_send_event(i)
XNoExposeEvent* i;
{
          return(i->send_event);
}

void set_XNoExposeEvent_send_event(i, j)
XNoExposeEvent* i;
int j;
{
          i->send_event = j;
}

int  XNoExposeEvent_serial(i)
XNoExposeEvent* i;
{
          return(i->serial);
}

void set_XNoExposeEvent_serial(i, j)
XNoExposeEvent* i;
int j;
{
          i->serial = j;
}

int  XNoExposeEvent_type(i)
XNoExposeEvent* i;
{
          return(i->type);
}

void set_XNoExposeEvent_type(i, j)
XNoExposeEvent* i;
int j;
{
          i->type = j;
}


/********* XVisibilityEvent functions *****/

long  make_XVisibilityEvent (){
          return ((long) calloc(1, sizeof(XVisibilityEvent)));
}

int  XVisibilityEvent_state(i)
XVisibilityEvent* i;
{
          return(i->state);
}

void set_XVisibilityEvent_state(i, j)
XVisibilityEvent* i;
int j;
{
          i->state = j;
}

int  XVisibilityEvent_window(i)
XVisibilityEvent* i;
{
          return(i->window);
}

void set_XVisibilityEvent_window(i, j)
XVisibilityEvent* i;
int j;
{
          i->window = j;
}

long  XVisibilityEvent_display(i)
XVisibilityEvent* i;
{
          return((long) i->display);
}

void set_XVisibilityEvent_display(i, j)
XVisibilityEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XVisibilityEvent_send_event(i)
XVisibilityEvent* i;
{
          return(i->send_event);
}

void set_XVisibilityEvent_send_event(i, j)
XVisibilityEvent* i;
int j;
{
          i->send_event = j;
}

int  XVisibilityEvent_serial(i)
XVisibilityEvent* i;
{
          return(i->serial);
}

void set_XVisibilityEvent_serial(i, j)
XVisibilityEvent* i;
int j;
{
          i->serial = j;
}

int  XVisibilityEvent_type(i)
XVisibilityEvent* i;
{
          return(i->type);
}

void set_XVisibilityEvent_type(i, j)
XVisibilityEvent* i;
int j;
{
          i->type = j;
}


/********* XCreateWindowEvent functions *****/

long  make_XCreateWindowEvent (){
          return ((long) calloc(1, sizeof(XCreateWindowEvent)));
}

int  XCreateWindowEvent_override_redirect(i)
XCreateWindowEvent* i;
{
          return(i->override_redirect);
}

void set_XCreateWindowEvent_override_redirect(i, j)
XCreateWindowEvent* i;
int j;
{
          i->override_redirect = j;
}

int  XCreateWindowEvent_border_width(i)
XCreateWindowEvent* i;
{
          return(i->border_width);
}

void set_XCreateWindowEvent_border_width(i, j)
XCreateWindowEvent* i;
int j;
{
          i->border_width = j;
}

int  XCreateWindowEvent_height(i)
XCreateWindowEvent* i;
{
          return(i->height);
}

void set_XCreateWindowEvent_height(i, j)
XCreateWindowEvent* i;
int j;
{
          i->height = j;
}

int  XCreateWindowEvent_width(i)
XCreateWindowEvent* i;
{
          return(i->width);
}

void set_XCreateWindowEvent_width(i, j)
XCreateWindowEvent* i;
int j;
{
          i->width = j;
}

int  XCreateWindowEvent_y(i)
XCreateWindowEvent* i;
{
          return(i->y);
}

void set_XCreateWindowEvent_y(i, j)
XCreateWindowEvent* i;
int j;
{
          i->y = j;
}

int  XCreateWindowEvent_x(i)
XCreateWindowEvent* i;
{
          return(i->x);
}

void set_XCreateWindowEvent_x(i, j)
XCreateWindowEvent* i;
int j;
{
          i->x = j;
}

int  XCreateWindowEvent_window(i)
XCreateWindowEvent* i;
{
          return(i->window);
}

void set_XCreateWindowEvent_window(i, j)
XCreateWindowEvent* i;
int j;
{
          i->window = j;
}

int  XCreateWindowEvent_parent(i)
XCreateWindowEvent* i;
{
          return(i->parent);
}

void set_XCreateWindowEvent_parent(i, j)
XCreateWindowEvent* i;
int j;
{
          i->parent = j;
}

long  XCreateWindowEvent_display(i)
XCreateWindowEvent* i;
{
          return((long) i->display);
}

void set_XCreateWindowEvent_display(i, j)
XCreateWindowEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XCreateWindowEvent_send_event(i)
XCreateWindowEvent* i;
{
          return(i->send_event);
}

void set_XCreateWindowEvent_send_event(i, j)
XCreateWindowEvent* i;
int j;
{
          i->send_event = j;
}

int  XCreateWindowEvent_serial(i)
XCreateWindowEvent* i;
{
          return(i->serial);
}

void set_XCreateWindowEvent_serial(i, j)
XCreateWindowEvent* i;
int j;
{
          i->serial = j;
}

int  XCreateWindowEvent_type(i)
XCreateWindowEvent* i;
{
          return(i->type);
}

void set_XCreateWindowEvent_type(i, j)
XCreateWindowEvent* i;
int j;
{
          i->type = j;
}


/********* XDestroyWindowEvent functions *****/

long  make_XDestroyWindowEvent (){
          return ((long) calloc(1, sizeof(XDestroyWindowEvent)));
}

int  XDestroyWindowEvent_window(i)
XDestroyWindowEvent* i;
{
          return(i->window);
}

void set_XDestroyWindowEvent_window(i, j)
XDestroyWindowEvent* i;
int j;
{
          i->window = j;
}

int  XDestroyWindowEvent_event(i)
XDestroyWindowEvent* i;
{
          return(i->event);
}

void set_XDestroyWindowEvent_event(i, j)
XDestroyWindowEvent* i;
int j;
{
          i->event = j;
}

long  XDestroyWindowEvent_display(i)
XDestroyWindowEvent* i;
{
          return((long) i->display);
}

void set_XDestroyWindowEvent_display(i, j)
XDestroyWindowEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XDestroyWindowEvent_send_event(i)
XDestroyWindowEvent* i;
{
          return(i->send_event);
}

void set_XDestroyWindowEvent_send_event(i, j)
XDestroyWindowEvent* i;
int j;
{
          i->send_event = j;
}

int  XDestroyWindowEvent_serial(i)
XDestroyWindowEvent* i;
{
          return(i->serial);
}

void set_XDestroyWindowEvent_serial(i, j)
XDestroyWindowEvent* i;
int j;
{
          i->serial = j;
}

int  XDestroyWindowEvent_type(i)
XDestroyWindowEvent* i;
{
          return(i->type);
}

void set_XDestroyWindowEvent_type(i, j)
XDestroyWindowEvent* i;
int j;
{
          i->type = j;
}


/********* XUnmapEvent functions *****/

long  make_XUnmapEvent (){
          return ((long) calloc(1, sizeof(XUnmapEvent)));
}

int  XUnmapEvent_from_configure(i)
XUnmapEvent* i;
{
          return(i->from_configure);
}

void set_XUnmapEvent_from_configure(i, j)
XUnmapEvent* i;
int j;
{
          i->from_configure = j;
}

int  XUnmapEvent_window(i)
XUnmapEvent* i;
{
          return(i->window);
}

void set_XUnmapEvent_window(i, j)
XUnmapEvent* i;
int j;
{
          i->window = j;
}

int  XUnmapEvent_event(i)
XUnmapEvent* i;
{
          return(i->event);
}

void set_XUnmapEvent_event(i, j)
XUnmapEvent* i;
int j;
{
          i->event = j;
}

long  XUnmapEvent_display(i)
XUnmapEvent* i;
{
          return((long) i->display);
}

void set_XUnmapEvent_display(i, j)
XUnmapEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XUnmapEvent_send_event(i)
XUnmapEvent* i;
{
          return(i->send_event);
}

void set_XUnmapEvent_send_event(i, j)
XUnmapEvent* i;
int j;
{
          i->send_event = j;
}

int  XUnmapEvent_serial(i)
XUnmapEvent* i;
{
          return(i->serial);
}

void set_XUnmapEvent_serial(i, j)
XUnmapEvent* i;
int j;
{
          i->serial = j;
}

int  XUnmapEvent_type(i)
XUnmapEvent* i;
{
          return(i->type);
}

void set_XUnmapEvent_type(i, j)
XUnmapEvent* i;
int j;
{
          i->type = j;
}


/********* XMapEvent functions *****/

long  make_XMapEvent (){
          return ((long) calloc(1, sizeof(XMapEvent)));
}

int  XMapEvent_override_redirect(i)
XMapEvent* i;
{
          return(i->override_redirect);
}

void set_XMapEvent_override_redirect(i, j)
XMapEvent* i;
int j;
{
          i->override_redirect = j;
}

int  XMapEvent_window(i)
XMapEvent* i;
{
          return(i->window);
}

void set_XMapEvent_window(i, j)
XMapEvent* i;
int j;
{
          i->window = j;
}

int  XMapEvent_event(i)
XMapEvent* i;
{
          return(i->event);
}

void set_XMapEvent_event(i, j)
XMapEvent* i;
int j;
{
          i->event = j;
}

long  XMapEvent_display(i)
XMapEvent* i;
{
          return((long) i->display);
}

void set_XMapEvent_display(i, j)
XMapEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XMapEvent_send_event(i)
XMapEvent* i;
{
          return(i->send_event);
}

void set_XMapEvent_send_event(i, j)
XMapEvent* i;
int j;
{
          i->send_event = j;
}

int  XMapEvent_serial(i)
XMapEvent* i;
{
          return(i->serial);
}

void set_XMapEvent_serial(i, j)
XMapEvent* i;
int j;
{
          i->serial = j;
}

int  XMapEvent_type(i)
XMapEvent* i;
{
          return(i->type);
}

void set_XMapEvent_type(i, j)
XMapEvent* i;
int j;
{
          i->type = j;
}


/********* XMapRequestEvent functions *****/

long  make_XMapRequestEvent (){
          return ((long) calloc(1, sizeof(XMapRequestEvent)));
}

int  XMapRequestEvent_window(i)
XMapRequestEvent* i;
{
          return(i->window);
}

void set_XMapRequestEvent_window(i, j)
XMapRequestEvent* i;
int j;
{
          i->window = j;
}

int  XMapRequestEvent_parent(i)
XMapRequestEvent* i;
{
          return(i->parent);
}

void set_XMapRequestEvent_parent(i, j)
XMapRequestEvent* i;
int j;
{
          i->parent = j;
}

long  XMapRequestEvent_display(i)
XMapRequestEvent* i;
{
          return((long) i->display);
}

void set_XMapRequestEvent_display(i, j)
XMapRequestEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XMapRequestEvent_send_event(i)
XMapRequestEvent* i;
{
          return(i->send_event);
}

void set_XMapRequestEvent_send_event(i, j)
XMapRequestEvent* i;
int j;
{
          i->send_event = j;
}

int  XMapRequestEvent_serial(i)
XMapRequestEvent* i;
{
          return(i->serial);
}

void set_XMapRequestEvent_serial(i, j)
XMapRequestEvent* i;
int j;
{
          i->serial = j;
}

int  XMapRequestEvent_type(i)
XMapRequestEvent* i;
{
          return(i->type);
}

void set_XMapRequestEvent_type(i, j)
XMapRequestEvent* i;
int j;
{
          i->type = j;
}


/********* XReparentEvent functions *****/

long  make_XReparentEvent (){
          return ((long) calloc(1, sizeof(XReparentEvent)));
}

int  XReparentEvent_override_redirect(i)
XReparentEvent* i;
{
          return(i->override_redirect);
}

void set_XReparentEvent_override_redirect(i, j)
XReparentEvent* i;
int j;
{
          i->override_redirect = j;
}

int  XReparentEvent_y(i)
XReparentEvent* i;
{
          return(i->y);
}

void set_XReparentEvent_y(i, j)
XReparentEvent* i;
int j;
{
          i->y = j;
}

int  XReparentEvent_x(i)
XReparentEvent* i;
{
          return(i->x);
}

void set_XReparentEvent_x(i, j)
XReparentEvent* i;
int j;
{
          i->x = j;
}

int  XReparentEvent_parent(i)
XReparentEvent* i;
{
          return(i->parent);
}

void set_XReparentEvent_parent(i, j)
XReparentEvent* i;
int j;
{
          i->parent = j;
}

int  XReparentEvent_window(i)
XReparentEvent* i;
{
          return(i->window);
}

void set_XReparentEvent_window(i, j)
XReparentEvent* i;
int j;
{
          i->window = j;
}

int  XReparentEvent_event(i)
XReparentEvent* i;
{
          return(i->event);
}

void set_XReparentEvent_event(i, j)
XReparentEvent* i;
int j;
{
          i->event = j;
}

long  XReparentEvent_display(i)
XReparentEvent* i;
{
          return((long) i->display);
}

void set_XReparentEvent_display(i, j)
XReparentEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XReparentEvent_send_event(i)
XReparentEvent* i;
{
          return(i->send_event);
}

void set_XReparentEvent_send_event(i, j)
XReparentEvent* i;
int j;
{
          i->send_event = j;
}

int  XReparentEvent_serial(i)
XReparentEvent* i;
{
          return(i->serial);
}

void set_XReparentEvent_serial(i, j)
XReparentEvent* i;
int j;
{
          i->serial = j;
}

int  XReparentEvent_type(i)
XReparentEvent* i;
{
          return(i->type);
}

void set_XReparentEvent_type(i, j)
XReparentEvent* i;
int j;
{
          i->type = j;
}


/********* XConfigureEvent functions *****/

long  make_XConfigureEvent (){
          return ((long) calloc(1, sizeof(XConfigureEvent)));
}

int  XConfigureEvent_override_redirect(i)
XConfigureEvent* i;
{
          return(i->override_redirect);
}

void set_XConfigureEvent_override_redirect(i, j)
XConfigureEvent* i;
int j;
{
          i->override_redirect = j;
}

int  XConfigureEvent_above(i)
XConfigureEvent* i;
{
          return(i->above);
}

void set_XConfigureEvent_above(i, j)
XConfigureEvent* i;
int j;
{
          i->above = j;
}

int  XConfigureEvent_border_width(i)
XConfigureEvent* i;
{
          return(i->border_width);
}

void set_XConfigureEvent_border_width(i, j)
XConfigureEvent* i;
int j;
{
          i->border_width = j;
}

int  XConfigureEvent_height(i)
XConfigureEvent* i;
{
          return(i->height);
}

void set_XConfigureEvent_height(i, j)
XConfigureEvent* i;
int j;
{
          i->height = j;
}

int  XConfigureEvent_width(i)
XConfigureEvent* i;
{
          return(i->width);
}

void set_XConfigureEvent_width(i, j)
XConfigureEvent* i;
int j;
{
          i->width = j;
}

int  XConfigureEvent_y(i)
XConfigureEvent* i;
{
          return(i->y);
}

void set_XConfigureEvent_y(i, j)
XConfigureEvent* i;
int j;
{
          i->y = j;
}

int  XConfigureEvent_x(i)
XConfigureEvent* i;
{
          return(i->x);
}

void set_XConfigureEvent_x(i, j)
XConfigureEvent* i;
int j;
{
          i->x = j;
}

int  XConfigureEvent_window(i)
XConfigureEvent* i;
{
          return(i->window);
}

void set_XConfigureEvent_window(i, j)
XConfigureEvent* i;
int j;
{
          i->window = j;
}

int  XConfigureEvent_event(i)
XConfigureEvent* i;
{
          return(i->event);
}

void set_XConfigureEvent_event(i, j)
XConfigureEvent* i;
int j;
{
          i->event = j;
}

long  XConfigureEvent_display(i)
XConfigureEvent* i;
{
          return((long) i->display);
}

void set_XConfigureEvent_display(i, j)
XConfigureEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XConfigureEvent_send_event(i)
XConfigureEvent* i;
{
          return(i->send_event);
}

void set_XConfigureEvent_send_event(i, j)
XConfigureEvent* i;
int j;
{
          i->send_event = j;
}

int  XConfigureEvent_serial(i)
XConfigureEvent* i;
{
          return(i->serial);
}

void set_XConfigureEvent_serial(i, j)
XConfigureEvent* i;
int j;
{
          i->serial = j;
}

int  XConfigureEvent_type(i)
XConfigureEvent* i;
{
          return(i->type);
}

void set_XConfigureEvent_type(i, j)
XConfigureEvent* i;
int j;
{
          i->type = j;
}


/********* XGravityEvent functions *****/

long  make_XGravityEvent (){
          return ((long) calloc(1, sizeof(XGravityEvent)));
}

int  XGravityEvent_y(i)
XGravityEvent* i;
{
          return(i->y);
}

void set_XGravityEvent_y(i, j)
XGravityEvent* i;
int j;
{
          i->y = j;
}

int  XGravityEvent_x(i)
XGravityEvent* i;
{
          return(i->x);
}

void set_XGravityEvent_x(i, j)
XGravityEvent* i;
int j;
{
          i->x = j;
}

int  XGravityEvent_window(i)
XGravityEvent* i;
{
          return(i->window);
}

void set_XGravityEvent_window(i, j)
XGravityEvent* i;
int j;
{
          i->window = j;
}

int  XGravityEvent_event(i)
XGravityEvent* i;
{
          return(i->event);
}

void set_XGravityEvent_event(i, j)
XGravityEvent* i;
int j;
{
          i->event = j;
}

long  XGravityEvent_display(i)
XGravityEvent* i;
{
          return((long) i->display);
}

void set_XGravityEvent_display(i, j)
XGravityEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XGravityEvent_send_event(i)
XGravityEvent* i;
{
          return(i->send_event);
}

void set_XGravityEvent_send_event(i, j)
XGravityEvent* i;
int j;
{
          i->send_event = j;
}

int  XGravityEvent_serial(i)
XGravityEvent* i;
{
          return(i->serial);
}

void set_XGravityEvent_serial(i, j)
XGravityEvent* i;
int j;
{
          i->serial = j;
}

int  XGravityEvent_type(i)
XGravityEvent* i;
{
          return(i->type);
}

void set_XGravityEvent_type(i, j)
XGravityEvent* i;
int j;
{
          i->type = j;
}


/********* XResizeRequestEvent functions *****/

long  make_XResizeRequestEvent (){
          return ((long) calloc(1, sizeof(XResizeRequestEvent)));
}

int  XResizeRequestEvent_height(i)
XResizeRequestEvent* i;
{
          return(i->height);
}

void set_XResizeRequestEvent_height(i, j)
XResizeRequestEvent* i;
int j;
{
          i->height = j;
}

int  XResizeRequestEvent_width(i)
XResizeRequestEvent* i;
{
          return(i->width);
}

void set_XResizeRequestEvent_width(i, j)
XResizeRequestEvent* i;
int j;
{
          i->width = j;
}

int  XResizeRequestEvent_window(i)
XResizeRequestEvent* i;
{
          return(i->window);
}

void set_XResizeRequestEvent_window(i, j)
XResizeRequestEvent* i;
int j;
{
          i->window = j;
}

long  XResizeRequestEvent_display(i)
XResizeRequestEvent* i;
{
          return((long) i->display);
}

void set_XResizeRequestEvent_display(i, j)
XResizeRequestEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XResizeRequestEvent_send_event(i)
XResizeRequestEvent* i;
{
          return(i->send_event);
}

void set_XResizeRequestEvent_send_event(i, j)
XResizeRequestEvent* i;
int j;
{
          i->send_event = j;
}

int  XResizeRequestEvent_serial(i)
XResizeRequestEvent* i;
{
          return(i->serial);
}

void set_XResizeRequestEvent_serial(i, j)
XResizeRequestEvent* i;
int j;
{
          i->serial = j;
}

int  XResizeRequestEvent_type(i)
XResizeRequestEvent* i;
{
          return(i->type);
}

void set_XResizeRequestEvent_type(i, j)
XResizeRequestEvent* i;
int j;
{
          i->type = j;
}


/********* XConfigureRequestEvent functions *****/

long  make_XConfigureRequestEvent (){
          return ((long) calloc(1, sizeof(XConfigureRequestEvent)));
}

int  XConfigureRequestEvent_value_mask(i)
XConfigureRequestEvent* i;
{
          return(i->value_mask);
}

void set_XConfigureRequestEvent_value_mask(i, j)
XConfigureRequestEvent* i;
int j;
{
          i->value_mask = j;
}

int  XConfigureRequestEvent_detail(i)
XConfigureRequestEvent* i;
{
          return(i->detail);
}

void set_XConfigureRequestEvent_detail(i, j)
XConfigureRequestEvent* i;
int j;
{
          i->detail = j;
}

int  XConfigureRequestEvent_above(i)
XConfigureRequestEvent* i;
{
          return(i->above);
}

void set_XConfigureRequestEvent_above(i, j)
XConfigureRequestEvent* i;
int j;
{
          i->above = j;
}

int  XConfigureRequestEvent_border_width(i)
XConfigureRequestEvent* i;
{
          return(i->border_width);
}

void set_XConfigureRequestEvent_border_width(i, j)
XConfigureRequestEvent* i;
int j;
{
          i->border_width = j;
}

int  XConfigureRequestEvent_height(i)
XConfigureRequestEvent* i;
{
          return(i->height);
}

void set_XConfigureRequestEvent_height(i, j)
XConfigureRequestEvent* i;
int j;
{
          i->height = j;
}

int  XConfigureRequestEvent_width(i)
XConfigureRequestEvent* i;
{
          return(i->width);
}

void set_XConfigureRequestEvent_width(i, j)
XConfigureRequestEvent* i;
int j;
{
          i->width = j;
}

int  XConfigureRequestEvent_y(i)
XConfigureRequestEvent* i;
{
          return(i->y);
}

void set_XConfigureRequestEvent_y(i, j)
XConfigureRequestEvent* i;
int j;
{
          i->y = j;
}

int  XConfigureRequestEvent_x(i)
XConfigureRequestEvent* i;
{
          return(i->x);
}

void set_XConfigureRequestEvent_x(i, j)
XConfigureRequestEvent* i;
int j;
{
          i->x = j;
}

int  XConfigureRequestEvent_window(i)
XConfigureRequestEvent* i;
{
          return(i->window);
}

void set_XConfigureRequestEvent_window(i, j)
XConfigureRequestEvent* i;
int j;
{
          i->window = j;
}

int  XConfigureRequestEvent_parent(i)
XConfigureRequestEvent* i;
{
          return(i->parent);
}

void set_XConfigureRequestEvent_parent(i, j)
XConfigureRequestEvent* i;
int j;
{
          i->parent = j;
}

long  XConfigureRequestEvent_display(i)
XConfigureRequestEvent* i;
{
          return((long) i->display);
}

void set_XConfigureRequestEvent_display(i, j)
XConfigureRequestEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XConfigureRequestEvent_send_event(i)
XConfigureRequestEvent* i;
{
          return(i->send_event);
}

void set_XConfigureRequestEvent_send_event(i, j)
XConfigureRequestEvent* i;
int j;
{
          i->send_event = j;
}

int  XConfigureRequestEvent_serial(i)
XConfigureRequestEvent* i;
{
          return(i->serial);
}

void set_XConfigureRequestEvent_serial(i, j)
XConfigureRequestEvent* i;
int j;
{
          i->serial = j;
}

int  XConfigureRequestEvent_type(i)
XConfigureRequestEvent* i;
{
          return(i->type);
}

void set_XConfigureRequestEvent_type(i, j)
XConfigureRequestEvent* i;
int j;
{
          i->type = j;
}


/********* XCirculateEvent functions *****/

long  make_XCirculateEvent (){
          return ((long) calloc(1, sizeof(XCirculateEvent)));
}

int  XCirculateEvent_place(i)
XCirculateEvent* i;
{
          return(i->place);
}

void set_XCirculateEvent_place(i, j)
XCirculateEvent* i;
int j;
{
          i->place = j;
}

int  XCirculateEvent_window(i)
XCirculateEvent* i;
{
          return(i->window);
}

void set_XCirculateEvent_window(i, j)
XCirculateEvent* i;
int j;
{
          i->window = j;
}

int  XCirculateEvent_event(i)
XCirculateEvent* i;
{
          return(i->event);
}

void set_XCirculateEvent_event(i, j)
XCirculateEvent* i;
int j;
{
          i->event = j;
}

long  XCirculateEvent_display(i)
XCirculateEvent* i;
{
          return((long) i->display);
}

void set_XCirculateEvent_display(i, j)
XCirculateEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XCirculateEvent_send_event(i)
XCirculateEvent* i;
{
          return(i->send_event);
}

void set_XCirculateEvent_send_event(i, j)
XCirculateEvent* i;
int j;
{
          i->send_event = j;
}

int  XCirculateEvent_serial(i)
XCirculateEvent* i;
{
          return(i->serial);
}

void set_XCirculateEvent_serial(i, j)
XCirculateEvent* i;
int j;
{
          i->serial = j;
}

int  XCirculateEvent_type(i)
XCirculateEvent* i;
{
          return(i->type);
}

void set_XCirculateEvent_type(i, j)
XCirculateEvent* i;
int j;
{
          i->type = j;
}


/********* XCirculateRequestEvent functions *****/

long  make_XCirculateRequestEvent (){
          return ((long) calloc(1, sizeof(XCirculateRequestEvent)));
}

int  XCirculateRequestEvent_place(i)
XCirculateRequestEvent* i;
{
          return(i->place);
}

void set_XCirculateRequestEvent_place(i, j)
XCirculateRequestEvent* i;
int j;
{
          i->place = j;
}

int  XCirculateRequestEvent_window(i)
XCirculateRequestEvent* i;
{
          return(i->window);
}

void set_XCirculateRequestEvent_window(i, j)
XCirculateRequestEvent* i;
int j;
{
          i->window = j;
}

int  XCirculateRequestEvent_parent(i)
XCirculateRequestEvent* i;
{
          return(i->parent);
}

void set_XCirculateRequestEvent_parent(i, j)
XCirculateRequestEvent* i;
int j;
{
          i->parent = j;
}

long  XCirculateRequestEvent_display(i)
XCirculateRequestEvent* i;
{
          return((long) i->display);
}

void set_XCirculateRequestEvent_display(i, j)
XCirculateRequestEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XCirculateRequestEvent_send_event(i)
XCirculateRequestEvent* i;
{
          return(i->send_event);
}

void set_XCirculateRequestEvent_send_event(i, j)
XCirculateRequestEvent* i;
int j;
{
          i->send_event = j;
}

int  XCirculateRequestEvent_serial(i)
XCirculateRequestEvent* i;
{
          return(i->serial);
}

void set_XCirculateRequestEvent_serial(i, j)
XCirculateRequestEvent* i;
int j;
{
          i->serial = j;
}

int  XCirculateRequestEvent_type(i)
XCirculateRequestEvent* i;
{
          return(i->type);
}

void set_XCirculateRequestEvent_type(i, j)
XCirculateRequestEvent* i;
int j;
{
          i->type = j;
}


/********* XPropertyEvent functions *****/

long  make_XPropertyEvent (){
          return ((long) calloc(1, sizeof(XPropertyEvent)));
}

int  XPropertyEvent_state(i)
XPropertyEvent* i;
{
          return(i->state);
}

void set_XPropertyEvent_state(i, j)
XPropertyEvent* i;
int j;
{
          i->state = j;
}

int  XPropertyEvent_time(i)
XPropertyEvent* i;
{
          return(i->time);
}

void set_XPropertyEvent_time(i, j)
XPropertyEvent* i;
int j;
{
          i->time = j;
}

int  XPropertyEvent_atom(i)
XPropertyEvent* i;
{
          return(i->atom);
}

void set_XPropertyEvent_atom(i, j)
XPropertyEvent* i;
int j;
{
          i->atom = j;
}

int  XPropertyEvent_window(i)
XPropertyEvent* i;
{
          return(i->window);
}

void set_XPropertyEvent_window(i, j)
XPropertyEvent* i;
int j;
{
          i->window = j;
}

long  XPropertyEvent_display(i)
XPropertyEvent* i;
{
          return((long) i->display);
}

void set_XPropertyEvent_display(i, j)
XPropertyEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XPropertyEvent_send_event(i)
XPropertyEvent* i;
{
          return(i->send_event);
}

void set_XPropertyEvent_send_event(i, j)
XPropertyEvent* i;
int j;
{
          i->send_event = j;
}

int  XPropertyEvent_serial(i)
XPropertyEvent* i;
{
          return(i->serial);
}

void set_XPropertyEvent_serial(i, j)
XPropertyEvent* i;
int j;
{
          i->serial = j;
}

int  XPropertyEvent_type(i)
XPropertyEvent* i;
{
          return(i->type);
}

void set_XPropertyEvent_type(i, j)
XPropertyEvent* i;
int j;
{
          i->type = j;
}


/********* XSelectionClearEvent functions *****/

long  make_XSelectionClearEvent (){
          return ((long) calloc(1, sizeof(XSelectionClearEvent)));
}

int  XSelectionClearEvent_time(i)
XSelectionClearEvent* i;
{
          return(i->time);
}

void set_XSelectionClearEvent_time(i, j)
XSelectionClearEvent* i;
int j;
{
          i->time = j;
}

int  XSelectionClearEvent_selection(i)
XSelectionClearEvent* i;
{
          return(i->selection);
}

void set_XSelectionClearEvent_selection(i, j)
XSelectionClearEvent* i;
int j;
{
          i->selection = j;
}

int  XSelectionClearEvent_window(i)
XSelectionClearEvent* i;
{
          return(i->window);
}

void set_XSelectionClearEvent_window(i, j)
XSelectionClearEvent* i;
int j;
{
          i->window = j;
}

long  XSelectionClearEvent_display(i)
XSelectionClearEvent* i;
{
          return((long) i->display);
}

void set_XSelectionClearEvent_display(i, j)
XSelectionClearEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XSelectionClearEvent_send_event(i)
XSelectionClearEvent* i;
{
          return(i->send_event);
}

void set_XSelectionClearEvent_send_event(i, j)
XSelectionClearEvent* i;
int j;
{
          i->send_event = j;
}

int  XSelectionClearEvent_serial(i)
XSelectionClearEvent* i;
{
          return(i->serial);
}

void set_XSelectionClearEvent_serial(i, j)
XSelectionClearEvent* i;
int j;
{
          i->serial = j;
}

int  XSelectionClearEvent_type(i)
XSelectionClearEvent* i;
{
          return(i->type);
}

void set_XSelectionClearEvent_type(i, j)
XSelectionClearEvent* i;
int j;
{
          i->type = j;
}


/********* XSelectionRequestEvent functions *****/

long  make_XSelectionRequestEvent (){
          return ((long) calloc(1, sizeof(XSelectionRequestEvent)));
}

int  XSelectionRequestEvent_time(i)
XSelectionRequestEvent* i;
{
          return(i->time);
}

void set_XSelectionRequestEvent_time(i, j)
XSelectionRequestEvent* i;
int j;
{
          i->time = j;
}

int  XSelectionRequestEvent_property(i)
XSelectionRequestEvent* i;
{
          return(i->property);
}

void set_XSelectionRequestEvent_property(i, j)
XSelectionRequestEvent* i;
int j;
{
          i->property = j;
}

int  XSelectionRequestEvent_target(i)
XSelectionRequestEvent* i;
{
          return(i->target);
}

void set_XSelectionRequestEvent_target(i, j)
XSelectionRequestEvent* i;
int j;
{
          i->target = j;
}

int  XSelectionRequestEvent_selection(i)
XSelectionRequestEvent* i;
{
          return(i->selection);
}

void set_XSelectionRequestEvent_selection(i, j)
XSelectionRequestEvent* i;
int j;
{
          i->selection = j;
}

int  XSelectionRequestEvent_requestor(i)
XSelectionRequestEvent* i;
{
          return(i->requestor);
}

void set_XSelectionRequestEvent_requestor(i, j)
XSelectionRequestEvent* i;
int j;
{
          i->requestor = j;
}

int  XSelectionRequestEvent_owner(i)
XSelectionRequestEvent* i;
{
          return(i->owner);
}

void set_XSelectionRequestEvent_owner(i, j)
XSelectionRequestEvent* i;
int j;
{
          i->owner = j;
}

long  XSelectionRequestEvent_display(i)
XSelectionRequestEvent* i;
{
          return((long) i->display);
}

void set_XSelectionRequestEvent_display(i, j)
XSelectionRequestEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XSelectionRequestEvent_send_event(i)
XSelectionRequestEvent* i;
{
          return(i->send_event);
}

void set_XSelectionRequestEvent_send_event(i, j)
XSelectionRequestEvent* i;
int j;
{
          i->send_event = j;
}

int  XSelectionRequestEvent_serial(i)
XSelectionRequestEvent* i;
{
          return(i->serial);
}

void set_XSelectionRequestEvent_serial(i, j)
XSelectionRequestEvent* i;
int j;
{
          i->serial = j;
}

int  XSelectionRequestEvent_type(i)
XSelectionRequestEvent* i;
{
          return(i->type);
}

void set_XSelectionRequestEvent_type(i, j)
XSelectionRequestEvent* i;
int j;
{
          i->type = j;
}


/********* XSelectionEvent functions *****/

long  make_XSelectionEvent (){
          return ((long) calloc(1, sizeof(XSelectionEvent)));
}

int  XSelectionEvent_time(i)
XSelectionEvent* i;
{
          return(i->time);
}

void set_XSelectionEvent_time(i, j)
XSelectionEvent* i;
int j;
{
          i->time = j;
}

int  XSelectionEvent_property(i)
XSelectionEvent* i;
{
          return(i->property);
}

void set_XSelectionEvent_property(i, j)
XSelectionEvent* i;
int j;
{
          i->property = j;
}

int  XSelectionEvent_target(i)
XSelectionEvent* i;
{
          return(i->target);
}

void set_XSelectionEvent_target(i, j)
XSelectionEvent* i;
int j;
{
          i->target = j;
}

int  XSelectionEvent_selection(i)
XSelectionEvent* i;
{
          return(i->selection);
}

void set_XSelectionEvent_selection(i, j)
XSelectionEvent* i;
int j;
{
          i->selection = j;
}

int  XSelectionEvent_requestor(i)
XSelectionEvent* i;
{
          return(i->requestor);
}

void set_XSelectionEvent_requestor(i, j)
XSelectionEvent* i;
int j;
{
          i->requestor = j;
}

long  XSelectionEvent_display(i)
XSelectionEvent* i;
{
          return((long) i->display);
}

void set_XSelectionEvent_display(i, j)
XSelectionEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XSelectionEvent_send_event(i)
XSelectionEvent* i;
{
          return(i->send_event);
}

void set_XSelectionEvent_send_event(i, j)
XSelectionEvent* i;
int j;
{
          i->send_event = j;
}

int  XSelectionEvent_serial(i)
XSelectionEvent* i;
{
          return(i->serial);
}

void set_XSelectionEvent_serial(i, j)
XSelectionEvent* i;
int j;
{
          i->serial = j;
}

int  XSelectionEvent_type(i)
XSelectionEvent* i;
{
          return(i->type);
}

void set_XSelectionEvent_type(i, j)
XSelectionEvent* i;
int j;
{
          i->type = j;
}


/********* XColormapEvent functions *****/

long  make_XColormapEvent (){
          return ((long) calloc(1, sizeof(XColormapEvent)));
}

int  XColormapEvent_state(i)
XColormapEvent* i;
{
          return(i->state);
}

void set_XColormapEvent_state(i, j)
XColormapEvent* i;
int j;
{
          i->state = j;
}

int  XColormapEvent_new(i)
XColormapEvent* i;
{
          return(i->new);
}

void set_XColormapEvent_new(i, j)
XColormapEvent* i;
int j;
{
          i->new = j;
}

int  XColormapEvent_colormap(i)
XColormapEvent* i;
{
          return(i->colormap);
}

void set_XColormapEvent_colormap(i, j)
XColormapEvent* i;
int j;
{
          i->colormap = j;
}

int  XColormapEvent_window(i)
XColormapEvent* i;
{
          return(i->window);
}

void set_XColormapEvent_window(i, j)
XColormapEvent* i;
int j;
{
          i->window = j;
}

long  XColormapEvent_display(i)
XColormapEvent* i;
{
          return((long) i->display);
}

void set_XColormapEvent_display(i, j)
XColormapEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XColormapEvent_send_event(i)
XColormapEvent* i;
{
          return(i->send_event);
}

void set_XColormapEvent_send_event(i, j)
XColormapEvent* i;
int j;
{
          i->send_event = j;
}

int  XColormapEvent_serial(i)
XColormapEvent* i;
{
          return(i->serial);
}

void set_XColormapEvent_serial(i, j)
XColormapEvent* i;
int j;
{
          i->serial = j;
}

int  XColormapEvent_type(i)
XColormapEvent* i;
{
          return(i->type);
}

void set_XColormapEvent_type(i, j)
XColormapEvent* i;
int j;
{
          i->type = j;
}


/********* XClientMessageEvent functions *****/

long  make_XClientMessageEvent (){
          return ((long) calloc(1, sizeof(XClientMessageEvent)));
}

int  XClientMessageEvent_format(i)
XClientMessageEvent* i;
{
          return(i->format);
}

void set_XClientMessageEvent_format(i, j)
XClientMessageEvent* i;
int j;
{
          i->format = j;
}

int  XClientMessageEvent_message_type(i)
XClientMessageEvent* i;
{
          return(i->message_type);
}

void set_XClientMessageEvent_message_type(i, j)
XClientMessageEvent* i;
int j;
{
          i->message_type = j;
}


int  XClientMessageEvent_window(i)
XClientMessageEvent* i;
{
          return(i->window);
}

void set_XClientMessageEvent_window(i, j)
XClientMessageEvent* i;
int j;
{
          i->window = j;
}

long  XClientMessageEvent_display(i)
XClientMessageEvent* i;
{
          return((long) i->display);
}

void set_XClientMessageEvent_display(i, j)
XClientMessageEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XClientMessageEvent_send_event(i)
XClientMessageEvent* i;
{
          return(i->send_event);
}

void set_XClientMessageEvent_send_event(i, j)
XClientMessageEvent* i;
int j;
{
          i->send_event = j;
}

int  XClientMessageEvent_serial(i)
XClientMessageEvent* i;
{
          return(i->serial);
}

void set_XClientMessageEvent_serial(i, j)
XClientMessageEvent* i;
int j;
{
          i->serial = j;
}

int  XClientMessageEvent_type(i)
XClientMessageEvent* i;
{
          return(i->type);
}

void set_XClientMessageEvent_type(i, j)
XClientMessageEvent* i;
int j;
{
          i->type = j;
}


/********* XMappingEvent functions *****/

long  make_XMappingEvent (){
          return ((long) calloc(1, sizeof(XMappingEvent)));
}

int  XMappingEvent_count(i)
XMappingEvent* i;
{
          return(i->count);
}

void set_XMappingEvent_count(i, j)
XMappingEvent* i;
int j;
{
          i->count = j;
}

int XMappingEvent_first_keycode(i)
XMappingEvent* i;
{
          return(i->first_keycode);
}

void set_XMappingEvent_first_keycode(i, j)
XMappingEvent* i;
int j;
{
          i->first_keycode = j;
}

int  XMappingEvent_request(i)
XMappingEvent* i;
{
          return(i->request);
}

void set_XMappingEvent_request(i, j)
XMappingEvent* i;
int j;
{
          i->request = j;
}

int  XMappingEvent_window(i)
XMappingEvent* i;
{
          return(i->window);
}

void set_XMappingEvent_window(i, j)
XMappingEvent* i;
int j;
{
          i->window = j;
}

long  XMappingEvent_display(i)
XMappingEvent* i;
{
          return((long) i->display);
}

void set_XMappingEvent_display(i, j)
XMappingEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XMappingEvent_send_event(i)
XMappingEvent* i;
{
          return(i->send_event);
}

void set_XMappingEvent_send_event(i, j)
XMappingEvent* i;
int j;
{
          i->send_event = j;
}

int  XMappingEvent_serial(i)
XMappingEvent* i;
{
          return(i->serial);
}

void set_XMappingEvent_serial(i, j)
XMappingEvent* i;
int j;
{
          i->serial = j;
}

int  XMappingEvent_type(i)
XMappingEvent* i;
{
          return(i->type);
}

void set_XMappingEvent_type(i, j)
XMappingEvent* i;
int j;
{
          i->type = j;
}


/********* XErrorEvent functions *****/

long  make_XErrorEvent (){
          return ((long) calloc(1, sizeof(XErrorEvent)));
}

char XErrorEvent_minor_code(i)
XErrorEvent* i;
{
          return(i->minor_code);
}

void set_XErrorEvent_minor_code(i, j)
XErrorEvent* i;
char j;
{
          i->minor_code = j;
}

char XErrorEvent_request_code(i)
XErrorEvent* i;
{
          return(i->request_code);
}

void set_XErrorEvent_request_code(i, j)
XErrorEvent* i;
char j;
{
          i->request_code = j;
}

char XErrorEvent_error_code(i)
XErrorEvent* i;
{
          return(i->error_code);
}

void set_XErrorEvent_error_code(i, j)
XErrorEvent* i;
char j;
{
          i->error_code = j;
}

int  XErrorEvent_serial(i)
XErrorEvent* i;
{
          return(i->serial);
}

void set_XErrorEvent_serial(i, j)
XErrorEvent* i;
int j;
{
          i->serial = j;
}

int  XErrorEvent_resourceid(i)
XErrorEvent* i;
{
          return(i->resourceid);
}

void set_XErrorEvent_resourceid(i, j)
XErrorEvent* i;
int j;
{
          i->resourceid = j;
}

long  XErrorEvent_display(i)
XErrorEvent* i;
{
          return((long) i->display);
}

void set_XErrorEvent_display(i, j)
XErrorEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XErrorEvent_type(i)
XErrorEvent* i;
{
          return(i->type);
}

void set_XErrorEvent_type(i, j)
XErrorEvent* i;
int j;
{
          i->type = j;
}


/********* XAnyEvent functions *****/

long  make_XAnyEvent (){
          return ((long) calloc(1, sizeof(XAnyEvent)));
}

int  XAnyEvent_window(i)
XAnyEvent* i;
{
          return(i->window);
}

void set_XAnyEvent_window(i, j)
XAnyEvent* i;
int j;
{
          i->window = j;
}

long  XAnyEvent_display(i)
XAnyEvent* i;
{
          return((long) i->display);
}

void set_XAnyEvent_display(i, j)
XAnyEvent* i;
long j;
{
          i->display = (Display *) j;
}

int  XAnyEvent_send_event(i)
XAnyEvent* i;
{
          return(i->send_event);
}

void set_XAnyEvent_send_event(i, j)
XAnyEvent* i;
int j;
{
          i->send_event = j;
}

int  XAnyEvent_serial(i)
XAnyEvent* i;
{
          return(i->serial);
}

void set_XAnyEvent_serial(i, j)
XAnyEvent* i;
int j;
{
          i->serial = j;
}

int  XAnyEvent_type(i)
XAnyEvent* i;
{
          return(i->type);
}

void set_XAnyEvent_type(i, j)
XAnyEvent* i;
int j;
{
          i->type = j;
}


/********* XEvent functions *****/

long  make_XEvent (){
          return ((long) calloc(1, sizeof(XEvent)));
}
