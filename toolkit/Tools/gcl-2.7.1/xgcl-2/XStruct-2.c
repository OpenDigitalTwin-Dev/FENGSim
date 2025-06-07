/* XStruct-2.c           Hiep Huu Nguyen         27 Jun 06 */

/* ; Copyright (c) 1994 Hiep Huu Nguyen and The University of Texas at Austin.
; Copyright (c) 2024 Camm Maguire
; edited 27 Aug 92; 12 Aug 02 by G. Novak; 24 Jun 06 by GSN
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


/********* _XQEvent functions *****/
#define NEED_EVENTS
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xlibint.h>


long  make__XQEvent (){
          return ((long) calloc(1, sizeof(_XQEvent)));
}

XEvent _XQEvent_event(i)
_XQEvent* i;
{
          return(i->event);
}

void set__XQEvent_event(i, j)
_XQEvent* i;
XEvent j;
{
          i->event = j;
}

long _XQEvent_next(i)
_XQEvent* i;
{
          return((long) i->next);
}

void set__XQEvent_next(i, j)
_XQEvent* i;
long j;
{
          i->next = (struct _XSQEvent *) j;
}


/********* XCharStruct functions *****/

long  make_XCharStruct (){
          return ((long) calloc(1, sizeof(XCharStruct)));
}

int  XCharStruct_attributes(i)
XCharStruct* i;
{
          return(i->attributes);
}

void set_XCharStruct_attributes(i, j)
XCharStruct* i;
int j;
{
          i->attributes = j;
}

int  XCharStruct_descent(i)
XCharStruct* i;
{
          return(i->descent);
}

void set_XCharStruct_descent(i, j)
XCharStruct* i;
int j;
{
          i->descent = j;
}

int  XCharStruct_ascent(i)
XCharStruct* i;
{
          return(i->ascent);
}

void set_XCharStruct_ascent(i, j)
XCharStruct* i;
int j;
{
          i->ascent = j;
}

int  XCharStruct_width(i)
XCharStruct* i;
{
          return(i->width);
}

void set_XCharStruct_width(i, j)
XCharStruct* i;
int j;
{
          i->width = j;
}

int  XCharStruct_rbearing(i)
XCharStruct* i;
{
          return(i->rbearing);
}

void set_XCharStruct_rbearing(i, j)
XCharStruct* i;
int j;
{
          i->rbearing = j;
}

int  XCharStruct_lbearing(i)
XCharStruct* i;
{
          return(i->lbearing);
}

void set_XCharStruct_lbearing(i, j)
XCharStruct* i;
int j;
{
          i->lbearing = j;
}


/********* XFontProp functions *****/

long  make_XFontProp (){
          return ((long) calloc(1, sizeof(XFontProp)));
}

int  XFontProp_card32(i)
XFontProp* i;
{
          return(i->card32);
}

void set_XFontProp_card32(i, j)
XFontProp* i;
int j;
{
          i->card32 = j;
}

int  XFontProp_name(i)
XFontProp* i;
{
          return(i->name);
}

void set_XFontProp_name(i, j)
XFontProp* i;
int j;
{
          i->name = j;
}


/********* XFontStruct functions *****/

long  make_XFontStruct (){
          return ((long) calloc(1, sizeof(XFontStruct)));
}

int  XFontStruct_descent(i)
XFontStruct* i;
{
          return(i->descent);
}

void set_XFontStruct_descent(i, j)
XFontStruct* i;
int j;
{
          i->descent = j;
}

int  XFontStruct_ascent(i)
XFontStruct* i;
{
          return(i->ascent);
}

void set_XFontStruct_ascent(i, j)
XFontStruct* i;
int j;
{
          i->ascent = j;
}

long  XFontStruct_per_char(i)
XFontStruct* i;
{
          return((long) i->per_char);
}

void set_XFontStruct_per_char(i, j)
XFontStruct* i;
long j;
{
          i->per_char = (XCharStruct *) j;
}

long XFontStruct_max_bounds(i)
XFontStruct* i;
{
          return((long) &i->max_bounds);
}
long XFontStruct_min_bounds(i)
XFontStruct* i;
{
          return((long) &i->min_bounds);
}
void set_XFontStruct_max_bounds(i, j)
XFontStruct* i;
XCharStruct j;
{
          i->max_bounds = j;
}
void set_XFontStruct_min_bounds(i, j)
XFontStruct* i;
XCharStruct j;
{
          i->min_bounds = j;
}

long  XFontStruct_properties(i)
XFontStruct* i;
{
          return((long) i->properties);
}

void set_XFontStruct_properties(i, j)
XFontStruct* i;
long j;
{
          i->properties = (XFontProp *) j;
}

int  XFontStruct_n_properties(i)
XFontStruct* i;
{
          return(i->n_properties);
}

void set_XFontStruct_n_properties(i, j)
XFontStruct* i;
int j;
{
          i->n_properties = j;
}

int  XFontStruct_default_char(i)
XFontStruct* i;
{
          return(i->default_char);
}

void set_XFontStruct_default_char(i, j)
XFontStruct* i;
int j;
{
          i->default_char = j;
}

int  XFontStruct_all_chars_exist(i)
XFontStruct* i;
{
          return(i->all_chars_exist);
}

void set_XFontStruct_all_chars_exist(i, j)
XFontStruct* i;
int j;
{
          i->all_chars_exist = j;
}

int  XFontStruct_max_byte1(i)
XFontStruct* i;
{
          return(i->max_byte1);
}

void set_XFontStruct_max_byte1(i, j)
XFontStruct* i;
int j;
{
          i->max_byte1 = j;
}

int  XFontStruct_min_byte1(i)
XFontStruct* i;
{
          return(i->min_byte1);
}

void set_XFontStruct_min_byte1(i, j)
XFontStruct* i;
int j;
{
          i->min_byte1 = j;
}

int  XFontStruct_max_char_or_byte2(i)
XFontStruct* i;
{
          return(i->max_char_or_byte2);
}

void set_XFontStruct_max_char_or_byte2(i, j)
XFontStruct* i;
int j;
{
          i->max_char_or_byte2 = j;
}

int  XFontStruct_min_char_or_byte2(i)
XFontStruct* i;
{
          return(i->min_char_or_byte2);
}

void set_XFontStruct_min_char_or_byte2(i, j)
XFontStruct* i;
int j;
{
          i->min_char_or_byte2 = j;
}

int  XFontStruct_direction(i)
XFontStruct* i;
{
          return(i->direction);
}

void set_XFontStruct_direction(i, j)
XFontStruct* i;
int j;
{
          i->direction = j;
}

int  XFontStruct_fid(i)
XFontStruct* i;
{
          return(i->fid);
}

void set_XFontStruct_fid(i, j)
XFontStruct* i;
int j;
{
          i->fid = j;
}

long  XFontStruct_ext_data(i)
XFontStruct* i;
{
          return((long) i->ext_data);
}

void set_XFontStruct_ext_data(i, j)
XFontStruct* i;
long j;
{
          i->ext_data = (XExtData *) j;
}


/********* XTextItem functions *****/

long  make_XTextItem (){
          return ((long) calloc(1, sizeof(XTextItem)));
}

int  XTextItem_font(i)
XTextItem* i;
{
          return(i->font);
}

void set_XTextItem_font(i, j)
XTextItem* i;
int j;
{
          i->font = j;
}

int  XTextItem_delta(i)
XTextItem* i;
{
          return(i->delta);
}

void set_XTextItem_delta(i, j)
XTextItem* i;
int j;
{
          i->delta = j;
}

int  XTextItem_nchars(i)
XTextItem* i;
{
          return(i->nchars);
}

void set_XTextItem_nchars(i, j)
XTextItem* i;
int j;
{
          i->nchars = j;
}

long  XTextItem_chars(i)
XTextItem* i;
{
          return((long) i->chars);
}

void set_XTextItem_chars(i, j)
XTextItem* i;
long j;
{
          i->chars = (char *) j;
}


/********* XChar2b functions *****/

long  make_XChar2b (){
          return ((long) calloc(1, sizeof(XChar2b)));
}

char XChar2b_byte2(i)
XChar2b* i;
{
          return(i->byte2);
}

void set_XChar2b_byte2(i, j)
XChar2b* i;
char j;
{
          i->byte2 = j;
}

char XChar2b_byte1(i)
XChar2b* i;
{
          return(i->byte1);
}

void set_XChar2b_byte1(i, j)
XChar2b* i;
char j;
{
          i->byte1 = j;
}


/********* XTextItem16 functions *****/

long  make_XTextItem16 (){
          return ((long) calloc(1, sizeof(XTextItem16)));
}

int  XTextItem16_font(i)
XTextItem16* i;
{
          return(i->font);
}

void set_XTextItem16_font(i, j)
XTextItem16* i;
int j;
{
          i->font = j;
}

int  XTextItem16_delta(i)
XTextItem16* i;
{
          return(i->delta);
}

void set_XTextItem16_delta(i, j)
XTextItem16* i;
int j;
{
          i->delta = j;
}

int  XTextItem16_nchars(i)
XTextItem16* i;
{
          return(i->nchars);
}

void set_XTextItem16_nchars(i, j)
XTextItem16* i;
int j;
{
          i->nchars = j;
}

long  XTextItem16_chars(i)
XTextItem16* i;
{
          return((long) i->chars);
}

void set_XTextItem16_chars(i, j)
XTextItem16* i;
long j;
{
          i->chars = (XChar2b *) j;
}


/********* XEDataObject functions *****/

long  make_XEDataObject (){
          return ((long) calloc(1, sizeof(XEDataObject)));
}

long  XEDataObject_font(i)
XEDataObject* i;
{
          return((long) i->font);
}

void set_XEDataObject_font(i, j)
XEDataObject* i;
long j;
{
          i->font = (XFontStruct *) j;
}

long  XEDataObject_pixmap_format(i)
XEDataObject* i;
{
          return((long) i->pixmap_format);
}

void set_XEDataObject_pixmap_format(i, j)
XEDataObject* i;
long j;
{
          i->pixmap_format = (ScreenFormat *) j;
}

long  XEDataObject_screen(i)
XEDataObject* i;
{
          return((long) i->screen);
}

void set_XEDataObject_screen(i, j)
XEDataObject* i;
long j;
{
          i->screen = (Screen *) j;
}

long  XEDataObject_visual(i)
XEDataObject* i;
{
          return((long) i->visual);
}

void set_XEDataObject_visual(i, j)
XEDataObject* i;
long j;
{
          i->visual = (Visual *) j;
}

GC   XEDataObject_gc(i)
XEDataObject* i;
{
          return(i->gc);
}

void set_XEDataObject_gc(i, j)
XEDataObject* i;
GC j;
{
          i->gc = j;
}


/********* XSizeHints functions *****/

long  make_XSizeHints (){
          return ((long) calloc(1, sizeof(XSizeHints)));
}

int  XSizeHints_win_gravity(i)
XSizeHints *i;
{
          return(i->win_gravity);
}

void set_XSizeHints_win_gravity(i, j)
XSizeHints *i;
int j;
{
          i->win_gravity = j;
}

int  XSizeHints_base_height(i)
XSizeHints* i;
{
          return(i->base_height);
}

void set_XSizeHints_base_height(i, j)
XSizeHints* i;
int j;
{
          i->base_height = j;
}

int  XSizeHints_base_width(i)
XSizeHints* i;
{
          return(i->base_width);
}

void set_XSizeHints_base_width(i, j)
XSizeHints* i;
int j;
{
          i->base_width = j;
}

int  XSizeHints_height_inc(i)
XSizeHints* i;
{
          return(i->height_inc);
}

void set_XSizeHints_height_inc(i, j)
XSizeHints* i;
int j;
{
          i->height_inc = j;
}

int  XSizeHints_width_inc(i)
XSizeHints* i;
{
          return(i->width_inc);
}

void set_XSizeHints_width_inc(i, j)
XSizeHints* i;
int j;
{
          i->width_inc = j;
}

int  XSizeHints_max_height(i)
XSizeHints* i;
{
          return(i->max_height);
}

void set_XSizeHints_max_height(i, j)
XSizeHints* i;
int j;
{
          i->max_height = j;
}

int  XSizeHints_max_width(i)
XSizeHints* i;
{
          return(i->max_width);
}

void set_XSizeHints_max_width(i, j)
XSizeHints* i;
int j;
{
          i->max_width = j;
}

int  XSizeHints_min_height(i)
XSizeHints* i;
{
          return(i->min_height);
}

void set_XSizeHints_min_height(i, j)
XSizeHints* i;
int j;
{
          i->min_height = j;
}

int  XSizeHints_min_width(i)
XSizeHints* i;
{
          return(i->min_width);
}

void set_XSizeHints_min_width(i, j)
XSizeHints* i;
int j;
{
          i->min_width = j;
}

int  XSizeHints_height(i)
XSizeHints* i;
{
          return(i->height);
}

void set_XSizeHints_height(i, j)
XSizeHints* i;
int j;
{
          i->height = j;
}

int  XSizeHints_width(i)
XSizeHints* i;
{
          return(i->width);
}

void set_XSizeHints_width(i, j)
XSizeHints* i;
int j;
{
          i->width = j;
}

int  XSizeHints_y(i)
XSizeHints* i;
{
          return(i->y);
}

void set_XSizeHints_y(i, j)
XSizeHints* i;
int j;
{
          i->y = j;
}

int  XSizeHints_x(i)
XSizeHints* i;
{
          return(i->x);
}

void set_XSizeHints_x(i, j)
XSizeHints* i;
int j;
{
          i->x = j;
}

int  XSizeHints_flags(i)
XSizeHints* i;
{
          return(i->flags);
}

void set_XSizeHints_flags(i, j)
XSizeHints* i;
int j;
{
          i->flags = j;
}


int  XSizeHints_max_aspect_x(i)
XSizeHints* i;
{
          return(i->max_aspect.x);
}

void  set_XSizeHints_max_aspect_x(i, j)
XSizeHints* i;
int	j;
{
          i->max_aspect.x = j;
}

int  XSizeHints_max_aspect_y(i)
XSizeHints* i;
{
          return(i->max_aspect.y);
}

void  set_XSizeHints_max_aspect_y(i, j)
XSizeHints* i;
int	j;
{
          i->max_aspect.y = j;
}

int  XSizeHints_min_aspect_x(i)
XSizeHints* i;
{
          return(i->min_aspect.x);
}

void  set_XSizeHints_min_aspect_x(i, j)
XSizeHints* i;
int	j;
{
          i->min_aspect.x = j;
}


int  XSizeHints_min_aspect_y(i)
XSizeHints* i;
{
          return(i->min_aspect.y);
}

void  set_XSizeHints_min_aspect_y(i, j)
XSizeHints* i;
int	j;
{
          i->min_aspect.y = j;
}


/********* XWMHints functions *****/

long  make_XWMHints (){
          return ((long) calloc(1, sizeof(XWMHints)));
}

int  XWMHints_window_group(i)
XWMHints* i;
{
          return(i->window_group);
}

void set_XWMHints_window_group(i, j)
XWMHints* i;
int j;
{
          i->window_group = j;
}

int  XWMHints_icon_mask(i)
XWMHints* i;
{
          return(i->icon_mask);
}

void set_XWMHints_icon_mask(i, j)
XWMHints* i;
int j;
{
          i->icon_mask = j;
}

int  XWMHints_icon_y(i)
XWMHints* i;
{
          return(i->icon_y);
}

void set_XWMHints_icon_y(i, j)
XWMHints* i;
int j;
{
          i->icon_y = j;
}

int  XWMHints_icon_x(i)
XWMHints* i;
{
          return(i->icon_x);
}

void set_XWMHints_icon_x(i, j)
XWMHints* i;
int j;
{
          i->icon_x = j;
}

int  XWMHints_icon_window(i)
XWMHints* i;
{
          return(i->icon_window);
}

void set_XWMHints_icon_window(i, j)
XWMHints* i;
int j;
{
          i->icon_window = j;
}

int  XWMHints_icon_pixmap(i)
XWMHints* i;
{
          return(i->icon_pixmap);
}

void set_XWMHints_icon_pixmap(i, j)
XWMHints* i;
int j;
{
          i->icon_pixmap = j;
}

int  XWMHints_initial_state(i)
XWMHints* i;
{
          return(i->initial_state);
}

void set_XWMHints_initial_state(i, j)
XWMHints* i;
int j;
{
          i->initial_state = j;
}

int  XWMHints_input(i)
XWMHints* i;
{
          return(i->input);
}

void set_XWMHints_input(i, j)
XWMHints* i;
int j;
{
          i->input = j;
}

int  XWMHints_flags(i)
XWMHints* i;
{
          return(i->flags);
}

void set_XWMHints_flags(i, j)
XWMHints* i;
int j;
{
          i->flags = j;
}


/********* XTextProperty functions *****/

long  make_XTextProperty (){
          return ((long) calloc(1, sizeof(XTextProperty)));
}

int  XTextProperty_nitems(i)
XTextProperty *i;
{
          return(i->nitems);
}

void set_XTextProperty_nitems(i, j)
XTextProperty* i;
int j;
{
          i->nitems = j;
}

int  XTextProperty_format(i)
XTextProperty* i;
{
          return(i->format);
}

void set_XTextProperty_format(i, j)
XTextProperty* i;
int j;
{
          i->format = j;
}

int  XTextProperty_encoding(i)
XTextProperty* i;
{
          return(i->encoding);
}

void set_XTextProperty_encoding(i, j)
XTextProperty* i;
int j;
{
          i->encoding = j;
}

long  XTextProperty_value(i)
XTextProperty* i;
{
          return((long) i->value);
}

void set_XTextProperty_value(i, j)
XTextProperty* i;
long j;
{
          i->value = (unsigned char *) j;
}


/********* XIconSize functions *****/

long  make_XIconSize (){
          return ((long) calloc(1, sizeof(XIconSize)));
}

int  XIconSize_height_inc(i)
XIconSize* i;
{
          return(i->height_inc);
}

void set_XIconSize_height_inc(i, j)
XIconSize* i;
int j;
{
          i->height_inc = j;
}

int  XIconSize_width_inc(i)
XIconSize* i;
{
          return(i->width_inc);
}

void set_XIconSize_width_inc(i, j)
XIconSize* i;
int j;
{
          i->width_inc = j;
}

int  XIconSize_max_height(i)
XIconSize* i;
{
          return(i->max_height);
}

void set_XIconSize_max_height(i, j)
XIconSize* i;
int j;
{
          i->max_height = j;
}

int  XIconSize_max_width(i)
XIconSize* i;
{
          return(i->max_width);
}

void set_XIconSize_max_width(i, j)
XIconSize* i;
int j;
{
          i->max_width = j;
}

int  XIconSize_min_height(i)
XIconSize* i;
{
          return(i->min_height);
}

void set_XIconSize_min_height(i, j)
XIconSize* i;
int j;
{
          i->min_height = j;
}

int  XIconSize_min_width(i)
XIconSize* i;
{
          return(i->min_width);
}

void set_XIconSize_min_width(i, j)
XIconSize* i;
int j;
{
          i->min_width = j;
}


/********* XClassHint functions *****/

long  make_XClassHint (){
          return ((long) calloc(1, sizeof(XClassHint)));
}

long  XClassHint_res_class(i)
XClassHint* i;
{
          return((long) i->res_class);
}

void set_XClassHint_res_class(i, j)
XClassHint* i;
long j;
{
          i->res_class = (char *) j;
}

long  XClassHint_res_name(i)
XClassHint* i;
{
          return((long) i->res_name);
}

void set_XClassHint_res_name(i, j)
XClassHint* i;
long j;
{
          i->res_name = (char *) j;
}


/********* XComposeStatus functions *****/

long  make_XComposeStatus (){
          return ((long) calloc(1, sizeof(XComposeStatus)));
}

int  XComposeStatus_chars_matched(i)
XComposeStatus* i;
{
          return(i->chars_matched);
}

void set_XComposeStatus_chars_matched(i, j)
XComposeStatus* i;
int j;
{
          i->chars_matched = j;
}

long  XComposeStatus_compose_ptr(i)
XComposeStatus* i;
{
          return((long) i->compose_ptr);
}

void set_XComposeStatus_compose_ptr(i, j)
XComposeStatus* i;
long j;
{
          i->compose_ptr = (XPointer) j;
}


/********* XVisualInfo functions *****/

long  make_XVisualInfo (){
          return ((long) calloc(1, sizeof(XVisualInfo)));
}

int  XVisualInfo_bits_per_rgb(i)
XVisualInfo* i;
{
          return(i->bits_per_rgb);
}

void set_XVisualInfo_bits_per_rgb(i, j)
XVisualInfo* i;
int j;
{
          i->bits_per_rgb = j;
}

int  XVisualInfo_colormap_size(i)
XVisualInfo* i;
{
          return(i->colormap_size);
}

void set_XVisualInfo_colormap_size(i, j)
XVisualInfo* i;
int j;
{
          i->colormap_size = j;
}

int  XVisualInfo_blue_mask(i)
XVisualInfo* i;
{
          return(i->blue_mask);
}

void set_XVisualInfo_blue_mask(i, j)
XVisualInfo* i;
int j;
{
          i->blue_mask = j;
}

int  XVisualInfo_green_mask(i)
XVisualInfo* i;
{
          return(i->green_mask);
}

void set_XVisualInfo_green_mask(i, j)
XVisualInfo* i;
int j;
{
          i->green_mask = j;
}

int  XVisualInfo_red_mask(i)
XVisualInfo* i;
{
          return(i->red_mask);
}

void set_XVisualInfo_red_mask(i, j)
XVisualInfo* i;
int j;
{
          i->red_mask = j;
}

int  XVisualInfo_class(i)
XVisualInfo* i;
{
          return(i->class);
}

void set_XVisualInfo_class(i, j)
XVisualInfo* i;
int j;
{
          i->class = j;
}

int  XVisualInfo_depth(i)
XVisualInfo* i;
{
          return(i->depth);
}

void set_XVisualInfo_depth(i, j)
XVisualInfo* i;
int j;
{
          i->depth = j;
}

int  XVisualInfo_screen(i)
XVisualInfo* i;
{
          return(i->screen);
}

void set_XVisualInfo_screen(i, j)
XVisualInfo* i;
int j;
{
          i->screen = j;
}

int  XVisualInfo_visualid(i)
XVisualInfo* i;
{
          return(i->visualid);
}

void set_XVisualInfo_visualid(i, j)
XVisualInfo* i;
int j;
{
          i->visualid = j;
}

long  XVisualInfo_visual(i)
XVisualInfo* i;
{
          return((long) i->visual);
}

void set_XVisualInfo_visual(i, j)
XVisualInfo* i;
long j;
{
          i->visual = (Visual *) j;
}


/********* XStandardColormap functions *****/

long  make_XStandardColormap (){
          return ((long) calloc(1, sizeof(XStandardColormap)));
}

int  XStandardColormap_killid(i)
XStandardColormap* i;
{
          return(i->killid);
}

void set_XStandardColormap_killid(i, j)
XStandardColormap* i;
int j;
{
          i->killid = j;
}

int  XStandardColormap_visualid(i)
XStandardColormap* i;
{
          return(i->visualid);
}

void set_XStandardColormap_visualid(i, j)
XStandardColormap* i;
int j;
{
          i->visualid = j;
}

int  XStandardColormap_base_pixel(i)
XStandardColormap* i;
{
          return(i->base_pixel);
}

void set_XStandardColormap_base_pixel(i, j)
XStandardColormap* i;
int j;
{
          i->base_pixel = j;
}

int  XStandardColormap_blue_mult(i)
XStandardColormap* i;
{
          return(i->blue_mult);
}

void set_XStandardColormap_blue_mult(i, j)
XStandardColormap* i;
int j;
{
          i->blue_mult = j;
}

int  XStandardColormap_blue_max(i)
XStandardColormap* i;
{
          return(i->blue_max);
}

void set_XStandardColormap_blue_max(i, j)
XStandardColormap* i;
int j;
{
          i->blue_max = j;
}

int  XStandardColormap_green_mult(i)
XStandardColormap* i;
{
          return(i->green_mult);
}

void set_XStandardColormap_green_mult(i, j)
XStandardColormap* i;
int j;
{
          i->green_mult = j;
}

int  XStandardColormap_green_max(i)
XStandardColormap* i;
{
          return(i->green_max);
}

void set_XStandardColormap_green_max(i, j)
XStandardColormap* i;
int j;
{
          i->green_max = j;
}

int  XStandardColormap_red_mult(i)
XStandardColormap* i;
{
          return(i->red_mult);
}

void set_XStandardColormap_red_mult(i, j)
XStandardColormap* i;
int j;
{
          i->red_mult = j;
}

int  XStandardColormap_red_max(i)
XStandardColormap* i;
{
          return(i->red_max);
}

void set_XStandardColormap_red_max(i, j)
XStandardColormap* i;
int j;
{
          i->red_max = j;
}

int  XStandardColormap_colormap(i)
XStandardColormap* i;
{
          return(i->colormap);
}

void set_XStandardColormap_colormap(i, j)
XStandardColormap* i;
int j;
{
          i->colormap = j;
}

