/* XStruct-4.c           Hiep Huu Nguyen       27 Jun 06 */

/* ; Copyright (c) 1994 Hiep Huu Nguyen and The University of Texas at Austin.
; Copyright (c) 2024 Camm Maguire
; edited 27 Aug 92; 12 Aug 2002 by G. Novak; 24 Jun 06 by GSN
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
#include <string.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>


/********* XExtCodes functions *****/

long  make_XExtCodes (){
          return ((long) calloc(1, sizeof(XExtCodes)));
}

int  XExtCodes_first_error(i)
XExtCodes* i;
{
          return(i->first_error);
}

void set_XExtCodes_first_error(i, j)
XExtCodes* i;
int j;
{
          i->first_error = j;
}

int  XExtCodes_first_event(i)
XExtCodes* i;
{
          return(i->first_event);
}

void set_XExtCodes_first_event(i, j)
XExtCodes* i;
int j;
{
          i->first_event = j;
}

int  XExtCodes_major_opcode(i)
XExtCodes* i;
{
          return(i->major_opcode);
}

void set_XExtCodes_major_opcode(i, j)
XExtCodes* i;
int j;
{
          i->major_opcode = j;
}

int  XExtCodes_extension(i)
XExtCodes* i;
{
          return(i->extension);
}

void set_XExtCodes_extension(i, j)
XExtCodes* i;
int j;
{
          i->extension = j;
}


/********* XPixmapFormatValues functions *****/

long  make_XPixmapFormatValues (){
          return ((long) calloc(1, sizeof(XPixmapFormatValues)));
}

int  XPixmapFormatValues_scanline_pad(i)
XPixmapFormatValues* i;
{
          return(i->scanline_pad);
}

void set_XPixmapFormatValues_scanline_pad(i, j)
XPixmapFormatValues* i;
int j;
{
          i->scanline_pad = j;
}

int  XPixmapFormatValues_bits_per_pixel(i)
XPixmapFormatValues* i;
{
          return(i->bits_per_pixel);
}

void set_XPixmapFormatValues_bits_per_pixel(i, j)
XPixmapFormatValues* i;
int j;
{
          i->bits_per_pixel = j;
}

int  XPixmapFormatValues_depth(i)
XPixmapFormatValues* i;
{
          return(i->depth);
}

void set_XPixmapFormatValues_depth(i, j)
XPixmapFormatValues* i;
int j;
{
          i->depth = j;
}


/********* XGCValues functions *****/

long  make_XGCValues (){
          return ((long) calloc(1, sizeof(XGCValues)));
}

char XGCValues_dashes(i)
XGCValues* i;
{
          return(i->dashes);
}

void set_XGCValues_dashes(i, j)
XGCValues* i;
char j;
{
          i->dashes = j;
}

int  XGCValues_dash_offset(i)
XGCValues* i;
{
          return(i->dash_offset);
}

void set_XGCValues_dash_offset(i, j)
XGCValues* i;
int j;
{
          i->dash_offset = j;
}

int  XGCValues_clip_mask(i)
XGCValues* i;
{
          return(i->clip_mask);
}

void set_XGCValues_clip_mask(i, j)
XGCValues* i;
int j;
{
          i->clip_mask = j;
}

int  XGCValues_clip_y_origin(i)
XGCValues* i;
{
          return(i->clip_y_origin);
}

void set_XGCValues_clip_y_origin(i, j)
XGCValues* i;
int j;
{
          i->clip_y_origin = j;
}

int  XGCValues_clip_x_origin(i)
XGCValues* i;
{
          return(i->clip_x_origin);
}

void set_XGCValues_clip_x_origin(i, j)
XGCValues* i;
int j;
{
          i->clip_x_origin = j;
}

int  XGCValues_graphics_exposures(i)
XGCValues* i;
{
          return(i->graphics_exposures);
}

void set_XGCValues_graphics_exposures(i, j)
XGCValues* i;
int j;
{
          i->graphics_exposures = j;
}

int  XGCValues_subwindow_mode(i)
XGCValues* i;
{
          return(i->subwindow_mode);
}

void set_XGCValues_subwindow_mode(i, j)
XGCValues* i;
int j;
{
          i->subwindow_mode = j;
}

int  XGCValues_font(i)
XGCValues* i;
{
          return(i->font);
}

void set_XGCValues_font(i, j)
XGCValues* i;
int j;
{
          i->font = j;
}

int  XGCValues_ts_y_origin(i)
XGCValues* i;
{
          return(i->ts_y_origin);
}

void set_XGCValues_ts_y_origin(i, j)
XGCValues* i;
int j;
{
          i->ts_y_origin = j;
}

int  XGCValues_ts_x_origin(i)
XGCValues* i;
{
          return(i->ts_x_origin);
}

void set_XGCValues_ts_x_origin(i, j)
XGCValues* i;
int j;
{
          i->ts_x_origin = j;
}

int  XGCValues_stipple(i)
XGCValues* i;
{
          return(i->stipple);
}

void set_XGCValues_stipple(i, j)
XGCValues* i;
int j;
{
          i->stipple = j;
}

int  XGCValues_tile(i)
XGCValues* i;
{
          return(i->tile);
}

void set_XGCValues_tile(i, j)
XGCValues* i;
int j;
{
          i->tile = j;
}

int  XGCValues_arc_mode(i)
XGCValues* i;
{
          return(i->arc_mode);
}

void set_XGCValues_arc_mode(i, j)
XGCValues* i;
int j;
{
          i->arc_mode = j;
}

int  XGCValues_fill_rule(i)
XGCValues* i;
{
          return(i->fill_rule);
}

void set_XGCValues_fill_rule(i, j)
XGCValues* i;
int j;
{
          i->fill_rule = j;
}

int  XGCValues_fill_style(i)
XGCValues* i;
{
          return(i->fill_style);
}

void set_XGCValues_fill_style(i, j)
XGCValues* i;
int j;
{
          i->fill_style = j;
}

int  XGCValues_join_style(i)
XGCValues* i;
{
          return(i->join_style);
}

void set_XGCValues_join_style(i, j)
XGCValues* i;
int j;
{
          i->join_style = j;
}

int  XGCValues_cap_style(i)
XGCValues* i;
{
          return(i->cap_style);
}

void set_XGCValues_cap_style(i, j)
XGCValues* i;
int j;
{
          i->cap_style = j;
}

int  XGCValues_line_style(i)
XGCValues* i;
{
          return(i->line_style);
}

void set_XGCValues_line_style(i, j)
XGCValues* i;
int j;
{
          i->line_style = j;
}

int  XGCValues_line_width(i)
XGCValues* i;
{
          return(i->line_width);
}

void set_XGCValues_line_width(i, j)
XGCValues* i;
int j;
{
          i->line_width = j;
}

int  XGCValues_background(i)
XGCValues* i;
{
          return(i->background);
}

void set_XGCValues_background(i, j)
XGCValues* i;
int j;
{
          i->background = j;
}

int  XGCValues_foreground(i)
XGCValues* i;
{
          return(i->foreground);
}

void set_XGCValues_foreground(i, j)
XGCValues* i;
int j;
{
          i->foreground = j;
}

int  XGCValues_plane_mask(i)
XGCValues* i;
{
          return(i->plane_mask);
}

void set_XGCValues_plane_mask(i, j)
XGCValues* i;
int j;
{
          i->plane_mask = j;
}

int  XGCValues_function(i)
XGCValues* i;
{
          return(i->function);
}

void set_XGCValues_function(i, j)
XGCValues* i;
int j;
{
          i->function = j;
}


/********* GC functions *****

int  make_GC (){
          GC i;
          return ((int) &i);
}

int  GC_values(i)
GC i;
{
          return(i->values);
}

void set_GC_values(i, j)
GC i;
int j;
{
          i->values = j;
}

int  GC_dirty(i)
GC i;
{
          return(i->dirty);
}

void set_GC_dirty(i, j)
GC i;
int j;
{
          i->dirty = j;
}

int  GC_dashes(i)
GC i;
{
          return(i->dashes);
}

void set_GC_dashes(i, j)
GC i;
int j;
{
          i->dashes = j;
}

int  GC_rects(i)
GC i;
{
          return(i->rects);
}

void set_GC_rects(i, j)
GC i;
int j;
{
          i->rects = j;
}

int  GC_gid(i)
GC i;
{
          return(i->gid);
}

void set_GC_gid(i, j)
GC i;
int j;
{
          i->gid = j;
}

int  GC_ext_data(i)
GC i;
{
          return(i->ext_data);
}

void set_GC_ext_data(i, j)
GC i;
int j;
{
          i->ext_data = j;
}

*/

/********* Visual functions *****/

long  make_Visual (){
          return ((long) calloc(1, sizeof(Visual)));
}

int  Visual_map_entries(i)
Visual* i;
{
          return(i->map_entries);
}

void set_Visual_map_entries(i, j)
Visual* i;
int j;
{
          i->map_entries = j;
}

int  Visual_bits_per_rgb(i)
Visual* i;
{
          return(i->bits_per_rgb);
}

void set_Visual_bits_per_rgb(i, j)
Visual* i;
int j;
{
          i->bits_per_rgb = j;
}

int   Visual_blue_mask(i)
Visual* i;
{
          return(i->blue_mask);
}

void set_Visual_blue_mask(i, j)
Visual* i;
int j;
{
          i->blue_mask = j;
}

int  Visual_green_mask(i)
Visual* i;
{
          return(i->green_mask);
}

void set_Visual_green_mask(i, j)
Visual* i;
int j;
{
          i->green_mask = j;
}

int  Visual_red_mask(i)
Visual* i;
{
          return(i->red_mask);
}

void set_Visual_red_mask(i, j)
Visual* i;
int j;
{
          i->red_mask = j;
}

int  Visual_class(i)
Visual* i;
{
          return(i->class);
}

void set_Visual_class(i, j)
Visual* i;
int j;
{
          i->class = j;
}

int  Visual_visualid(i)
Visual* i;
{
          return(i->visualid);
}

void set_Visual_visualid(i, j)
Visual* i;
int j;
{
          i->visualid = j;
}

long  Visual_ext_data(i)
Visual* i;
{
          return((long) i->ext_data);
}

void set_Visual_ext_data(i, j)
Visual* i;
long j;
{
          i->ext_data = (XExtData *) j;
}


/********* Depth functions *****/

long  make_Depth (){
          return ((long) calloc(1, sizeof(Depth)));
}

long  Depth_visuals(i)
Depth* i;
{
          return((long) i->visuals);
}

void set_Depth_visuals(i, j)
Depth* i;
long j;
{
          i->visuals = (Visual *) j;
}

int  Depth_nvisuals(i)
Depth* i;
{
          return(i->nvisuals);
}

void set_Depth_nvisuals(i, j)
Depth* i;
int j;
{
          i->nvisuals = j;
}

int  Depth_depth(i)
Depth* i;
{
          return(i->depth);
}

void set_Depth_depth(i, j)
Depth* i;
int j;
{
          i->depth = j;
}


/********* Screen functions *****/

long  make_Screen (){
          return ((long) calloc(1, sizeof(Screen)));
}

int  Screen_root_input_mask(i)
Screen* i;
{
          return(i->root_input_mask);
}

void set_Screen_root_input_mask(i, j)
Screen* i;
int j;
{
          i->root_input_mask = j;
}

int  Screen_save_unders(i)
Screen* i;
{
          return(i->save_unders);
}

void set_Screen_save_unders(i, j)
Screen* i;
int j;
{
          i->save_unders = j;
}

int  Screen_backing_store(i)
Screen* i;
{
          return(i->backing_store);
}

void set_Screen_backing_store(i, j)
Screen* i;
int j;
{
          i->backing_store = j;
}

int  Screen_min_maps(i)
Screen* i;
{
          return(i->min_maps);
}

void set_Screen_min_maps(i, j)
Screen* i;
int j;
{
          i->min_maps = j;
}

int  Screen_max_maps(i)
Screen* i;
{
          return(i->max_maps);
}

void set_Screen_max_maps(i, j)
Screen* i;
int j;
{
          i->max_maps = j;
}

int  Screen_black_pixel(i)
Screen* i;
{
          return(i->black_pixel);
}

void set_Screen_black_pixel(i, j)
Screen* i;
int j;
{
          i->black_pixel = j;
}

int  Screen_white_pixel(i)
Screen* i;
{
          return(i->white_pixel);
}

void set_Screen_white_pixel(i, j)
Screen* i;
int j;
{
          i->white_pixel = j;
}

int  Screen_cmap(i)
Screen* i;
{
          return(i->cmap);
}

void set_Screen_cmap(i, j)
Screen* i;
int j;
{
          i->cmap = j;
}

long  Screen_default_gc(i)
Screen* i;
{
          return((long) i->default_gc);
}

void set_Screen_default_gc(i, j)
Screen* i;
long j;
{
          i->default_gc = (GC) j;
}

long  Screen_root_visual(i)
Screen* i;
{
          return((long) i->root_visual);
}

void set_Screen_root_visual(i, j)
Screen* i;
long j;
{
          i->root_visual = (Visual *) j;
}

int  Screen_root_depth(i)
Screen* i;
{
          return(i->root_depth);
}

void set_Screen_root_depth(i, j)
Screen* i;
int j;
{
          i->root_depth = j;
}

long  Screen_depths(i)
Screen* i;
{
          return((long) i->depths);
}

void set_Screen_depths(i, j)
Screen* i;
long j;
{
          i->depths = (Depth *) j;
}

int  Screen_ndepths(i)
Screen* i;
{
          return(i->ndepths);
}

void set_Screen_ndepths(i, j)
Screen* i;
int j;
{
          i->ndepths = j;
}

int  Screen_mheight(i)
Screen* i;
{
          return(i->mheight);
}

void set_Screen_mheight(i, j)
Screen* i;
int j;
{
          i->mheight = j;
}

int  Screen_mwidth(i)
Screen* i;
{
          return(i->mwidth);
}

void set_Screen_mwidth(i, j)
Screen* i;
int j;
{
          i->mwidth = j;
}

int  Screen_height(i)
Screen* i;
{
          return(i->height);
}

void set_Screen_height(i, j)
Screen* i;
int j;
{
          i->height = j;
}

int  Screen_width(i)
Screen* i;
{
          return(i->width);
}

void set_Screen_width(i, j)
Screen* i;
int j;
{
          i->width = j;
}

int  Screen_root(i)
Screen* i;
{
          return(i->root);
}

void set_Screen_root(i, j)
Screen* i;
int j;
{
          i->root = j;
}

long Screen_display(i)
Screen* i;
{
          return((long) i->display);
}

void set_Screen_display(i, j)
Screen* i;
long j;
{
           i->display = (struct _XDisplay *) j;
}

long  Screen_ext_data(i)
Screen* i;
{
          return((long) i->ext_data);
}

void set_Screen_ext_data(i, j)
Screen* i;
long j;
{
          i->ext_data = (XExtData *) j;
}


/********* ScreenFormat functions *****/

long  make_ScreenFormat (){
          return ((long) calloc(1, sizeof(ScreenFormat)));
}

int  ScreenFormat_scanline_pad(i)
ScreenFormat* i;
{
          return(i->scanline_pad);
}

void set_ScreenFormat_scanline_pad(i, j)
ScreenFormat* i;
int j;
{
          i->scanline_pad = j;
}

int  ScreenFormat_bits_per_pixel(i)
ScreenFormat* i;
{
          return(i->bits_per_pixel);
}

void set_ScreenFormat_bits_per_pixel(i, j)
ScreenFormat* i;
int j;
{
          i->bits_per_pixel = j;
}

int  ScreenFormat_depth(i)
ScreenFormat* i;
{
          return(i->depth);
}

void set_ScreenFormat_depth(i, j)
ScreenFormat* i;
int j;
{
          i->depth = j;
}

long  ScreenFormat_ext_data(i)
ScreenFormat* i;
{
          return((long) i->ext_data);
}

void set_ScreenFormat_ext_data(i, j)
ScreenFormat* i;
long j;
{
          i->ext_data = (XExtData *) j;
}


/********* XSetWindowAttributes functions *****/

long  make_XSetWindowAttributes (){
          return ((long) calloc(1, sizeof(XSetWindowAttributes)));
}

int  XSetWindowAttributes_cursor(i)
XSetWindowAttributes* i;
{
          return(i->cursor);
}

void set_XSetWindowAttributes_cursor(i, j)
XSetWindowAttributes* i;
int j;
{
          i->cursor = j;
}

int  XSetWindowAttributes_colormap(i)
XSetWindowAttributes* i;
{
          return(i->colormap);
}

void set_XSetWindowAttributes_colormap(i, j)
XSetWindowAttributes* i;
int j;
{
          i->colormap = j;
}

int  XSetWindowAttributes_override_redirect(i)
XSetWindowAttributes* i;
{
          return(i->override_redirect);
}

void set_XSetWindowAttributes_override_redirect(i, j)
XSetWindowAttributes* i;
int j;
{
          i->override_redirect = j;
}

int  XSetWindowAttributes_do_not_propagate_mask(i)
XSetWindowAttributes* i;
{
          return(i->do_not_propagate_mask);
}

void set_XSetWindowAttributes_do_not_propagate_mask(i, j)
XSetWindowAttributes* i;
int j;
{
          i->do_not_propagate_mask = j;
}

int  XSetWindowAttributes_event_mask(i)
XSetWindowAttributes* i;
{
          return(i->event_mask);
}

void set_XSetWindowAttributes_event_mask(i, j)
XSetWindowAttributes* i;
int j;
{
          i->event_mask = j;
}

int  XSetWindowAttributes_save_under(i)
XSetWindowAttributes* i;
{
          return(i->save_under);
}

void set_XSetWindowAttributes_save_under(i, j)
XSetWindowAttributes* i;
int j;
{
          i->save_under = j;
}

int  XSetWindowAttributes_backing_pixel(i)
XSetWindowAttributes* i;
{
          return(i->backing_pixel);
}

void set_XSetWindowAttributes_backing_pixel(i, j)
XSetWindowAttributes* i;
int j;
{
          i->backing_pixel = j;
}

int  XSetWindowAttributes_backing_planes(i)
XSetWindowAttributes* i;
{
          return(i->backing_planes);
}

void set_XSetWindowAttributes_backing_planes(i, j)
XSetWindowAttributes* i;
int j;
{
          i->backing_planes = j;
}

int  XSetWindowAttributes_backing_store(i)
XSetWindowAttributes* i;
{
          return(i->backing_store);
}

void set_XSetWindowAttributes_backing_store(i, j)
XSetWindowAttributes* i;
int j;
{
          i->backing_store = j;
}

int  XSetWindowAttributes_win_gravity(i)
XSetWindowAttributes* i;
{
          return(i->win_gravity);
}

void set_XSetWindowAttributes_win_gravity(i, j)
XSetWindowAttributes* i;
int j;
{
          i->win_gravity = j;
}

int  XSetWindowAttributes_bit_gravity(i)
XSetWindowAttributes* i;
{
          return(i->bit_gravity);
}

void set_XSetWindowAttributes_bit_gravity(i, j)
XSetWindowAttributes* i;
int j;
{
          i->bit_gravity = j;
}

int  XSetWindowAttributes_border_pixel(i)
XSetWindowAttributes* i;
{
          return(i->border_pixel);
}

void set_XSetWindowAttributes_border_pixel(i, j)
XSetWindowAttributes* i;
int j;
{
          i->border_pixel = j;
}

int  XSetWindowAttributes_border_pixmap(i)
XSetWindowAttributes* i;
{
          return(i->border_pixmap);
}

void set_XSetWindowAttributes_border_pixmap(i, j)
XSetWindowAttributes* i;
int j;
{
          i->border_pixmap = j;
}

int  XSetWindowAttributes_background_pixel(i)
XSetWindowAttributes* i;
{
          return(i->background_pixel);
}

void set_XSetWindowAttributes_background_pixel(i, j)
XSetWindowAttributes* i;
int j;
{
          i->background_pixel = j;
}

int  XSetWindowAttributes_background_pixmap(i)
XSetWindowAttributes* i;
{
          return(i->background_pixmap);
}

void set_XSetWindowAttributes_background_pixmap(i, j)
XSetWindowAttributes* i;
int j;
{
          i->background_pixmap = j;
}


/********* XWindowAttributes functions *****/

long  make_XWindowAttributes (){
          return ((long) calloc(1, sizeof(XWindowAttributes)));
}

long  XWindowAttributes_screen(i)
XWindowAttributes* i;
{
          return((long) i->screen);
}

void set_XWindowAttributes_screen(i, j)
XWindowAttributes* i;
long j;
{
          i->screen = (Screen *) j;
}

int  XWindowAttributes_override_redirect(i)
XWindowAttributes* i;
{
          return(i->override_redirect);
}

void set_XWindowAttributes_override_redirect(i, j)
XWindowAttributes* i;
int j;
{
          i->override_redirect = j;
}

int  XWindowAttributes_do_not_propagate_mask(i)
XWindowAttributes* i;
{
          return(i->do_not_propagate_mask);
}

void set_XWindowAttributes_do_not_propagate_mask(i, j)
XWindowAttributes* i;
int j;
{
          i->do_not_propagate_mask = j;
}

int  XWindowAttributes_your_event_mask(i)
XWindowAttributes* i;
{
          return(i->your_event_mask);
}

void set_XWindowAttributes_your_event_mask(i, j)
XWindowAttributes* i;
int j;
{
          i->your_event_mask = j;
}

int  XWindowAttributes_all_event_masks(i)
XWindowAttributes* i;
{
          return(i->all_event_masks);
}

void set_XWindowAttributes_all_event_masks(i, j)
XWindowAttributes* i;
int j;
{
          i->all_event_masks = j;
}

int  XWindowAttributes_map_state(i)
XWindowAttributes* i;
{
          return(i->map_state);
}

void set_XWindowAttributes_map_state(i, j)
XWindowAttributes* i;
int j;
{
          i->map_state = j;
}

int  XWindowAttributes_map_installed(i)
XWindowAttributes* i;
{
          return(i->map_installed);
}

void set_XWindowAttributes_map_installed(i, j)
XWindowAttributes* i;
int j;
{
          i->map_installed = j;
}

int  XWindowAttributes_colormap(i)
XWindowAttributes* i;
{
          return(i->colormap);
}

void set_XWindowAttributes_colormap(i, j)
XWindowAttributes* i;
int j;
{
          i->colormap = j;
}

int  XWindowAttributes_save_under(i)
XWindowAttributes* i;
{
          return(i->save_under);
}

void set_XWindowAttributes_save_under(i, j)
XWindowAttributes* i;
int j;
{
          i->save_under = j;
}

int  XWindowAttributes_backing_pixel(i)
XWindowAttributes* i;
{
          return(i->backing_pixel);
}

void set_XWindowAttributes_backing_pixel(i, j)
XWindowAttributes* i;
int j;
{
          i->backing_pixel = j;
}

int  XWindowAttributes_backing_planes(i)
XWindowAttributes* i;
{
          return(i->backing_planes);
}

void set_XWindowAttributes_backing_planes(i, j)
XWindowAttributes* i;
int j;
{
          i->backing_planes = j;
}

int  XWindowAttributes_backing_store(i)
XWindowAttributes* i;
{
          return(i->backing_store);
}

void set_XWindowAttributes_backing_store(i, j)
XWindowAttributes* i;
int j;
{
          i->backing_store = j;
}

int  XWindowAttributes_win_gravity(i)
XWindowAttributes* i;
{
          return(i->win_gravity);
}

void set_XWindowAttributes_win_gravity(i, j)
XWindowAttributes* i;
int j;
{
          i->win_gravity = j;
}

int  XWindowAttributes_bit_gravity(i)
XWindowAttributes* i;
{
          return(i->bit_gravity);
}

void set_XWindowAttributes_bit_gravity(i, j)
XWindowAttributes* i;
int j;
{
          i->bit_gravity = j;
}

int  XWindowAttributes_class(i)
XWindowAttributes* i;
{
          return(i->class);
}

void set_XWindowAttributes_class(i, j)
XWindowAttributes* i;
int j;
{
          i->class = j;
}

int  XWindowAttributes_root(i)
XWindowAttributes* i;
{
          return(i->root);
}

void set_XWindowAttributes_root(i, j)
XWindowAttributes* i;
int j;
{
          i->root = j;
}

long  XWindowAttributes_visual(i)
XWindowAttributes* i;
{
          return((long) i->visual);
}

void set_XWindowAttributes_visual(i, j)
XWindowAttributes* i;
long j;
{
          i->visual = (Visual *) j;
}

int  XWindowAttributes_depth(i)
XWindowAttributes* i;
{
          return(i->depth);
}

void set_XWindowAttributes_depth(i, j)
XWindowAttributes* i;
int j;
{
          i->depth = j;
}

int  XWindowAttributes_border_width(i)
XWindowAttributes* i;
{
          return(i->border_width);
}

void set_XWindowAttributes_border_width(i, j)
XWindowAttributes* i;
int j;
{
          i->border_width = j;
}

int  XWindowAttributes_height(i)
XWindowAttributes* i;
{
          return(i->height);
}

void set_XWindowAttributes_height(i, j)
XWindowAttributes* i;
int j;
{
          i->height = j;
}

int  XWindowAttributes_width(i)
XWindowAttributes* i;
{
          return(i->width);
}

void set_XWindowAttributes_width(i, j)
XWindowAttributes* i;
int j;
{
          i->width = j;
}

int  XWindowAttributes_y(i)
XWindowAttributes* i;
{
          return(i->y);
}

void set_XWindowAttributes_y(i, j)
XWindowAttributes* i;
int j;
{
          i->y = j;
}

int  XWindowAttributes_x(i)
XWindowAttributes* i;
{
          return(i->x);
}

void set_XWindowAttributes_x(i, j)
XWindowAttributes* i;
int j;
{
          i->x = j;
}


/********* XHostAddress functions *****/

long  make_XHostAddress (){
          return ((long) calloc(1, sizeof(XHostAddress)));
}

long  XHostAddress_address(i)
XHostAddress* i;
{
          return((long) i->address);
}

void set_XHostAddress_address(i, j)
XHostAddress* i;
long j;
{
          i->address = (char *) j;
}

int  XHostAddress_length(i)
XHostAddress* i;
{
          return(i->length);
}

void set_XHostAddress_length(i, j)
XHostAddress* i;
int j;
{
          i->length = j;
}

int  XHostAddress_family(i)
XHostAddress* i;
{
          return(i->family);
}

void set_XHostAddress_family(i, j)
XHostAddress* i;
int j;
{
          i->family = j;
}


/********* XImage functions *****/

long  make_XImage (){
          return ((long) calloc(1, sizeof(XImage)));
}

long  XImage_obdata(i)
XImage* i;
{
          return((long) i->obdata);
}

void set_XImage_obdata(i, j)
XImage* i;
long j;
{
          i->obdata = (XPointer) j;
}

int  XImage_blue_mask(i)
XImage* i;
{
          return(i->blue_mask);
}

void set_XImage_blue_mask(i, j)
XImage* i;
int j;
{
          i->blue_mask = j;
}

int  XImage_green_mask(i)
XImage* i;
{
          return(i->green_mask);
}

void set_XImage_green_mask(i, j)
XImage* i;
int j;
{
          i->green_mask = j;
}

int  XImage_red_mask(i)
XImage* i;
{
          return(i->red_mask);
}

void set_XImage_red_mask(i, j)
XImage* i;
int j;
{
          i->red_mask = j;
}

int  XImage_bits_per_pixel(i)
XImage* i;
{
          return(i->bits_per_pixel);
}

void set_XImage_bits_per_pixel(i, j)
XImage* i;
int j;
{
          i->bits_per_pixel = j;
}

int  XImage_bytes_per_line(i)
XImage* i;
{
          return(i->bytes_per_line);
}

void set_XImage_bytes_per_line(i, j)
XImage* i;
int j;
{
          i->bytes_per_line = j;
}

int  XImage_depth(i)
XImage* i;
{
          return(i->depth);
}

void set_XImage_depth(i, j)
XImage* i;
int j;
{
          i->depth = j;
}

int  XImage_bitmap_pad(i)
XImage* i;
{
          return(i->bitmap_pad);
}

void set_XImage_bitmap_pad(i, j)
XImage* i;
int j;
{
          i->bitmap_pad = j;
}

int  XImage_bitmap_bit_order(i)
XImage* i;
{
          return(i->bitmap_bit_order);
}

void set_XImage_bitmap_bit_order(i, j)
XImage* i;
int j;
{
          i->bitmap_bit_order = j;
}

int  XImage_bitmap_unit(i)
XImage* i;
{
          return(i->bitmap_unit);
}

void set_XImage_bitmap_unit(i, j)
XImage* i;
int j;
{
          i->bitmap_unit = j;
}

int  XImage_byte_order(i)
XImage* i;
{
          return(i->byte_order);
}

void set_XImage_byte_order(i, j)
XImage* i;
int j;
{
          i->byte_order = j;
}

long  XImage_data(i)
XImage* i;
{
          return((long) i->data);
}

void set_XImage_data(i, j)
XImage* i;
long j;
{
          i->data = (char *) j;
}

int  XImage_format(i)
XImage* i;
{
          return(i->format);
}

void set_XImage_format(i, j)
XImage* i;
int j;
{
          i->format = j;
}

int  XImage_xoffset(i)
XImage* i;
{
          return(i->xoffset);
}

void set_XImage_xoffset(i, j)
XImage* i;
int j;
{
          i->xoffset = j;
}

int  XImage_height(i)
XImage* i;
{
          return(i->height);
}

void set_XImage_height(i, j)
XImage* i;
int j;
{
          i->height = j;
}

int  XImage_width(i)
XImage* i;
{
          return(i->width);
}

void set_XImage_width(i, j)
XImage* i;
int j;
{
          i->width = j;
}


/********* XWindowChanges functions *****/

long  make_XWindowChanges (){
          return ((long) calloc(1, sizeof(XWindowChanges)));
}

int  XWindowChanges_stack_mode(i)
XWindowChanges* i;
{
          return(i->stack_mode);
}

void set_XWindowChanges_stack_mode(i, j)
XWindowChanges* i;
int j;
{
          i->stack_mode = j;
}

int  XWindowChanges_sibling(i)
XWindowChanges* i;
{
          return(i->sibling);
}

void set_XWindowChanges_sibling(i, j)
XWindowChanges* i;
int j;
{
          i->sibling = j;
}

int  XWindowChanges_border_width(i)
XWindowChanges* i;
{
          return(i->border_width);
}

void set_XWindowChanges_border_width(i, j)
XWindowChanges* i;
int j;
{
          i->border_width = j;
}

int  XWindowChanges_height(i)
XWindowChanges* i;
{
          return(i->height);
}

void set_XWindowChanges_height(i, j)
XWindowChanges* i;
int j;
{
          i->height = j;
}

int  XWindowChanges_width(i)
XWindowChanges* i;
{
          return(i->width);
}

void set_XWindowChanges_width(i, j)
XWindowChanges* i;
int j;
{
          i->width = j;
}

int  XWindowChanges_y(i)
XWindowChanges* i;
{
          return(i->y);
}

void set_XWindowChanges_y(i, j)
XWindowChanges* i;
int j;
{
          i->y = j;
}

int  XWindowChanges_x(i)
XWindowChanges* i;
{
          return(i->x);
}

void set_XWindowChanges_x(i, j)
XWindowChanges* i;
int j;
{
          i->x = j;
}


/********* XColor functions *****/

long  make_XColor (){
          return ((long) calloc(1, sizeof(XColor)));
}

char XColor_pad(i)
XColor* i;
{
          return(i->pad);
}

void set_XColor_pad(i, j)
XColor* i;
char j;
{
          i->pad = j;
}

char XColor_flags(i)
XColor* i;
{
          return(i->flags);
}

void set_XColor_flags(i, j)
XColor* i;
char j;
{
          i->flags = j;
}

int  XColor_blue(i)
XColor* i;
{
          return(i->blue);
}

void set_XColor_blue(i, j)
XColor* i;
int j;
{
          i->blue = j;
}

int  XColor_green(i)
XColor* i;
{
          return(i->green);
}

void set_XColor_green(i, j)
XColor* i;
int j;
{
          i->green = j;
}

int  XColor_red(i)
XColor* i;
{
          return(i->red);
}

void set_XColor_red(i, j)
XColor* i;
int j;
{
          i->red = j;
}

int  XColor_pixel(i)
XColor* i;
{
          return(i->pixel);
}

void set_XColor_pixel(i, j)
XColor* i;
int j;
{
          i->pixel = j;
}


/********* XSegment functions *****/

long  make_XSegment (){
          return ((long) calloc(1, sizeof(XSegment)));
}

int  XSegment_y2(i)
XSegment* i;
{
          return(i->y2);
}

void set_XSegment_y2(i, j)
XSegment* i;
int j;
{
          i->y2 = j;
}

int  XSegment_x2(i)
XSegment* i;
{
          return(i->x2);
}

void set_XSegment_x2(i, j)
XSegment* i;
int j;
{
          i->x2 = j;
}

int  XSegment_y1(i)
XSegment* i;
{
          return(i->y1);
}

void set_XSegment_y1(i, j)
XSegment* i;
int j;
{
          i->y1 = j;
}

int  XSegment_x1(i)
XSegment* i;
{
          return(i->x1);
}

void set_XSegment_x1(i, j)
XSegment* i;
int j;
{
          i->x1 = j;
}


/********* XPoint functions *****/

long  make_XPoint (){
          return ((long) calloc(1, sizeof(XPoint)));
}

int  XPoint_y(i)
XPoint* i;
{
          return(i->y);
}

void set_XPoint_y(i, j)
XPoint* i;
int j;
{
          i->y = j;
}

int  XPoint_x(i)
XPoint* i;
{
          return(i->x);
}

void set_XPoint_x(i, j)
XPoint* i;
int j;
{
          i->x = j;
}


/********* XRectangle functions *****/

long  make_XRectangle (){
          return ((long) calloc(1, sizeof(XRectangle)));
}

int  XRectangle_height(i)
XRectangle* i;
{
          return(i->height);
}

void set_XRectangle_height(i, j)
XRectangle* i;
int j;
{
          i->height = j;
}

int  XRectangle_width(i)
XRectangle* i;
{
          return(i->width);
}

void set_XRectangle_width(i, j)
XRectangle* i;
int j;
{
          i->width = j;
}

int  XRectangle_y(i)
XRectangle* i;
{
          return(i->y);
}

void set_XRectangle_y(i, j)
XRectangle* i;
int j;
{
          i->y = j;
}

int  XRectangle_x(i)
XRectangle* i;
{
          return(i->x);
}

void set_XRectangle_x(i, j)
XRectangle* i;
int j;
{
          i->x = j;
}


/********* XArc functions *****/

long  make_XArc (){
          return ((long) calloc(1, sizeof(XArc)));
}

int  XArc_angle2(i)
XArc* i;
{
          return(i->angle2);
}

void set_XArc_angle2(i, j)
XArc* i;
int j;
{
          i->angle2 = j;
}

int  XArc_angle1(i)
XArc* i;
{
          return(i->angle1);
}

void set_XArc_angle1(i, j)
XArc* i;
int j;
{
          i->angle1 = j;
}

int  XArc_height(i)
XArc* i;
{
          return(i->height);
}

void set_XArc_height(i, j)
XArc* i;
int j;
{
          i->height = j;
}

int  XArc_width(i)
XArc* i;
{
          return(i->width);
}

void set_XArc_width(i, j)
XArc* i;
int j;
{
          i->width = j;
}

int  XArc_y(i)
XArc* i;
{
          return(i->y);
}

void set_XArc_y(i, j)
XArc* i;
int j;
{
          i->y = j;
}

int  XArc_x(i)
XArc* i;
{
          return(i->x);
}

void set_XArc_x(i, j)
XArc* i;
int j;
{
          i->x = j;
}


/********* XKeyboardControl functions *****/

long  make_XKeyboardControl (){
          return ((long) calloc(1, sizeof(XKeyboardControl)));
}

int  XKeyboardControl_auto_repeat_mode(i)
XKeyboardControl* i;
{
          return(i->auto_repeat_mode);
}

void set_XKeyboardControl_auto_repeat_mode(i, j)
XKeyboardControl* i;
int j;
{
          i->auto_repeat_mode = j;
}

int  XKeyboardControl_key(i)
XKeyboardControl* i;
{
          return(i->key);
}

void set_XKeyboardControl_key(i, j)
XKeyboardControl* i;
int j;
{
          i->key = j;
}

int  XKeyboardControl_led_mode(i)
XKeyboardControl* i;
{
          return(i->led_mode);
}

void set_XKeyboardControl_led_mode(i, j)
XKeyboardControl* i;
int j;
{
          i->led_mode = j;
}

int  XKeyboardControl_led(i)
XKeyboardControl* i;
{
          return(i->led);
}

void set_XKeyboardControl_led(i, j)
XKeyboardControl* i;
int j;
{
          i->led = j;
}

int  XKeyboardControl_bell_duration(i)
XKeyboardControl* i;
{
          return(i->bell_duration);
}

void set_XKeyboardControl_bell_duration(i, j)
XKeyboardControl* i;
int j;
{
          i->bell_duration = j;
}

int  XKeyboardControl_bell_pitch(i)
XKeyboardControl* i;
{
          return(i->bell_pitch);
}

void set_XKeyboardControl_bell_pitch(i, j)
XKeyboardControl* i;
int j;
{
          i->bell_pitch = j;
}

int  XKeyboardControl_bell_percent(i)
XKeyboardControl* i;
{
          return(i->bell_percent);
}

void set_XKeyboardControl_bell_percent(i, j)
XKeyboardControl* i;
int j;
{
          i->bell_percent = j;
}

int  XKeyboardControl_key_click_percent(i)
XKeyboardControl* i;
{
          return(i->key_click_percent);
}

void set_XKeyboardControl_key_click_percent(i, j)
XKeyboardControl* i;
int j;
{
          i->key_click_percent = j;
}


/********* XKeyboardState functions *****/

long  make_XKeyboardState (){
          return ((long) calloc(1, sizeof(XKeyboardState)));
}

char *XKeyboardState_auto_repeats(i)
XKeyboardState* i;
{
          return(i->auto_repeats);
}

void set_XKeyboardState_auto_repeats(i, j)
XKeyboardState* i;
char	*j;
{
          strcpy(i->auto_repeats,  j);
}

int  XKeyboardState_global_auto_repeat(i)
XKeyboardState* i;
{
          return(i->global_auto_repeat);
}

void set_XKeyboardState_global_auto_repeat(i, j)
XKeyboardState* i;
int j;
{
          i->global_auto_repeat = j;
}

int  XKeyboardState_led_mask(i)
XKeyboardState* i;
{
          return(i->led_mask);
}

void set_XKeyboardState_led_mask(i, j)
XKeyboardState* i;
int j;
{
          i->led_mask = j;
}

int  XKeyboardState_bell_duration(i)
XKeyboardState* i;
{
          return(i->bell_duration);
}

void set_XKeyboardState_bell_duration(i, j)
XKeyboardState* i;
int j;
{
          i->bell_duration = j;
}

int  XKeyboardState_bell_pitch(i)
XKeyboardState* i;
{
          return(i->bell_pitch);
}

void set_XKeyboardState_bell_pitch(i, j)
XKeyboardState* i;
int j;
{
          i->bell_pitch = j;
}

int  XKeyboardState_bell_percent(i)
XKeyboardState* i;
{
          return(i->bell_percent);
}

void set_XKeyboardState_bell_percent(i, j)
XKeyboardState* i;
int j;
{
          i->bell_percent = j;
}

int  XKeyboardState_key_click_percent(i)
XKeyboardState* i;
{
          return(i->key_click_percent);
}

void set_XKeyboardState_key_click_percent(i, j)
XKeyboardState* i;
int j;
{
          i->key_click_percent = j;
}


/********* XTimeCoord functions *****/

long  make_XTimeCoord (){
          return ((long) calloc(1, sizeof(XTimeCoord)));
}

int  XTimeCoord_y(i)
XTimeCoord* i;
{
          return(i->y);
}

void set_XTimeCoord_y(i, j)
XTimeCoord* i;
int j;
{
          i->y = j;
}

int  XTimeCoord_x(i)
XTimeCoord* i;
{
          return(i->x);
}

void set_XTimeCoord_x(i, j)
XTimeCoord* i;
int j;
{
          i->x = j;
}

int  XTimeCoord_time(i)
XTimeCoord* i;
{
          return(i->time);
}

void set_XTimeCoord_time(i, j)
XTimeCoord* i;
int j;
{
          i->time = j;
}


/********* XModifierKeymap functions *****/

long  make_XModifierKeymap (){
          return ((long) calloc(1, sizeof(XModifierKeymap)));
}

long  XModifierKeymap_modifiermap(i)
XModifierKeymap* i;
{
          return((long) i->modifiermap);
}

void set_XModifierKeymap_modifiermap(i, j)
XModifierKeymap* i;
long j;
{
          i->modifiermap = (KeyCode *) j;
}

int  XModifierKeymap_max_keypermod(i)
XModifierKeymap* i;
{
          return(i->max_keypermod);
}

void set_XModifierKeymap_max_keypermod(i, j)
XModifierKeymap* i;
int j;
{
          i->max_keypermod = j;
}
