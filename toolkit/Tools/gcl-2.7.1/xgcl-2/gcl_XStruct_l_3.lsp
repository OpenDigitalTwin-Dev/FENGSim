(in-package :XLIB)
; XStruct-l-3.lsp        modified by Hiep Huu Nguyen                27 Aug 92

; Copyright (c) 1994 Hiep Huu Nguyen and The University of Texas at Austin.
; Copyright (c) 2024 Camm Maguire

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
; See the file dec.copyright for details.




;;;;;; XExtCodes functions ;;;;;;

(defentry make-XExtCodes () ( fixnum  "make_XExtCodes" ))
(defentry XExtCodes-first_error (fixnum) ( fixnum "XExtCodes_first_error" ))
(defentry set-XExtCodes-first_error (fixnum fixnum) ( void "set_XExtCodes_first_error" ))
(defentry XExtCodes-first_event (fixnum) ( fixnum "XExtCodes_first_event" ))
(defentry set-XExtCodes-first_event (fixnum fixnum) ( void "set_XExtCodes_first_event" ))
(defentry XExtCodes-major_opcode (fixnum) ( fixnum "XExtCodes_major_opcode" ))
(defentry set-XExtCodes-major_opcode (fixnum fixnum) ( void "set_XExtCodes_major_opcode" ))
(defentry XExtCodes-extension (fixnum) ( fixnum "XExtCodes_extension" ))
(defentry set-XExtCodes-extension (fixnum fixnum) ( void "set_XExtCodes_extension" ))


;;;;;; XPixmapFormatValues functions ;;;;;;

(defentry make-XPixmapFormatValues () ( fixnum  "make_XPixmapFormatValues" ))
(defentry XPixmapFormatValues-scanline_pad (fixnum) ( fixnum "XPixmapFormatValues_scanline_pad" ))
(defentry set-XPixmapFormatValues-scanline_pad (fixnum fixnum) ( void "set_XPixmapFormatValues_scanline_pad" ))
(defentry XPixmapFormatValues-bits_per_pixel (fixnum) ( fixnum "XPixmapFormatValues_bits_per_pixel" ))
(defentry set-XPixmapFormatValues-bits_per_pixel (fixnum fixnum) ( void "set_XPixmapFormatValues_bits_per_pixel" ))
(defentry XPixmapFormatValues-depth (fixnum) ( fixnum "XPixmapFormatValues_depth" ))
(defentry set-XPixmapFormatValues-depth (fixnum fixnum) ( void "set_XPixmapFormatValues_depth" ))


;;;;;; XGCValues functions ;;;;;;

(defentry make-XGCValues () ( fixnum  "make_XGCValues" ))
(defentry XGCValues-dashes (fixnum) ( char "XGCValues_dashes" ))
(defentry set-XGCValues-dashes (fixnum char) ( void "set_XGCValues_dashes" ))
(defentry XGCValues-dash_offset (fixnum) ( fixnum "XGCValues_dash_offset" ))
(defentry set-XGCValues-dash_offset (fixnum fixnum) ( void "set_XGCValues_dash_offset" ))
(defentry XGCValues-clip_mask (fixnum) ( fixnum "XGCValues_clip_mask" ))
(defentry set-XGCValues-clip_mask (fixnum fixnum) ( void "set_XGCValues_clip_mask" ))
(defentry XGCValues-clip_y_origin (fixnum) ( fixnum "XGCValues_clip_y_origin" ))
(defentry set-XGCValues-clip_y_origin (fixnum fixnum) ( void "set_XGCValues_clip_y_origin" ))
(defentry XGCValues-clip_x_origin (fixnum) ( fixnum "XGCValues_clip_x_origin" ))
(defentry set-XGCValues-clip_x_origin (fixnum fixnum) ( void "set_XGCValues_clip_x_origin" ))
(defentry XGCValues-graphics_exposures (fixnum) ( fixnum "XGCValues_graphics_exposures" ))
(defentry set-XGCValues-graphics_exposures (fixnum fixnum) ( void "set_XGCValues_graphics_exposures" ))
(defentry XGCValues-subwindow_mode (fixnum) ( fixnum "XGCValues_subwindow_mode" ))
(defentry set-XGCValues-subwindow_mode (fixnum fixnum) ( void "set_XGCValues_subwindow_mode" ))
(defentry XGCValues-font (fixnum) ( fixnum "XGCValues_font" ))
(defentry set-XGCValues-font (fixnum fixnum) ( void "set_XGCValues_font" ))
(defentry XGCValues-ts_y_origin (fixnum) ( fixnum "XGCValues_ts_y_origin" ))
(defentry set-XGCValues-ts_y_origin (fixnum fixnum) ( void "set_XGCValues_ts_y_origin" ))
(defentry XGCValues-ts_x_origin (fixnum) ( fixnum "XGCValues_ts_x_origin" ))
(defentry set-XGCValues-ts_x_origin (fixnum fixnum) ( void "set_XGCValues_ts_x_origin" ))
(defentry XGCValues-stipple (fixnum) ( fixnum "XGCValues_stipple" ))
(defentry set-XGCValues-stipple (fixnum fixnum) ( void "set_XGCValues_stipple" ))
(defentry XGCValues-tile (fixnum) ( fixnum "XGCValues_tile" ))
(defentry set-XGCValues-tile (fixnum fixnum) ( void "set_XGCValues_tile" ))
(defentry XGCValues-arc_mode (fixnum) ( fixnum "XGCValues_arc_mode" ))
(defentry set-XGCValues-arc_mode (fixnum fixnum) ( void "set_XGCValues_arc_mode" ))
(defentry XGCValues-fill_rule (fixnum) ( fixnum "XGCValues_fill_rule" ))
(defentry set-XGCValues-fill_rule (fixnum fixnum) ( void "set_XGCValues_fill_rule" ))
(defentry XGCValues-fill_style (fixnum) ( fixnum "XGCValues_fill_style" ))
(defentry set-XGCValues-fill_style (fixnum fixnum) ( void "set_XGCValues_fill_style" ))
(defentry XGCValues-join_style (fixnum) ( fixnum "XGCValues_join_style" ))
(defentry set-XGCValues-join_style (fixnum fixnum) ( void "set_XGCValues_join_style" ))
(defentry XGCValues-cap_style (fixnum) ( fixnum "XGCValues_cap_style" ))
(defentry set-XGCValues-cap_style (fixnum fixnum) ( void "set_XGCValues_cap_style" ))
(defentry XGCValues-line_style (fixnum) ( fixnum "XGCValues_line_style" ))
(defentry set-XGCValues-line_style (fixnum fixnum) ( void "set_XGCValues_line_style" ))
(defentry XGCValues-line_width (fixnum) ( fixnum "XGCValues_line_width" ))
(defentry set-XGCValues-line_width (fixnum fixnum) ( void "set_XGCValues_line_width" ))
(defentry XGCValues-background (fixnum) ( fixnum "XGCValues_background" ))
(defentry set-XGCValues-background (fixnum fixnum) ( void "set_XGCValues_background" ))
(defentry XGCValues-foreground (fixnum) ( fixnum "XGCValues_foreground" ))
(defentry set-XGCValues-foreground (fixnum fixnum) ( void "set_XGCValues_foreground" ))
(defentry XGCValues-plane_mask (fixnum) ( fixnum "XGCValues_plane_mask" ))
(defentry set-XGCValues-plane_mask (fixnum fixnum) ( void "set_XGCValues_plane_mask" ))
(defentry XGCValues-function (fixnum) ( fixnum "XGCValues_function" ))
(defentry set-XGCValues-function (fixnum fixnum) ( void "set_XGCValues_function" ))


;;;;;; *GC functions ;;;;;;

;;(defentry make-*GC () ( fixnum  "make_*GC" ))
;;(defentry *GC-values (fixnum) ( fixnum "*GC_values" ))
;;(defentry set-*GC-values (fixnum fixnum) ( void "set_*GC_values" ))
;;(defentry *GC-dirty (fixnum) ( fixnum "*GC_dirty" ))
;;(defentry set-*GC-dirty (fixnum fixnum) ( void "set_*GC_dirty" ))
;;(defentry *GC-dashes (fixnum) ( fixnum "*GC_dashes" ))
;;(defentry set-*GC-dashes (fixnum fixnum) ( void "set_*GC_dashes" ))
;;(defentry *GC-rects (fixnum) ( fixnum "*GC_rects" ))
;;(defentry set-*GC-rects (fixnum fixnum) ( void "set_*GC_rects" ))
;;(defentry *GC-gid (fixnum) ( fixnum "*GC_gid" ))
;;(defentry set-*GC-gid (fixnum fixnum) ( void "set_*GC_gid" ))
;;(defentry *GC-ext_data (fixnum) ( fixnum "*GC_ext_data" ))
;;(defentry set-*GC-ext_data (fixnum fixnum) ( void "set_*GC_ext_data" ))


;;;;;; Visual functions ;;;;;;

(defentry make-Visual () ( fixnum  "make_Visual" ))
(defentry Visual-map_entries (fixnum) ( fixnum "Visual_map_entries" ))
(defentry set-Visual-map_entries (fixnum fixnum) ( void "set_Visual_map_entries" ))
(defentry Visual-bits_per_rgb (fixnum) ( fixnum "Visual_bits_per_rgb" ))
(defentry set-Visual-bits_per_rgb (fixnum fixnum) ( void "set_Visual_bits_per_rgb" ))
(defentry Visual-blue_mask (fixnum) ( fixnum "Visual_blue_mask" ))
(defentry set-Visual-blue_mask (fixnum fixnum) ( void "set_Visual_blue_mask" ))
(defentry Visual-green_mask (fixnum) ( fixnum "Visual_green_mask" ))
(defentry set-Visual-green_mask (fixnum fixnum) ( void "set_Visual_green_mask" ))
(defentry Visual-red_mask (fixnum) ( fixnum "Visual_red_mask" ))
(defentry set-Visual-red_mask (fixnum fixnum) ( void "set_Visual_red_mask" ))
(defentry Visual-class (fixnum) ( fixnum "Visual_class" ))
(defentry set-Visual-class (fixnum fixnum) ( void "set_Visual_class" ))
(defentry Visual-visualid (fixnum) ( fixnum "Visual_visualid" ))
(defentry set-Visual-visualid (fixnum fixnum) ( void "set_Visual_visualid" ))
(defentry Visual-ext_data (fixnum) ( fixnum "Visual_ext_data" ))
(defentry set-Visual-ext_data (fixnum fixnum) ( void "set_Visual_ext_data" ))


;;;;;; Depth functions ;;;;;;

(defentry make-Depth () ( fixnum  "make_Depth" ))
(defentry Depth-visuals (fixnum) ( fixnum "Depth_visuals" ))
(defentry set-Depth-visuals (fixnum fixnum) ( void "set_Depth_visuals" ))
(defentry Depth-nvisuals (fixnum) ( fixnum "Depth_nvisuals" ))
(defentry set-Depth-nvisuals (fixnum fixnum) ( void "set_Depth_nvisuals" ))
(defentry Depth-depth (fixnum) ( fixnum "Depth_depth" ))
(defentry set-Depth-depth (fixnum fixnum) ( void "set_Depth_depth" ))


;;;;;; Screen functions ;;;;;;

(defentry make-Screen () ( fixnum  "make_Screen" ))
(defentry Screen-root_input_mask (fixnum) ( fixnum "Screen_root_input_mask" ))
(defentry set-Screen-root_input_mask (fixnum fixnum) ( void "set_Screen_root_input_mask" ))
(defentry Screen-save_unders (fixnum) ( fixnum "Screen_save_unders" ))
(defentry set-Screen-save_unders (fixnum fixnum) ( void "set_Screen_save_unders" ))
(defentry Screen-backing_store (fixnum) ( fixnum "Screen_backing_store" ))
(defentry set-Screen-backing_store (fixnum fixnum) ( void "set_Screen_backing_store" ))
(defentry Screen-min_maps (fixnum) ( fixnum "Screen_min_maps" ))
(defentry set-Screen-min_maps (fixnum fixnum) ( void "set_Screen_min_maps" ))
(defentry Screen-max_maps (fixnum) ( fixnum "Screen_max_maps" ))
(defentry set-Screen-max_maps (fixnum fixnum) ( void "set_Screen_max_maps" ))
(defentry Screen-black_pixel (fixnum) ( fixnum "Screen_black_pixel" ))
(defentry set-Screen-black_pixel (fixnum fixnum) ( void "set_Screen_black_pixel" ))
(defentry Screen-white_pixel (fixnum) ( fixnum "Screen_white_pixel" ))
(defentry set-Screen-white_pixel (fixnum fixnum) ( void "set_Screen_white_pixel" ))
(defentry Screen-cmap (fixnum) ( fixnum "Screen_cmap" ))
(defentry set-Screen-cmap (fixnum fixnum) ( void "set_Screen_cmap" ))
(defentry Screen-default_gc (fixnum) ( fixnum "Screen_default_gc" ))
(defentry set-Screen-default_gc (fixnum fixnum) ( void "set_Screen_default_gc" ))
(defentry Screen-root_visual (fixnum) ( fixnum "Screen_root_visual" ))
(defentry set-Screen-root_visual (fixnum fixnum) ( void "set_Screen_root_visual" ))
(defentry Screen-root_depth (fixnum) ( fixnum "Screen_root_depth" ))
(defentry set-Screen-root_depth (fixnum fixnum) ( void "set_Screen_root_depth" ))
(defentry Screen-depths (fixnum) ( fixnum "Screen_depths" ))
(defentry set-Screen-depths (fixnum fixnum) ( void "set_Screen_depths" ))
(defentry Screen-ndepths (fixnum) ( fixnum "Screen_ndepths" ))
(defentry set-Screen-ndepths (fixnum fixnum) ( void "set_Screen_ndepths" ))
(defentry Screen-mheight (fixnum) ( fixnum "Screen_mheight" ))
(defentry set-Screen-mheight (fixnum fixnum) ( void "set_Screen_mheight" ))
(defentry Screen-mwidth (fixnum) ( fixnum "Screen_mwidth" ))
(defentry set-Screen-mwidth (fixnum fixnum) ( void "set_Screen_mwidth" ))
(defentry Screen-height (fixnum) ( fixnum "Screen_height" ))
(defentry set-Screen-height (fixnum fixnum) ( void "set_Screen_height" ))
(defentry Screen-width (fixnum) ( fixnum "Screen_width" ))
(defentry set-Screen-width (fixnum fixnum) ( void "set_Screen_width" ))
(defentry Screen-root (fixnum) ( fixnum "Screen_root" ))
(defentry set-Screen-root (fixnum fixnum) ( void "set_Screen_root" ))
(defentry Screen-display (fixnum) ( fixnum "Screen_display" ))
(defentry set-Screen-display (fixnum fixnum) ( void "set_Screen_display" ))
(defentry Screen-ext_data (fixnum) ( fixnum "Screen_ext_data" ))
(defentry set-Screen-ext_data (fixnum fixnum) ( void "set_Screen_ext_data" ))


;;;;;; ScreenFormat functions ;;;;;;

(defentry make-ScreenFormat () ( fixnum  "make_ScreenFormat" ))
(defentry ScreenFormat-scanline_pad (fixnum) ( fixnum "ScreenFormat_scanline_pad" ))
(defentry set-ScreenFormat-scanline_pad (fixnum fixnum) ( void "set_ScreenFormat_scanline_pad" ))
(defentry ScreenFormat-bits_per_pixel (fixnum) ( fixnum "ScreenFormat_bits_per_pixel" ))
(defentry set-ScreenFormat-bits_per_pixel (fixnum fixnum) ( void "set_ScreenFormat_bits_per_pixel" ))
(defentry ScreenFormat-depth (fixnum) ( fixnum "ScreenFormat_depth" ))
(defentry set-ScreenFormat-depth (fixnum fixnum) ( void "set_ScreenFormat_depth" ))
(defentry ScreenFormat-ext_data (fixnum) ( fixnum "ScreenFormat_ext_data" ))
(defentry set-ScreenFormat-ext_data (fixnum fixnum) ( void "set_ScreenFormat_ext_data" ))


;;;;;; XSetWindowAttributes functions ;;;;;;

(defentry make-XSetWindowAttributes () ( fixnum  "make_XSetWindowAttributes" ))
(defentry XSetWindowAttributes-cursor (fixnum) ( fixnum "XSetWindowAttributes_cursor" ))
(defentry set-XSetWindowAttributes-cursor (fixnum fixnum) ( void "set_XSetWindowAttributes_cursor" ))
(defentry XSetWindowAttributes-colormap (fixnum) ( fixnum "XSetWindowAttributes_colormap" ))
(defentry set-XSetWindowAttributes-colormap (fixnum fixnum) ( void "set_XSetWindowAttributes_colormap" ))
(defentry XSetWindowAttributes-override_redirect (fixnum) ( fixnum "XSetWindowAttributes_override_redirect" ))
(defentry set-XSetWindowAttributes-override_redirect (fixnum fixnum) ( void "set_XSetWindowAttributes_override_redirect" ))
(defentry XSetWindowAttributes-do_not_propagate_mask (fixnum) ( fixnum "XSetWindowAttributes_do_not_propagate_mask" ))
(defentry set-XSetWindowAttributes-do_not_propagate_mask (fixnum fixnum) ( void "set_XSetWindowAttributes_do_not_propagate_mask" ))
(defentry XSetWindowAttributes-event_mask (fixnum) ( fixnum "XSetWindowAttributes_event_mask" ))
(defentry set-XSetWindowAttributes-event_mask (fixnum fixnum) ( void "set_XSetWindowAttributes_event_mask" ))
(defentry XSetWindowAttributes-save_under (fixnum) ( fixnum "XSetWindowAttributes_save_under" ))
(defentry set-XSetWindowAttributes-save_under (fixnum fixnum) ( void "set_XSetWindowAttributes_save_under" ))
(defentry XSetWindowAttributes-backing_pixel (fixnum) ( fixnum "XSetWindowAttributes_backing_pixel" ))
(defentry set-XSetWindowAttributes-backing_pixel (fixnum fixnum) ( void "set_XSetWindowAttributes_backing_pixel" ))
(defentry XSetWindowAttributes-backing_planes (fixnum) ( fixnum "XSetWindowAttributes_backing_planes" ))
(defentry set-XSetWindowAttributes-backing_planes (fixnum fixnum) ( void "set_XSetWindowAttributes_backing_planes" ))
(defentry XSetWindowAttributes-backing_store (fixnum) ( fixnum "XSetWindowAttributes_backing_store" ))
(defentry set-XSetWindowAttributes-backing_store (fixnum fixnum) ( void "set_XSetWindowAttributes_backing_store" ))
(defentry XSetWindowAttributes-win_gravity (fixnum) ( fixnum "XSetWindowAttributes_win_gravity" ))
(defentry set-XSetWindowAttributes-win_gravity (fixnum fixnum) ( void "set_XSetWindowAttributes_win_gravity" ))
(defentry XSetWindowAttributes-bit_gravity (fixnum) ( fixnum "XSetWindowAttributes_bit_gravity" ))
(defentry set-XSetWindowAttributes-bit_gravity (fixnum fixnum) ( void "set_XSetWindowAttributes_bit_gravity" ))
(defentry XSetWindowAttributes-border_pixel (fixnum) ( fixnum "XSetWindowAttributes_border_pixel" ))
(defentry set-XSetWindowAttributes-border_pixel (fixnum fixnum) ( void "set_XSetWindowAttributes_border_pixel" ))
(defentry XSetWindowAttributes-border_pixmap (fixnum) ( fixnum "XSetWindowAttributes_border_pixmap" ))
(defentry set-XSetWindowAttributes-border_pixmap (fixnum fixnum) ( void "set_XSetWindowAttributes_border_pixmap" ))
(defentry XSetWindowAttributes-background_pixel (fixnum) ( fixnum "XSetWindowAttributes_background_pixel" ))
(defentry set-XSetWindowAttributes-background_pixel (fixnum fixnum) ( void "set_XSetWindowAttributes_background_pixel" ))
(defentry XSetWindowAttributes-background_pixmap (fixnum) ( fixnum "XSetWindowAttributes_background_pixmap" ))
(defentry set-XSetWindowAttributes-background_pixmap (fixnum fixnum) ( void "set_XSetWindowAttributes_background_pixmap" ))


;;;;;; XWindowAttributes functions ;;;;;;

(defentry make-XWindowAttributes () ( fixnum  "make_XWindowAttributes" ))
(defentry XWindowAttributes-screen (fixnum) ( fixnum "XWindowAttributes_screen" ))
(defentry set-XWindowAttributes-screen (fixnum fixnum) ( void "set_XWindowAttributes_screen" ))
(defentry XWindowAttributes-override_redirect (fixnum) ( fixnum "XWindowAttributes_override_redirect" ))
(defentry set-XWindowAttributes-override_redirect (fixnum fixnum) ( void "set_XWindowAttributes_override_redirect" ))
(defentry XWindowAttributes-do_not_propagate_mask (fixnum) ( fixnum "XWindowAttributes_do_not_propagate_mask" ))
(defentry set-XWindowAttributes-do_not_propagate_mask (fixnum fixnum) ( void "set_XWindowAttributes_do_not_propagate_mask" ))
(defentry XWindowAttributes-your_event_mask (fixnum) ( fixnum "XWindowAttributes_your_event_mask" ))
(defentry set-XWindowAttributes-your_event_mask (fixnum fixnum) ( void "set_XWindowAttributes_your_event_mask" ))
(defentry XWindowAttributes-all_event_masks (fixnum) ( fixnum "XWindowAttributes_all_event_masks" ))
(defentry set-XWindowAttributes-all_event_masks (fixnum fixnum) ( void "set_XWindowAttributes_all_event_masks" ))
(defentry XWindowAttributes-map_state (fixnum) ( fixnum "XWindowAttributes_map_state" ))
(defentry set-XWindowAttributes-map_state (fixnum fixnum) ( void "set_XWindowAttributes_map_state" ))
(defentry XWindowAttributes-map_installed (fixnum) ( fixnum "XWindowAttributes_map_installed" ))
(defentry set-XWindowAttributes-map_installed (fixnum fixnum) ( void "set_XWindowAttributes_map_installed" ))
(defentry XWindowAttributes-colormap (fixnum) ( fixnum "XWindowAttributes_colormap" ))
(defentry set-XWindowAttributes-colormap (fixnum fixnum) ( void "set_XWindowAttributes_colormap" ))
(defentry XWindowAttributes-save_under (fixnum) ( fixnum "XWindowAttributes_save_under" ))
(defentry set-XWindowAttributes-save_under (fixnum fixnum) ( void "set_XWindowAttributes_save_under" ))
(defentry XWindowAttributes-backing_pixel (fixnum) ( fixnum "XWindowAttributes_backing_pixel" ))
(defentry set-XWindowAttributes-backing_pixel (fixnum fixnum) ( void "set_XWindowAttributes_backing_pixel" ))
(defentry XWindowAttributes-backing_planes (fixnum) ( fixnum "XWindowAttributes_backing_planes" ))
(defentry set-XWindowAttributes-backing_planes (fixnum fixnum) ( void "set_XWindowAttributes_backing_planes" ))
(defentry XWindowAttributes-backing_store (fixnum) ( fixnum "XWindowAttributes_backing_store" ))
(defentry set-XWindowAttributes-backing_store (fixnum fixnum) ( void "set_XWindowAttributes_backing_store" ))
(defentry XWindowAttributes-win_gravity (fixnum) ( fixnum "XWindowAttributes_win_gravity" ))
(defentry set-XWindowAttributes-win_gravity (fixnum fixnum) ( void "set_XWindowAttributes_win_gravity" ))
(defentry XWindowAttributes-bit_gravity (fixnum) ( fixnum "XWindowAttributes_bit_gravity" ))
(defentry set-XWindowAttributes-bit_gravity (fixnum fixnum) ( void "set_XWindowAttributes_bit_gravity" ))
(defentry XWindowAttributes-class (fixnum) ( fixnum "XWindowAttributes_class" ))
(defentry set-XWindowAttributes-class (fixnum fixnum) ( void "set_XWindowAttributes_class" ))
(defentry XWindowAttributes-root (fixnum) ( fixnum "XWindowAttributes_root" ))
(defentry set-XWindowAttributes-root (fixnum fixnum) ( void "set_XWindowAttributes_root" ))
(defentry XWindowAttributes-visual (fixnum) ( fixnum "XWindowAttributes_visual" ))
(defentry set-XWindowAttributes-visual (fixnum fixnum) ( void "set_XWindowAttributes_visual" ))
(defentry XWindowAttributes-depth (fixnum) ( fixnum "XWindowAttributes_depth" ))
(defentry set-XWindowAttributes-depth (fixnum fixnum) ( void "set_XWindowAttributes_depth" ))
(defentry XWindowAttributes-border_width (fixnum) ( fixnum "XWindowAttributes_border_width" ))
(defentry set-XWindowAttributes-border_width (fixnum fixnum) ( void "set_XWindowAttributes_border_width" ))
(defentry XWindowAttributes-height (fixnum) ( fixnum "XWindowAttributes_height" ))
(defentry set-XWindowAttributes-height (fixnum fixnum) ( void "set_XWindowAttributes_height" ))
(defentry XWindowAttributes-width (fixnum) ( fixnum "XWindowAttributes_width" ))
(defentry set-XWindowAttributes-width (fixnum fixnum) ( void "set_XWindowAttributes_width" ))
(defentry XWindowAttributes-y (fixnum) ( fixnum "XWindowAttributes_y" ))
(defentry set-XWindowAttributes-y (fixnum fixnum) ( void "set_XWindowAttributes_y" ))
(defentry XWindowAttributes-x (fixnum) ( fixnum "XWindowAttributes_x" ))
(defentry set-XWindowAttributes-x (fixnum fixnum) ( void "set_XWindowAttributes_x" ))


;;;;;; XHostAddress functions ;;;;;;

(defentry make-XHostAddress () ( fixnum  "make_XHostAddress" ))
(defentry XHostAddress-address (fixnum) ( fixnum "XHostAddress_address" ))
(defentry set-XHostAddress-address (fixnum fixnum) ( void "set_XHostAddress_address" ))
(defentry XHostAddress-length (fixnum) ( fixnum "XHostAddress_length" ))
(defentry set-XHostAddress-length (fixnum fixnum) ( void "set_XHostAddress_length" ))
(defentry XHostAddress-family (fixnum) ( fixnum "XHostAddress_family" ))
(defentry set-XHostAddress-family (fixnum fixnum) ( void "set_XHostAddress_family" ))


;;;;;; XImage functions ;;;;;;

(defentry make-XImage () ( fixnum  "make_XImage" ))
;;(defentry XImage-f (fixnum) ( fixnum "XImage_f" ))
;;(defentry set-XImage-f (fixnum fixnum) ( void "set_XImage_f" ))
(defentry XImage-obdata (fixnum) ( fixnum "XImage_obdata" ))
(defentry set-XImage-obdata (fixnum fixnum) ( void "set_XImage_obdata" ))
(defentry XImage-blue_mask (fixnum) ( fixnum "XImage_blue_mask" ))
(defentry set-XImage-blue_mask (fixnum fixnum) ( void "set_XImage_blue_mask" ))
(defentry XImage-green_mask (fixnum) ( fixnum "XImage_green_mask" ))
(defentry set-XImage-green_mask (fixnum fixnum) ( void "set_XImage_green_mask" ))
(defentry XImage-red_mask (fixnum) ( fixnum "XImage_red_mask" ))
(defentry set-XImage-red_mask (fixnum fixnum) ( void "set_XImage_red_mask" ))
(defentry XImage-bits_per_pixel (fixnum) ( fixnum "XImage_bits_per_pixel" ))
(defentry set-XImage-bits_per_pixel (fixnum fixnum) ( void "set_XImage_bits_per_pixel" ))
(defentry XImage-bytes_per_line (fixnum) ( fixnum "XImage_bytes_per_line" ))
(defentry set-XImage-bytes_per_line (fixnum fixnum) ( void "set_XImage_bytes_per_line" ))
(defentry XImage-depth (fixnum) ( fixnum "XImage_depth" ))
(defentry set-XImage-depth (fixnum fixnum) ( void "set_XImage_depth" ))
(defentry XImage-bitmap_pad (fixnum) ( fixnum "XImage_bitmap_pad" ))
(defentry set-XImage-bitmap_pad (fixnum fixnum) ( void "set_XImage_bitmap_pad" ))
(defentry XImage-bitmap_bit_order (fixnum) ( fixnum "XImage_bitmap_bit_order" ))
(defentry set-XImage-bitmap_bit_order (fixnum fixnum) ( void "set_XImage_bitmap_bit_order" ))
(defentry XImage-bitmap_unit (fixnum) ( fixnum "XImage_bitmap_unit" ))
(defentry set-XImage-bitmap_unit (fixnum fixnum) ( void "set_XImage_bitmap_unit" ))
(defentry XImage-byte_order (fixnum) ( fixnum "XImage_byte_order" ))
(defentry set-XImage-byte_order (fixnum fixnum) ( void "set_XImage_byte_order" ))
(defentry XImage-data (fixnum) ( fixnum "XImage_data" ))
(defentry set-XImage-data (fixnum fixnum) ( void "set_XImage_data" ))
(defentry XImage-format (fixnum) ( fixnum "XImage_format" ))
(defentry set-XImage-format (fixnum fixnum) ( void "set_XImage_format" ))
(defentry XImage-xoffset (fixnum) ( fixnum "XImage_xoffset" ))
(defentry set-XImage-xoffset (fixnum fixnum) ( void "set_XImage_xoffset" ))
(defentry XImage-height (fixnum) ( fixnum "XImage_height" ))
(defentry set-XImage-height (fixnum fixnum) ( void "set_XImage_height" ))
(defentry XImage-width (fixnum) ( fixnum "XImage_width" ))
(defentry set-XImage-width (fixnum fixnum) ( void "set_XImage_width" ))


;;;;;; XWindowChanges functions ;;;;;;

(defentry make-XWindowChanges () ( fixnum  "make_XWindowChanges" ))
(defentry XWindowChanges-stack_mode (fixnum) ( fixnum "XWindowChanges_stack_mode" ))
(defentry set-XWindowChanges-stack_mode (fixnum fixnum) ( void "set_XWindowChanges_stack_mode" ))
(defentry XWindowChanges-sibling (fixnum) ( fixnum "XWindowChanges_sibling" ))
(defentry set-XWindowChanges-sibling (fixnum fixnum) ( void "set_XWindowChanges_sibling" ))
(defentry XWindowChanges-border_width (fixnum) ( fixnum "XWindowChanges_border_width" ))
(defentry set-XWindowChanges-border_width (fixnum fixnum) ( void "set_XWindowChanges_border_width" ))
(defentry XWindowChanges-height (fixnum) ( fixnum "XWindowChanges_height" ))
(defentry set-XWindowChanges-height (fixnum fixnum) ( void "set_XWindowChanges_height" ))
(defentry XWindowChanges-width (fixnum) ( fixnum "XWindowChanges_width" ))
(defentry set-XWindowChanges-width (fixnum fixnum) ( void "set_XWindowChanges_width" ))
(defentry XWindowChanges-y (fixnum) ( fixnum "XWindowChanges_y" ))
(defentry set-XWindowChanges-y (fixnum fixnum) ( void "set_XWindowChanges_y" ))
(defentry XWindowChanges-x (fixnum) ( fixnum "XWindowChanges_x" ))
(defentry set-XWindowChanges-x (fixnum fixnum) ( void "set_XWindowChanges_x" ))


;;;;;; XColor functions ;;;;;;

(defentry make-XColor () ( fixnum  "make_XColor" ))
(defentry XColor-pad (fixnum) ( char "XColor_pad" ))
(defentry set-XColor-pad (fixnum char) ( void "set_XColor_pad" ))
(defentry XColor-flags (fixnum) ( char "XColor_flags" ))
(defentry set-XColor-flags (fixnum char) ( void "set_XColor_flags" ))
(defentry XColor-blue (fixnum) ( fixnum "XColor_blue" ))
(defentry set-XColor-blue (fixnum fixnum) ( void "set_XColor_blue" ))
(defentry XColor-green (fixnum) ( fixnum "XColor_green" ))
(defentry set-XColor-green (fixnum fixnum) ( void "set_XColor_green" ))
(defentry XColor-red (fixnum) ( fixnum "XColor_red" ))
(defentry set-XColor-red (fixnum fixnum) ( void "set_XColor_red" ))
(defentry XColor-pixel (fixnum) ( fixnum "XColor_pixel" ))
(defentry set-XColor-pixel (fixnum fixnum) ( void "set_XColor_pixel" ))


;;;;;; XSegment functions ;;;;;;

(defentry make-XSegment () ( fixnum  "make_XSegment" ))
(defentry XSegment-y2 (fixnum) ( fixnum "XSegment_y2" ))
(defentry set-XSegment-y2 (fixnum fixnum) ( void "set_XSegment_y2" ))
(defentry XSegment-x2 (fixnum) ( fixnum "XSegment_x2" ))
(defentry set-XSegment-x2 (fixnum fixnum) ( void "set_XSegment_x2" ))
(defentry XSegment-y1 (fixnum) ( fixnum "XSegment_y1" ))
(defentry set-XSegment-y1 (fixnum fixnum) ( void "set_XSegment_y1" ))
(defentry XSegment-x1 (fixnum) ( fixnum "XSegment_x1" ))
(defentry set-XSegment-x1 (fixnum fixnum) ( void "set_XSegment_x1" ))


;;;;;; XPoint functions ;;;;;;

(defentry make-XPoint () ( fixnum  "make_XPoint" ))
(defentry XPoint-y (fixnum) ( fixnum "XPoint_y" ))
(defentry set-XPoint-y (fixnum fixnum) ( void "set_XPoint_y" ))
(defentry XPoint-x (fixnum) ( fixnum "XPoint_x" ))
(defentry set-XPoint-x (fixnum fixnum) ( void "set_XPoint_x" ))


;;;;;; XRectangle functions ;;;;;;

(defentry make-XRectangle () ( fixnum  "make_XRectangle" ))
(defentry XRectangle-height (fixnum) ( fixnum "XRectangle_height" ))
(defentry set-XRectangle-height (fixnum fixnum) ( void "set_XRectangle_height" ))
(defentry XRectangle-width (fixnum) ( fixnum "XRectangle_width" ))
(defentry set-XRectangle-width (fixnum fixnum) ( void "set_XRectangle_width" ))
(defentry XRectangle-y (fixnum) ( fixnum "XRectangle_y" ))
(defentry set-XRectangle-y (fixnum fixnum) ( void "set_XRectangle_y" ))
(defentry XRectangle-x (fixnum) ( fixnum "XRectangle_x" ))
(defentry set-XRectangle-x (fixnum fixnum) ( void "set_XRectangle_x" ))


;;;;;; XArc functions ;;;;;;

(defentry make-XArc () ( fixnum  "make_XArc" ))
(defentry XArc-angle2 (fixnum) ( fixnum "XArc_angle2" ))
(defentry set-XArc-angle2 (fixnum fixnum) ( void "set_XArc_angle2" ))
(defentry XArc-angle1 (fixnum) ( fixnum "XArc_angle1" ))
(defentry set-XArc-angle1 (fixnum fixnum) ( void "set_XArc_angle1" ))
(defentry XArc-height (fixnum) ( fixnum "XArc_height" ))
(defentry set-XArc-height (fixnum fixnum) ( void "set_XArc_height" ))
(defentry XArc-width (fixnum) ( fixnum "XArc_width" ))
(defentry set-XArc-width (fixnum fixnum) ( void "set_XArc_width" ))
(defentry XArc-y (fixnum) ( fixnum "XArc_y" ))
(defentry set-XArc-y (fixnum fixnum) ( void "set_XArc_y" ))
(defentry XArc-x (fixnum) ( fixnum "XArc_x" ))
(defentry set-XArc-x (fixnum fixnum) ( void "set_XArc_x" ))


;;;;;; XKeyboardControl functions ;;;;;;

(defentry make-XKeyboardControl () ( fixnum  "make_XKeyboardControl" ))
(defentry XKeyboardControl-auto_repeat_mode (fixnum) ( fixnum "XKeyboardControl_auto_repeat_mode" ))
;;(defentry set-XKeyboardControl-auto_repeat_mode (fixnum fixnum) ( void "set_XKeyboardControl_auto_repeat_mode" ))
(defentry XKeyboardControl-key (fixnum) ( fixnum "XKeyboardControl_key" ))
(defentry set-XKeyboardControl-key (fixnum fixnum) ( void "set_XKeyboardControl_key" ))
(defentry XKeyboardControl-led_mode (fixnum) ( fixnum "XKeyboardControl_led_mode" ))
(defentry set-XKeyboardControl-led_mode (fixnum fixnum) ( void "set_XKeyboardControl_led_mode" ))
(defentry XKeyboardControl-led (fixnum) ( fixnum "XKeyboardControl_led" ))
(defentry set-XKeyboardControl-led (fixnum fixnum) ( void "set_XKeyboardControl_led" ))
(defentry XKeyboardControl-bell_duration (fixnum) ( fixnum "XKeyboardControl_bell_duration" ))
(defentry set-XKeyboardControl-bell_duration (fixnum fixnum) ( void "set_XKeyboardControl_bell_duration" ))
(defentry XKeyboardControl-bell_pitch (fixnum) ( fixnum "XKeyboardControl_bell_pitch" ))
(defentry set-XKeyboardControl-bell_pitch (fixnum fixnum) ( void "set_XKeyboardControl_bell_pitch" ))
(defentry XKeyboardControl-bell_percent (fixnum) ( fixnum "XKeyboardControl_bell_percent" ))
(defentry set-XKeyboardControl-bell_percent (fixnum fixnum) ( void "set_XKeyboardControl_bell_percent" ))
(defentry XKeyboardControl-key_click_percent (fixnum) ( fixnum "XKeyboardControl_key_click_percent" ))
(defentry set-XKeyboardControl-key_click_percent (fixnum fixnum) ( void "set_XKeyboardControl_key_click_percent" ))


;;;;;; XKeyboardState functions ;;;;;;

(defentry make-XKeyboardState () ( fixnum  "make_XKeyboardState" ))
(defentry XKeyboardState-auto_repeats (fixnum) ( fixnum "XKeyboardState_auto_repeats" ))
(defentry set-XKeyboardState-auto_repeats (fixnum object) ( void "set_XKeyboardState_auto_repeats" ))
(defentry XKeyboardState-global_auto_repeat (fixnum) ( fixnum "XKeyboardState_global_auto_repeat" ))
(defentry set-XKeyboardState-global_auto_repeat (fixnum fixnum) ( void "set_XKeyboardState_global_auto_repeat" ))
(defentry XKeyboardState-led_mask (fixnum) ( fixnum "XKeyboardState_led_mask" ))
(defentry set-XKeyboardState-led_mask (fixnum fixnum) ( void "set_XKeyboardState_led_mask" ))
(defentry XKeyboardState-bell_duration (fixnum) ( fixnum "XKeyboardState_bell_duration" ))
(defentry set-XKeyboardState-bell_duration (fixnum fixnum) ( void "set_XKeyboardState_bell_duration" ))
(defentry XKeyboardState-bell_pitch (fixnum) ( fixnum "XKeyboardState_bell_pitch" ))
(defentry set-XKeyboardState-bell_pitch (fixnum fixnum) ( void "set_XKeyboardState_bell_pitch" ))
(defentry XKeyboardState-bell_percent (fixnum) ( fixnum "XKeyboardState_bell_percent" ))
(defentry set-XKeyboardState-bell_percent (fixnum fixnum) ( void "set_XKeyboardState_bell_percent" ))
(defentry XKeyboardState-key_click_percent (fixnum) ( fixnum "XKeyboardState_key_click_percent" ))
(defentry set-XKeyboardState-key_click_percent (fixnum fixnum) ( void "set_XKeyboardState_key_click_percent" ))


;;;;;; XTimeCoord functions ;;;;;;

(defentry make-XTimeCoord () ( fixnum  "make_XTimeCoord" ))
(defentry XTimeCoord-y (fixnum) ( fixnum "XTimeCoord_y" ))
(defentry set-XTimeCoord-y (fixnum fixnum) ( void "set_XTimeCoord_y" ))
(defentry XTimeCoord-x (fixnum) ( fixnum "XTimeCoord_x" ))
(defentry set-XTimeCoord-x (fixnum fixnum) ( void "set_XTimeCoord_x" ))
(defentry XTimeCoord-time (fixnum) ( fixnum "XTimeCoord_time" ))
(defentry set-XTimeCoord-time (fixnum fixnum) ( void "set_XTimeCoord_time" ))


;;;;;; XModifierKeymap functions ;;;;;;

(defentry make-XModifierKeymap () ( fixnum  "make_XModifierKeymap" ))
(defentry XModifierKeymap-modifiermap (fixnum) ( fixnum "XModifierKeymap_modifiermap" ))
(defentry set-XModifierKeymap-modifiermap (fixnum fixnum) ( void "set_XModifierKeymap_modifiermap" ))
(defentry XModifierKeymap-max_keypermod (fixnum) ( fixnum "XModifierKeymap_max_keypermod" ))
(defentry set-XModifierKeymap-max_keypermod (fixnum fixnum) ( void "set_XModifierKeymap_max_keypermod" ))
