(in-package :XLIB)
; Xstruct.lsp         Hiep Huu Nguyen                      27 Aug 92

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




;;;;;; _XQEvent functions ;;;;;;

(defentry make-_XQEvent () ( fixnum  "make__XQEvent" ))
(defentry _XQEvent-event (fixnum) ( fixnum "_XQEvent_event" ))
(defentry set-_XQEvent-event (fixnum fixnum) ( void "set__XQEvent_event" ))
(defentry _XQEvent-next (fixnum) ( fixnum "_XQEvent_next" ))
(defentry set-_XQEvent-next (fixnum fixnum) ( void "set__XQEvent_next" ))


;;;;;; XCharStruct functions ;;;;;;

(defentry make-XCharStruct () ( fixnum  "make_XCharStruct" ))
(defentry XCharStruct-attributes (fixnum) ( fixnum "XCharStruct_attributes" ))
(defentry set-XCharStruct-attributes (fixnum fixnum) ( void "set_XCharStruct_attributes" ))
(defentry XCharStruct-descent (fixnum) ( fixnum "XCharStruct_descent" ))
(defentry set-XCharStruct-descent (fixnum fixnum) ( void "set_XCharStruct_descent" ))
(defentry XCharStruct-ascent (fixnum) ( fixnum "XCharStruct_ascent" ))
(defentry set-XCharStruct-ascent (fixnum fixnum) ( void "set_XCharStruct_ascent" ))
(defentry XCharStruct-width (fixnum) ( fixnum "XCharStruct_width" ))
(defentry set-XCharStruct-width (fixnum fixnum) ( void "set_XCharStruct_width" ))
(defentry XCharStruct-rbearing (fixnum) ( fixnum "XCharStruct_rbearing" ))
(defentry set-XCharStruct-rbearing (fixnum fixnum) ( void "set_XCharStruct_rbearing" ))
(defentry XCharStruct-lbearing (fixnum) ( fixnum "XCharStruct_lbearing" ))
(defentry set-XCharStruct-lbearing (fixnum fixnum) ( void "set_XCharStruct_lbearing" ))


;;;;;; XFontProp functions ;;;;;;

(defentry make-XFontProp () ( fixnum  "make_XFontProp" ))
(defentry XFontProp-card32 (fixnum) ( fixnum "XFontProp_card32" ))
(defentry set-XFontProp-card32 (fixnum fixnum) ( void "set_XFontProp_card32" ))
(defentry XFontProp-name (fixnum) ( fixnum "XFontProp_name" ))
(defentry set-XFontProp-name (fixnum fixnum) ( void "set_XFontProp_name" ))


;;;;;; XFontStruct functions ;;;;;;

(defentry make-XFontStruct () ( fixnum  "make_XFontStruct" ))
(defentry XFontStruct-descent (fixnum) ( fixnum "XFontStruct_descent" ))
(defentry set-XFontStruct-descent (fixnum fixnum) ( void "set_XFontStruct_descent" ))
(defentry XFontStruct-ascent (fixnum) ( fixnum "XFontStruct_ascent" ))
(defentry set-XFontStruct-ascent (fixnum fixnum) ( void "set_XFontStruct_ascent" ))
(defentry XFontStruct-per_char (fixnum) ( fixnum "XFontStruct_per_char" ))
(defentry set-XFontStruct-per_char (fixnum fixnum) ( void "set_XFontStruct_per_char" ))
(defentry XFontStruct-max_bounds (fixnum) ( fixnum "XFontStruct_max_bounds" ))
(defentry set-XFontStruct-max_bounds (fixnum fixnum) ( void "set_XFontStruct_max_bounds" ))
(defentry XFontStruct-min_bounds (fixnum) ( fixnum "XFontStruct_min_bounds" ))
(defentry set-XFontStruct-min_bounds (fixnum fixnum) ( void "set_XFontStruct_min_bounds" ))
(defentry XFontStruct-properties (fixnum) ( fixnum "XFontStruct_properties" ))
(defentry set-XFontStruct-properties (fixnum fixnum) ( void "set_XFontStruct_properties" ))
(defentry XFontStruct-n_properties (fixnum) ( fixnum "XFontStruct_n_properties" ))
(defentry set-XFontStruct-n_properties (fixnum fixnum) ( void "set_XFontStruct_n_properties" ))
(defentry XFontStruct-default_char (fixnum) ( fixnum "XFontStruct_default_char" ))
(defentry set-XFontStruct-default_char (fixnum fixnum) ( void "set_XFontStruct_default_char" ))
(defentry XFontStruct-all_chars_exist (fixnum) ( fixnum "XFontStruct_all_chars_exist" ))
(defentry set-XFontStruct-all_chars_exist (fixnum fixnum) ( void "set_XFontStruct_all_chars_exist" ))
(defentry XFontStruct-max_byte1 (fixnum) ( fixnum "XFontStruct_max_byte1" ))
(defentry set-XFontStruct-max_byte1 (fixnum fixnum) ( void "set_XFontStruct_max_byte1" ))
(defentry XFontStruct-min_byte1 (fixnum) ( fixnum "XFontStruct_min_byte1" ))
(defentry set-XFontStruct-min_byte1 (fixnum fixnum) ( void "set_XFontStruct_min_byte1" ))
(defentry XFontStruct-max_char_or_byte2 (fixnum) ( fixnum "XFontStruct_max_char_or_byte2" ))
(defentry set-XFontStruct-max_char_or_byte2 (fixnum fixnum) ( void "set_XFontStruct_max_char_or_byte2" ))
(defentry XFontStruct-min_char_or_byte2 (fixnum) ( fixnum "XFontStruct_min_char_or_byte2" ))
(defentry set-XFontStruct-min_char_or_byte2 (fixnum fixnum) ( void "set_XFontStruct_min_char_or_byte2" ))
(defentry XFontStruct-direction (fixnum) ( fixnum "XFontStruct_direction" ))
(defentry set-XFontStruct-direction (fixnum fixnum) ( void "set_XFontStruct_direction" ))
(defentry XFontStruct-fid (fixnum) ( fixnum "XFontStruct_fid" ))
(defentry set-XFontStruct-fid (fixnum fixnum) ( void "set_XFontStruct_fid" ))
(defentry XFontStruct-ext_data (fixnum) ( fixnum "XFontStruct_ext_data" ))
(defentry set-XFontStruct-ext_data (fixnum fixnum) ( void "set_XFontStruct_ext_data" ))


;;;;;; XTextItem functions ;;;;;;

(defentry make-XTextItem () ( fixnum  "make_XTextItem" ))
(defentry XTextItem-font (fixnum) ( fixnum "XTextItem_font" ))
(defentry set-XTextItem-font (fixnum fixnum) ( void "set_XTextItem_font" ))
(defentry XTextItem-delta (fixnum) ( fixnum "XTextItem_delta" ))
(defentry set-XTextItem-delta (fixnum fixnum) ( void "set_XTextItem_delta" ))
(defentry XTextItem-nchars (fixnum) ( fixnum "XTextItem_nchars" ))
(defentry set-XTextItem-nchars (fixnum fixnum) ( void "set_XTextItem_nchars" ))
(defentry XTextItem-chars (fixnum) ( fixnum "XTextItem_chars" ))
(defentry set-XTextItem-chars (fixnum fixnum) ( void "set_XTextItem_chars" ))


;;;;;; XChar2b functions ;;;;;;

(defentry make-XChar2b () ( fixnum  "make_XChar2b" ))
(defentry XChar2b-byte2 (fixnum) ( char "XChar2b_byte2" ))
(defentry set-XChar2b-byte2 (fixnum char) ( void "set_XChar2b_byte2" ))
(defentry XChar2b-byte1 (fixnum) ( char "XChar2b_byte1" ))
(defentry set-XChar2b-byte1 (fixnum char) ( void "set_XChar2b_byte1" ))


;;;;;; XTextItem16 functions ;;;;;;

(defentry make-XTextItem16 () ( fixnum  "make_XTextItem16" ))
(defentry XTextItem16-font (fixnum) ( fixnum "XTextItem16_font" ))
(defentry set-XTextItem16-font (fixnum fixnum) ( void "set_XTextItem16_font" ))
(defentry XTextItem16-delta (fixnum) ( fixnum "XTextItem16_delta" ))
(defentry set-XTextItem16-delta (fixnum fixnum) ( void "set_XTextItem16_delta" ))
(defentry XTextItem16-nchars (fixnum) ( fixnum "XTextItem16_nchars" ))
(defentry set-XTextItem16-nchars (fixnum fixnum) ( void "set_XTextItem16_nchars" ))
(defentry XTextItem16-chars (fixnum) ( fixnum "XTextItem16_chars" ))
(defentry set-XTextItem16-chars (fixnum fixnum) ( void "set_XTextItem16_chars" ))


;;;;;; XEDataObject functions ;;;;;;

(defentry make-XEDataObject () ( fixnum  "make_XEDataObject" ))
(defentry XEDataObject-font (fixnum) ( fixnum "XEDataObject_font" ))
(defentry set-XEDataObject-font (fixnum fixnum) ( void "set_XEDataObject_font" ))
(defentry XEDataObject-pixmap_format (fixnum) ( fixnum "XEDataObject_pixmap_format" ))
(defentry set-XEDataObject-pixmap_format (fixnum fixnum) ( void "set_XEDataObject_pixmap_format" ))
(defentry XEDataObject-screen (fixnum) ( fixnum "XEDataObject_screen" ))
(defentry set-XEDataObject-screen (fixnum fixnum) ( void "set_XEDataObject_screen" ))
(defentry XEDataObject-visual (fixnum) ( fixnum "XEDataObject_visual" ))
(defentry set-XEDataObject-visual (fixnum fixnum) ( void "set_XEDataObject_visual" ))
(defentry XEDataObject-gc (fixnum) ( fixnum "XEDataObject_gc" ))
(defentry set-XEDataObject-gc (fixnum fixnum) ( void "set_XEDataObject_gc" ))


;;;;;; XSizeHints functions ;;;;;;

(defentry make-XSizeHints () ( fixnum  "make_XSizeHints" ))
(defentry XSizeHints-win_gravity (fixnum) ( fixnum "XSizeHints_win_gravity" ))
(defentry set-XSizeHints-win_gravity (fixnum fixnum) ( void "set_XSizeHints_win_gravity" ))
(defentry XSizeHints-base_height (fixnum) ( fixnum "XSizeHints_base_height" ))
(defentry set-XSizeHints-base_height (fixnum fixnum) ( void "set_XSizeHints_base_height" ))
(defentry XSizeHints-base_width (fixnum) ( fixnum "XSizeHints_base_width" ))
(defentry set-XSizeHints-base_width (fixnum fixnum) ( void "set_XSizeHints_base_width" ))

(defentry XSizeHints-max_aspect_x (fixnum) ( fixnum "XSizeHints_max_aspect_x" ))
(defentry set-XSizeHints-max_aspect_x (fixnum fixnum) ( void "set_XSizeHints_max_aspect_x" ))
(defentry XSizeHints-max_aspect_y (fixnum) ( fixnum "XSizeHints_max_aspect_y" ))
(defentry set-XSizeHints-max_aspect_y (fixnum fixnum) ( void "set_XSizeHints_max_aspect_y" ))
(defentry XSizeHints-min_aspect_x (fixnum) ( fixnum "XSizeHints_min_aspect_x" ))
(defentry set-XSizeHints-min_aspect_x (fixnum fixnum) ( void "set_XSizeHints_min_aspect_x" ))
(defentry XSizeHints-min_aspect_y (fixnum) ( fixnum "XSizeHints_min_aspect_y" ))
(defentry set-XSizeHints-min_aspect_y (fixnum fixnum) ( void "set_XSizeHints_min_aspect_y" ))

(defentry XSizeHints-height_inc (fixnum) ( fixnum "XSizeHints_height_inc" ))
(defentry set-XSizeHints-height_inc (fixnum fixnum) ( void "set_XSizeHints_height_inc" ))
(defentry XSizeHints-width_inc (fixnum) ( fixnum "XSizeHints_width_inc" ))
(defentry set-XSizeHints-width_inc (fixnum fixnum) ( void "set_XSizeHints_width_inc" ))
(defentry XSizeHints-max_height (fixnum) ( fixnum "XSizeHints_max_height" ))
(defentry set-XSizeHints-max_height (fixnum fixnum) ( void "set_XSizeHints_max_height" ))
(defentry XSizeHints-max_width (fixnum) ( fixnum "XSizeHints_max_width" ))
(defentry set-XSizeHints-max_width (fixnum fixnum) ( void "set_XSizeHints_max_width" ))
(defentry XSizeHints-min_height (fixnum) ( fixnum "XSizeHints_min_height" ))
(defentry set-XSizeHints-min_height (fixnum fixnum) ( void "set_XSizeHints_min_height" ))
(defentry XSizeHints-min_width (fixnum) ( fixnum "XSizeHints_min_width" ))
(defentry set-XSizeHints-min_width (fixnum fixnum) ( void "set_XSizeHints_min_width" ))
(defentry XSizeHints-height (fixnum) ( fixnum "XSizeHints_height" ))
(defentry set-XSizeHints-height (fixnum fixnum) ( void "set_XSizeHints_height" ))
(defentry XSizeHints-width (fixnum) ( fixnum "XSizeHints_width" ))
(defentry set-XSizeHints-width (fixnum fixnum) ( void "set_XSizeHints_width" ))
(defentry XSizeHints-y (fixnum) ( fixnum "XSizeHints_y" ))
(defentry set-XSizeHints-y (fixnum fixnum) ( void "set_XSizeHints_y" ))
(defentry XSizeHints-x (fixnum) ( fixnum "XSizeHints_x" ))
(defentry set-XSizeHints-x (fixnum fixnum) ( void "set_XSizeHints_x" ))
(defentry XSizeHints-flags (fixnum) ( fixnum "XSizeHints_flags" ))
(defentry set-XSizeHints-flags (fixnum fixnum) ( void "set_XSizeHints_flags" ))


;;;;;; XWMHints functions ;;;;;;

(defentry make-XWMHints () ( fixnum  "make_XWMHints" ))
(defentry XWMHints-window_group (fixnum) ( fixnum "XWMHints_window_group" ))
(defentry set-XWMHints-window_group (fixnum fixnum) ( void "set_XWMHints_window_group" ))
(defentry XWMHints-icon_mask (fixnum) ( fixnum "XWMHints_icon_mask" ))
(defentry set-XWMHints-icon_mask (fixnum fixnum) ( void "set_XWMHints_icon_mask" ))
(defentry XWMHints-icon_y (fixnum) ( fixnum "XWMHints_icon_y" ))
(defentry set-XWMHints-icon_y (fixnum fixnum) ( void "set_XWMHints_icon_y" ))
(defentry XWMHints-icon_x (fixnum) ( fixnum "XWMHints_icon_x" ))
(defentry set-XWMHints-icon_x (fixnum fixnum) ( void "set_XWMHints_icon_x" ))
(defentry XWMHints-icon_window (fixnum) ( fixnum "XWMHints_icon_window" ))
(defentry set-XWMHints-icon_window (fixnum fixnum) ( void "set_XWMHints_icon_window" ))
(defentry XWMHints-icon_pixmap (fixnum) ( fixnum "XWMHints_icon_pixmap" ))
(defentry set-XWMHints-icon_pixmap (fixnum fixnum) ( void "set_XWMHints_icon_pixmap" ))
(defentry XWMHints-initial_state (fixnum) ( fixnum "XWMHints_initial_state" ))
(defentry set-XWMHints-initial_state (fixnum fixnum) ( void "set_XWMHints_initial_state" ))
(defentry XWMHints-input (fixnum) ( fixnum "XWMHints_input" ))
(defentry set-XWMHints-input (fixnum fixnum) ( void "set_XWMHints_input" ))
(defentry XWMHints-flags (fixnum) ( fixnum "XWMHints_flags" ))
(defentry set-XWMHints-flags (fixnum fixnum) ( void "set_XWMHints_flags" ))


;;;;;; XTextProperty functions ;;;;;;

(defentry make-XTextProperty () ( fixnum  "make_XTextProperty" ))
(defentry XTextProperty-nitems (fixnum) ( fixnum "XTextProperty_nitems" ))
(defentry set-XTextProperty-nitems (fixnum fixnum) ( void "set_XTextProperty_nitems" ))
(defentry XTextProperty-format (fixnum) ( fixnum "XTextProperty_format" ))
(defentry set-XTextProperty-format (fixnum fixnum) ( void "set_XTextProperty_format" ))
(defentry XTextProperty-encoding (fixnum) ( fixnum "XTextProperty_encoding" ))
(defentry set-XTextProperty-encoding (fixnum fixnum) ( void "set_XTextProperty_encoding" ))
(defentry XTextProperty-value (fixnum) ( fixnum "XTextProperty_value" ))
(defentry set-XTextProperty-value (fixnum fixnum) ( void "set_XTextProperty_value" ))


;;;;;; XIconSize functions ;;;;;;

(defentry make-XIconSize () ( fixnum  "make_XIconSize" ))
(defentry XIconSize-height_inc (fixnum) ( fixnum "XIconSize_height_inc" ))
(defentry set-XIconSize-height_inc (fixnum fixnum) ( void "set_XIconSize_height_inc" ))
(defentry XIconSize-width_inc (fixnum) ( fixnum "XIconSize_width_inc" ))
(defentry set-XIconSize-width_inc (fixnum fixnum) ( void "set_XIconSize_width_inc" ))
(defentry XIconSize-max_height (fixnum) ( fixnum "XIconSize_max_height" ))
(defentry set-XIconSize-max_height (fixnum fixnum) ( void "set_XIconSize_max_height" ))
(defentry XIconSize-max_width (fixnum) ( fixnum "XIconSize_max_width" ))
(defentry set-XIconSize-max_width (fixnum fixnum) ( void "set_XIconSize_max_width" ))
(defentry XIconSize-min_height (fixnum) ( fixnum "XIconSize_min_height" ))
(defentry set-XIconSize-min_height (fixnum fixnum) ( void "set_XIconSize_min_height" ))
(defentry XIconSize-min_width (fixnum) ( fixnum "XIconSize_min_width" ))
(defentry set-XIconSize-min_width (fixnum fixnum) ( void "set_XIconSize_min_width" ))


;;;;;; XClassHint functions ;;;;;;

(defentry make-XClassHint () ( fixnum  "make_XClassHint" ))
(defentry XClassHint-res_class (fixnum) ( fixnum "XClassHint_res_class" ))
(defentry set-XClassHint-res_class (fixnum fixnum) ( void "set_XClassHint_res_class" ))
(defentry XClassHint-res_name (fixnum) ( fixnum "XClassHint_res_name" ))
(defentry set-XClassHint-res_name (fixnum fixnum) ( void "set_XClassHint_res_name" ))


;;;;;; XComposeStatus functions ;;;;;;

(defentry make-XComposeStatus () ( fixnum  "make_XComposeStatus" ))
(defentry XComposeStatus-chars_matched (fixnum) ( fixnum "XComposeStatus_chars_matched" ))
(defentry set-XComposeStatus-chars_matched (fixnum fixnum) ( void "set_XComposeStatus_chars_matched" ))
(defentry XComposeStatus-compose_ptr (fixnum) ( fixnum "XComposeStatus_compose_ptr" ))
(defentry set-XComposeStatus-compose_ptr (fixnum fixnum) ( void "set_XComposeStatus_compose_ptr" ))


;;;;;; XVisualInfo functions ;;;;;;

(defentry make-XVisualInfo () ( fixnum  "make_XVisualInfo" ))
(defentry XVisualInfo-bits_per_rgb (fixnum) ( fixnum "XVisualInfo_bits_per_rgb" ))
(defentry set-XVisualInfo-bits_per_rgb (fixnum fixnum) ( void "set_XVisualInfo_bits_per_rgb" ))
(defentry XVisualInfo-colormap_size (fixnum) ( fixnum "XVisualInfo_colormap_size" ))
(defentry set-XVisualInfo-colormap_size (fixnum fixnum) ( void "set_XVisualInfo_colormap_size" ))
(defentry XVisualInfo-blue_mask (fixnum) ( fixnum "XVisualInfo_blue_mask" ))
(defentry set-XVisualInfo-blue_mask (fixnum fixnum) ( void "set_XVisualInfo_blue_mask" ))
(defentry XVisualInfo-green_mask (fixnum) ( fixnum "XVisualInfo_green_mask" ))
(defentry set-XVisualInfo-green_mask (fixnum fixnum) ( void "set_XVisualInfo_green_mask" ))
(defentry XVisualInfo-red_mask (fixnum) ( fixnum "XVisualInfo_red_mask" ))
(defentry set-XVisualInfo-red_mask (fixnum fixnum) ( void "set_XVisualInfo_red_mask" ))
(defentry XVisualInfo-class (fixnum) ( fixnum "XVisualInfo_class" ))
(defentry set-XVisualInfo-class (fixnum fixnum) ( void "set_XVisualInfo_class" ))
(defentry XVisualInfo-depth (fixnum) ( fixnum "XVisualInfo_depth" ))
(defentry set-XVisualInfo-depth (fixnum fixnum) ( void "set_XVisualInfo_depth" ))
(defentry XVisualInfo-screen (fixnum) ( fixnum "XVisualInfo_screen" ))
(defentry set-XVisualInfo-screen (fixnum fixnum) ( void "set_XVisualInfo_screen" ))
(defentry XVisualInfo-visualid (fixnum) ( fixnum "XVisualInfo_visualid" ))
(defentry set-XVisualInfo-visualid (fixnum fixnum) ( void "set_XVisualInfo_visualid" ))
(defentry XVisualInfo-visual (fixnum) ( fixnum "XVisualInfo_visual" ))
(defentry set-XVisualInfo-visual (fixnum fixnum) ( void "set_XVisualInfo_visual" ))


;;;;;; XStandardColormap functions ;;;;;;

(defentry make-XStandardColormap () ( fixnum  "make_XStandardColormap" ))
(defentry XStandardColormap-killid (fixnum) ( fixnum "XStandardColormap_killid" ))
(defentry set-XStandardColormap-killid (fixnum fixnum) ( void "set_XStandardColormap_killid" ))
(defentry XStandardColormap-visualid (fixnum) ( fixnum "XStandardColormap_visualid" ))
(defentry set-XStandardColormap-visualid (fixnum fixnum) ( void "set_XStandardColormap_visualid" ))
(defentry XStandardColormap-base_pixel (fixnum) ( fixnum "XStandardColormap_base_pixel" ))
(defentry set-XStandardColormap-base_pixel (fixnum fixnum) ( void "set_XStandardColormap_base_pixel" ))
(defentry XStandardColormap-blue_mult (fixnum) ( fixnum "XStandardColormap_blue_mult" ))
(defentry set-XStandardColormap-blue_mult (fixnum fixnum) ( void "set_XStandardColormap_blue_mult" ))
(defentry XStandardColormap-blue_max (fixnum) ( fixnum "XStandardColormap_blue_max" ))
(defentry set-XStandardColormap-blue_max (fixnum fixnum) ( void "set_XStandardColormap_blue_max" ))
(defentry XStandardColormap-green_mult (fixnum) ( fixnum "XStandardColormap_green_mult" ))
(defentry set-XStandardColormap-green_mult (fixnum fixnum) ( void "set_XStandardColormap_green_mult" ))
(defentry XStandardColormap-green_max (fixnum) ( fixnum "XStandardColormap_green_max" ))
(defentry set-XStandardColormap-green_max (fixnum fixnum) ( void "set_XStandardColormap_green_max" ))
(defentry XStandardColormap-red_mult (fixnum) ( fixnum "XStandardColormap_red_mult" ))
(defentry set-XStandardColormap-red_mult (fixnum fixnum) ( void "set_XStandardColormap_red_mult" ))
(defentry XStandardColormap-red_max (fixnum) ( fixnum "XStandardColormap_red_max" ))
(defentry set-XStandardColormap-red_max (fixnum fixnum) ( void "set_XStandardColormap_red_max" ))
(defentry XStandardColormap-colormap (fixnum) ( fixnum "XStandardColormap_colormap" ))
(defentry set-XStandardColormap-colormap (fixnum fixnum) ( void "set_XStandardColormap_colormap" ))
