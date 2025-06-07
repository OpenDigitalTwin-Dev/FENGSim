(in-package :XLIB)
; XAtom.lsp        modified by Hiep Huu Nguyen                      27 Aug 92

; Copyright (c) 1994 Hiep Huu Nguyen and The University of Texas at Austin.

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



;; THIS IS A GENERATED FILE
 ;;
 ;; Do not change!  Changing this file implies a protocol change!
 

(defconstant  XA_PRIMARY  1)
(defconstant  XA_SECONDARY  2)
(defconstant  XA_ARC  3) 
(defconstant  XA_ATOM  4)
(defconstant  XA_BITMAP  5)
(defconstant  XA_CARDINAL  6)
(defconstant  XA_COLORMAP  7) 
(defconstant  XA_CURSOR  8) 
(defconstant  XA_CUT_BUFFER0  9)
(defconstant  XA_CUT_BUFFER1  10)
(defconstant  XA_CUT_BUFFER2  11) 
(defconstant  XA_CUT_BUFFER3  12) 
(defconstant  XA_CUT_BUFFER4  13) 
(defconstant  XA_CUT_BUFFER5  14) 
(defconstant  XA_CUT_BUFFER6  15)
(defconstant  XA_CUT_BUFFER7  16)
(defconstant  XA_DRAWABLE  17)
(defconstant  XA_FONT  18)
(defconstant  XA_INTEGER  19)
(defconstant  XA_PIXMAP  20)
(defconstant  XA_POINT  21)
(defconstant  XA_RECTANGLE  22)
(defconstant  XA_RESOURCE_MANAGER  23)
(defconstant  XA_RGB_COLOR_MAP  24)
(defconstant  XA_RGB_BEST_MAP  25)
(defconstant  XA_RGB_BLUE_MAP  26)
(defconstant  XA_RGB_DEFAULT_MAP  27)
(defconstant  XA_RGB_GRAY_MAP  28)
(defconstant  XA_RGB_GREEN_MAP  29)
(defconstant  XA_RGB_RED_MAP  30)
(defconstant  XA_STRING  31)
(defconstant  XA_VISUALID  32)
(defconstant  XA_WINDOW  33)
(defconstant  XA_WM_COMMAND  34)
(defconstant  XA_WM_HINTS  35)
(defconstant  XA_WM_CLIENT_MACHINE  36)
(defconstant  XA_WM_ICON_NAME  37)
(defconstant  XA_WM_ICON_SIZE  38)
(defconstant  XA_WM_NAME  39)
(defconstant  XA_WM_NORMAL_HINTS  40)
(defconstant  XA_WM_SIZE_HINTS  41)
(defconstant  XA_WM_ZOOM_HINTS  42)
(defconstant  XA_MIN_SPACE  43)
(defconstant  XA_NORM_SPACE  44)
(defconstant  XA_MAX_SPACE  45)
















(defconstant  XA_END_SPACE  46)
(defconstant  XA_SUPERSCRIPT_X  47)
(defconstant  XA_SUPERSCRIPT_Y  48)
(defconstant  XA_SUBSCRIPT_X  49)
(defconstant  XA_SUBSCRIPT_Y  50)
(defconstant  XA_UNDERLINE_POSITION  51)
(defconstant  XA_UNDERLINE_THICKNESS  52)
(defconstant  XA_STRIKEOUT_ASCENT  53)
(defconstant  XA_STRIKEOUT_DESCENT  54)
(defconstant  XA_ITALIC_ANGLE  55)
(defconstant  XA_X_HEIGHT  56)
(defconstant  XA_QUAD_WIDTH  57)
(defconstant  XA_WEIGHT  58)
(defconstant  XA_POINT_SIZE  59)
(defconstant  XA_RESOLUTION  60)
(defconstant  XA_COPYRIGHT  61)
(defconstant XA_NOTICE  62)
(defconstant XA_FONT_NAME  63)
(defconstant XA_FAMILY_NAME  64)
(defconstant XA_FULL_NAME  65)
(defconstant XA_CAP_HEIGHT  66)
(defconstant XA_WM_CLASS  67)
(defconstant XA_WM_TRANSIENT_FOR  68)

(defconstant XA_LAST_PREDEFINED  68)

