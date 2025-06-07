(in-package :XLIB)
; Xutil.lsp      modified by Hiep Huu Nguyen                    27 Aug 92

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

;; $XConsortium: Xutil.h,v 11.58 89/12/12 20:15:40 jim Exp $ */

;;**********************************************************
;;Copyright 1987 by Digital Equipment Corporation, Maynard, Massachusetts,
;;and the Massachusetts Institute of Technology, Cambridge, Massachusetts.

;;modified by Hiep H Nguyen 28 Jul 91

;;                        All Rights Reserved

;;Permission to use, copy, modify, and distribute this software and its 
;;documentation for any purpose and without fee is hereby granted, 
;;provided that the above copyright notice appear in all copies and that
;;both that copyright notice and this permission notice appear in 
;;supporting documentation, and that the names of Digital or MIT not be
;;used in advertising or publicity pertaining to distribution of the
;;software without specific, written prior permission.  

;;DIGITAL DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
;;ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL
;;DIGITAL BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
;;ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
;;WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
;;ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
;;SOFTWARE.

;;*****************************************************************

;; 
;; * Bitmask returned by XParseGeometry().  Each bit tells if the corresponding)
;; * value (x, y, width, height) was found in the parsed string.)
 
(defconstant NoValue		0000)
(defconstant XValue  	0001)
(defconstant YValue		0002)
(defconstant WidthValue  	0004)
(defconstant HeightValue  	0008)
(defconstant AllValues 	15)
(defconstant XNegative 	16)
(defconstant YNegative 	32)

;;
 ;; The next block of definitions are for window manager properties that
 ;; clients and applications use for communication.
 

;; flags argument in size hints 
(defconstant USPosition	(expt 2  0) ) ;; user specified x, y 
(defconstant USSize		(expt 2  1) ) ;; user specified width, height 

(defconstant PPosition	(expt 2  2) ) ;; program specified position 
(defconstant PSize		(expt 2  3) ) ;; program specified size 
(defconstant PMinSize	(expt 2  4) ) ;; program specified minimum size 
(defconstant PMaxSize	(expt 2  5) ) ;; program specified maximum size 
(defconstant PResizeInc	(expt 2  6) ) ;; program specified resize increments 
(defconstant PAspect		(expt 2  7) ) ;; program specified min and max aspect ratios 
(defconstant PBaseSize	(expt 2  8) ) ;; program specified base for incrementing 
(defconstant PWinGravity	(expt 2  9) ) ;; program specified window gravity 

;; obsolete 
(defconstant PAllHints (+ PPosition PSize PMinSize PMaxSize PResizeInc PAspect))

;; definition for flags of XWMHints 

(defconstant InputHint 		(expt 2  0))
(defconstant StateHint 		(expt 2  1))
(defconstant IconPixmapHint		(expt 2  2))
(defconstant IconWindowHint		(expt 2  3))
(defconstant IconPositionHint 	(expt 2  4))
(defconstant IconMaskHint		(expt 2  5))
(defconstant WindowGroupHint		(expt 2  6))
(defconstant AllHints ( + InputHint StateHint IconPixmapHint IconWindowHint 
IconPositionHint IconMaskHint WindowGroupHint))

;; definitions for initial window state 
(defconstant WithdrawnState 0	) ;; for windows that are not mapped 
(defconstant NormalState 1	) ;; most applications want to start this way 
(defconstant IconicState 3	) ;; application wants to start as an icon 

;;
 ;; Obsolete states no longer defined by ICCCM
 
(defconstant DontCareState 0	) ;; don't know or care 
(defconstant ZoomState 2	) ;; application wants to start zoomed 
(defconstant InactiveState 4	) ;; application believes it is seldom used; 
			 ;; some wm's may put it on inactive menu 


 
;;
 ;; opaque reference to Region data type 
 
;;typedef struct _XRegion *Region; 

;; Return values from XRectInRegion() 
 
(defconstant RectangleOut 0)
(defconstant RectangleIn  1)
(defconstant RectanglePart 2)
 

(defconstant VisualNoMask		0)
(defconstant VisualIDMask 		1)
(defconstant VisualScreenMask	2)
(defconstant VisualDepthMask		4)
(defconstant VisualClassMask		8)
(defconstant VisualRedMaskMask	16)
(defconstant VisualGreenMaskMask	32)
(defconstant VisualBlueMaskMask	64)
(defconstant VisualColormapSizeMask	128)
(defconstant VisualBitsPerRGBMask	256)
(defconstant VisualAllMask		511)

(defconstant ReleaseByFreeingColormap 1) ;; for killid field above 


;;
;; return codes for XReadBitmapFile and XWriteBitmapFile
 
(defconstant BitmapSuccess		0)
(defconstant BitmapOpenFailed 	1)
(defconstant BitmapFileInvalid 	2)
(defconstant BitmapNoMemory		3)
;;
 ;; Declare the routines that don't return int.
 

;; ***************************************************************
;; *
;; * Context Management
;; *
;; ***************************************************************


;; Associative lookup table return codes 

(defconstant XCSUCCESS 0	) ;; No error. 
(defconstant XCNOMEM   1    ) ;; Out of memory 
(defconstant XCNOENT   2    ) ;; No entry in table 

;;typedef fixnum XContext;

(defentry XSaveContext(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum		;; context 
     fixnum 	;; data 

)( fixnum "XSaveContext"))



(defentry XFindContext(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum		;; context 
    fixnum 	;; data_return 

)( fixnum "XFindContext"))



(defentry XDeleteContext(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum		;; context 

)( fixnum "XDeleteContext"))




(defentry  XGetWMHints(

    fixnum 	;; display 
    fixnum		;; w 		      

)( fixnum  "XGetWMHints"))


(defentry XCreateRegion(

;;    void

)( fixnum "XCreateRegion"))


(defentry XPolygonRegion(

    fixnum 	;; points 
    fixnum			;; n 
    fixnum			;; fill_rule 

)( fixnum "XPolygonRegion"))



(defentry  XGetVisualInfo(

    fixnum 	;; display 
    fixnum		;; vinfo_mask 
    fixnum ;; vinfo_template 
    fixnum 	;; nitems_return 

)( fixnum  "XGetVisualInfo"))

;; Allocation routines for properties that may get longer 


(defentry  XAllocSizeHints (

;;    void

)( fixnum  "XAllocSizeHints" ))


(defentry  XAllocStandardColormap (

;;    void

)( fixnum  "XAllocStandardColormap" ))


(defentry  XAllocWMHints (

;;    void

)( fixnum  "XAllocWMHints" ))


(defentry  XAllocClassHint (

;;    void

)( fixnum  "XAllocClassHint" ))


(defentry  XAllocIconSize (

;;    void

)( fixnum  "XAllocIconSize" ))

;; ICCCM routines for data structures defined in this file 


(defentry XGetWMSizeHints(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum 	;; hints_return 
    fixnum 	;; supplied_return 
    fixnum		;; property 

)( fixnum "XGetWMSizeHints"))


(defentry XGetWMNormalHints(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum 	;; hints_return 
    fixnum 	;; supplied_return  

)( fixnum "XGetWMNormalHints"))


(defentry XGetRGBColormaps(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum  ;; stdcmap_return 
    fixnum 	;; count_return 
    fixnum		;; property 

)( fixnum "XGetRGBColormaps"))


(defentry XGetTextProperty(

    fixnum 	;; display 
    fixnum		;; window 
    fixnum ;; text_prop_return 
    fixnum		;; property 

)( fixnum "XGetTextProperty"))


(defentry XGetWMName(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum ;; text_prop_return 

)( fixnum "XGetWMName"))


(defentry XGetWMIconName(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum ;; text_prop_return 

)( fixnum "XGetWMIconName"))


(defentry XGetWMClientMachine(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum ;; text_prop_return 

)( fixnum "XGetWMClientMachine"))


(defentry XSetWMProperties(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum ;; window_name 
    fixnum ;; icon_name 
    fixnum 	;; argv 
    fixnum			;; argc 
    fixnum 	;; normal_hints 
    fixnum 	;; wm_hints 
    fixnum 	;; class_hints 

)( void "XSetWMProperties"))


(defentry XSetWMSizeHints(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum 	;; hints 
    fixnum		;; property 

)( void "XSetWMSizeHints"))


(defentry XSetWMNormalHints(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum 	;; hints 

)( void "XSetWMNormalHints"))


(defentry XSetRGBColormaps(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum ;; stdcmaps 
    fixnum			;; count 
    fixnum		;; property 

)( void "XSetRGBColormaps"))


(defentry XSetTextProperty(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum ;; text_prop 
    fixnum		;; property 

)( void "XSetTextProperty"))


(defentry XSetWMName(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum ;; text_prop 

)( void "XSetWMName"))


(defentry XSetWMIconName(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum ;; text_prop 

)( void "XSetWMIconName"))


(defentry XSetWMClientMachine(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum ;; text_prop 

)( void "XSetWMClientMachine"))


(defentry XStringListToTextProperty(

    fixnum 	;; list 
    fixnum			;; count 
    fixnum ;; text_prop_return 

)( fixnum "XStringListToTextProperty"))


(defentry XTextPropertyToStringList(

    fixnum ;; text_prop 
    fixnum 		;; list_return 
    fixnum 	;; count_return 

)( fixnum "XTextPropertyToStringList"))

;; The following declarations are alphabetized. 



(defentry XClipBox(

    fixnum		;; r 
    fixnum 	;; rect_return 

)( void "XClipBox"))



(defentry XDestroyRegion(

    fixnum		;; r 

)( void "XDestroyRegion"))



(defentry XEmptyRegion(

    fixnum		;; r 

)( void "XEmptyRegion"))



(defentry XEqualRegion(

    fixnum		;; r1 
    fixnum		;; r2 

)( void "XEqualRegion"))



(defentry XGetClassHint(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum 	;; class_hints_return 

)( fixnum "XGetClassHint"))



(defentry XGetIconSizes(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum 		;; size_list_return 
    fixnum 	;; count_return 

)( fixnum "XGetIconSizes"))



(defentry XGetNormalHints(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum 	;; hints_return 

)( fixnum "XGetNormalHints"))



(defentry XGetSizeHints(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum 	;; hints_return 
    fixnum		;; property 

)( fixnum "XGetSizeHints"))



(defentry XGetStandardColormap(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum ;; colormap_return 
    fixnum		;; property 			    

)( fixnum "XGetStandardColormap"))



(defentry XGetZoomHints(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum 	;; zhints_return 

)( fixnum "XGetZoomHints"))



(defentry XIntersectRegion(

    fixnum		;; sra 
    fixnum		;; srb 
    fixnum		;; dr_return 

)( void "XIntersectRegion"))



(defentry XLookupString(

    fixnum 	;; event_struct 
    object		;; buffer_return 
    fixnum			;; bytes_buffer 
    fixnum 	;; keysym_return 
    fixnum ;; int_in_out 

)( fixnum "XLookupString"))



(defentry XMatchVisualInfo(

    fixnum 	;; display 
    fixnum			;; screen 
    fixnum			;; depth 
    fixnum			;; class 
    fixnum ;; vinfo_return 

)( fixnum "XMatchVisualInfo"))



(defentry XOffsetRegion(

    fixnum		;; r 
    fixnum			;; dx 
    fixnum			;; dy 

)( void "XOffsetRegion"))



(defentry XPointInRegion(

    fixnum		;; r 
    fixnum			;; x 
    fixnum			;; y 

)( fixnum "XPointInRegion"))


(defentry XRectInRegion(

    fixnum		;; r 
    fixnum			;; x 
    fixnum			;; y 
     fixnum	;; width 
     fixnum	;; height 

)( fixnum "XRectInRegion"))



(defentry XSetClassHint(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum 	;; class_hints 

)( void "XSetClassHint"))



(defentry XSetIconSizes(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum 	;; size_list 
    fixnum			;; count     

)( void "XSetIconSizes"))



(defentry XSetNormalHints(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum 	;; hints 

)( void "XSetNormalHints"))



(defentry XSetSizeHints(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum 	;; hints 
    fixnum		;; property 

)( void "XSetSizeHints"))



(defentry XSetStandardProperties(

    fixnum 	;; display 
    fixnum		;; w 
     object		;; window_name 
     object		;; icon_name 
    fixnum		;; icon_pixmap 
    fixnum 	;; argv 
    fixnum			;; argc 
    fixnum 	;; hints 

)( void "XSetStandardProperties"))



(defentry XSetWMHints(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum 	;; wm_hints 

)( void "XSetWMHints"))



(defentry XSetRegion(

    fixnum 	;; display 
    fixnum			;; gc 
    fixnum		;; r 

)( void "XSetRegion"))



(defentry XSetStandardColormap(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum ;; colormap 
    fixnum		;; property 

)( void "XSetStandardColormap"))



(defentry XSetZoomHints(

    fixnum 	;; display 
    fixnum		;; w 
    fixnum 	;; zhints 

)( void "XSetZoomHints"))



(defentry XShrinkRegion(

    fixnum		;; r 
    fixnum			;; dx 
    fixnum			;; dy 

)( void "XShrinkRegion"))



(defentry XSubtractRegion(

    fixnum		;; sra 
    fixnum		;; srb 
    fixnum		;; dr_return 

)( void "XSubtractRegion"))



(defentry XUnionRectWithRegion(

    fixnum 	;; rectangle 
    fixnum		;; src_region 
    fixnum		;; dest_region_return 

)( void "XUnionRectWithRegion"))



(defentry XUnionRegion(

    fixnum		;; sra 
    fixnum		;; srb 
    fixnum		;; dr_return 

)( void "XUnionRegion"))



(defentry XWMGeometry(

    fixnum 	;; display 
    fixnum			;; screen_number 
     object		;; user_geometry 
     object		;; default_geometry 
     fixnum	;; border_width 
    fixnum 	;; hints 
    fixnum 	;; x_return 
    fixnum 	;; y_return 
    fixnum 	;; width_return 
    fixnum 	;; height_return 
    fixnum 	;; gravity_return 

)( fixnum "XWMGeometry"))



(defentry XXorRegion(

    fixnum		;; sra 
    fixnum		;; srb 
    fixnum		;; dr_return 

)( void "XXorRegion"))
;;
 ;; These macros are used to give some sugar to the image routines so that
 ;; naive people are more comfortable with them.
 
(defentry XDestroyImage(fixnum) (fixnum "XDestroyImage"))
(defentry XGetPixel(fixnum fixnum fixnum) (fixnum "XGetPixel" ))
(defentry XPutPixel(fixnum fixnum int fixnum) ( fixnum "XPutPixel"))
(defentry XSubImage(fixnum  fixnum int fixnum fixnum) (fixnum  "XSubImage"))
(defentry XAddPixel(fixnum  fixnum) (fixnum  "XAddPixel"))
;;
 ;; Keysym macros, used on Keysyms to test for classes of symbols

(defentry IsKeypadKey(fixnum)  (fixnum "IsKeypadKey"))

(defentry IsCursorKey(fixnum) (fixnum "IsCursorKey"))

(defentry IsPFKey(fixnum) (fixnum "IsPFKey"))

(defentry  IsFunctionKey(fixnum) (fixnum "IsFunctionKey"))

(defentry IsMiscFunctionKey(fixnum)  (fixnum "IsMiscFunctionKey"))

(defentry IsModifierKey(fixnum) (fixnum "IsModifierKey"))
(defentry XUniqueContext() (fixnum  "XUniqueContext"))
(defentry  XStringToContext(object) (fixnum  "XStringToContext"))

