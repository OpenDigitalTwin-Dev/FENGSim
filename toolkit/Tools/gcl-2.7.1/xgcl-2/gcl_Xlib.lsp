(in-package :XLIB)
; Xlib.lsp         Hiep Huu Nguyen                      27 Aug 92

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

;;typedef unsigned long XID) ;

;;typedef XID Window) ;
;;typedef XID Drawable) ;
;;typedef XID Font) ;
;;typedef XID Pixmap) ;
;;typedef XID Cursor) ;
;;typedef XID Colormap) ;
;;typedef XID GContext) ;
;;typedef XID KeySym) ;

;;typedef unsigned long Mask) ;

;;typedef unsigned long Atom) ;

;;typedef unsigned long VisualID) ;

;;typedef unsigned long Time) ;

;;typedef unsigned char KeyCode) ;

(defconstant  True 1)
(defconstant  False 0)

(defconstant  QueuedAlready 0)
(defconstant  QueuedAfterReading 1)
(defconstant  QueuedAfterFlush 2)

(defentry XLoadQueryFont(

    fixnum		;; display 
    object		;; name 

)( fixnum  "XLoadQueryFont"))



(defentry  XQueryFont(

    fixnum		;; display 
    fixnum			;; font_ID 

)( fixnum  "XQueryFont"))




(defentry  XGetMotionEvents(

    fixnum		;; display 
    fixnum		;; w 
    fixnum		;; start 
    fixnum		;; stop 
    fixnum		;; nevents_return 

)( fixnum  "XGetMotionEvents"))



(defentry  XDeleteModifiermapEntry(

    fixnum	;; modmap 

    fixnum		;; keycode_entry 

    fixnum			;; modifier 

)( fixnum  "XDeleteModifiermapEntry"))



(defentry  XGetModifierMapping(

    fixnum		;; display 

)( 	fixnum "XGetModifierMapping"))



(defentry  XInsertModifiermapEntry(

    fixnum	;; modmap 

    fixnum		;; keycode_entry 

    fixnum			;; modifier     

)( 	fixnum  "XInsertModifiermapEntry"))



(defentry  XNewModifiermap(

    fixnum			;; max_keys_per_mod 

)( fixnum  "XNewModifiermap"))



(defentry  XCreateImage(

    fixnum		;; display 
    fixnum		;; visual 
     fixnum	;; depth 
    fixnum			;; format 
    fixnum			;; offset 
   object		;; data 
     fixnum	;; width 
     fixnum	;; height 
    fixnum			;; bitmap_pad 
    fixnum			;; bytes_per_line 

)( fixnum  "XCreateImage"))


(defentry  XGetImage(

    fixnum		;; display 
    fixnum		;; d 
    fixnum			;; x 
    fixnum			;; y 
     fixnum	;; width 
     fixnum	;; height 
     fixnum	;; plane_mask 
    fixnum			;; format 

)( fixnum  "XGetImage"))


(defentry  XGetSubImage(

    fixnum		;; display 
    fixnum		;; d 
    fixnum			;; x 
    fixnum			;; y 
     fixnum	;; width 
     fixnum	;; height 
     fixnum	;; plane_mask 
    fixnum			;; format 
    fixnum 	;; dest_image 
    fixnum			;; dest_x 
    fixnum			;; dest_y 

)( fixnum "XGetSubImage"))

;;Window  X function declarations.
 


(defentry  XOpenDisplay(

    object		;; display_name 

)( fixnum  "XOpenDisplay"))



(defentry XrmInitialize(

;;    void

)( void "XrmInitialize"))



(defentry  XFetchBytes(

    fixnum		;; display 
    fixnum		;; nbytes_return 

)( fixnum  "XFetchBytes"))


(defentry  XFetchBuffer(

    fixnum		;; display 
    fixnum		;; nbytes_return 
    fixnum			;; buffer 

)( fixnum  "XFetchBuffer"))


(defentry  XGetAtomName(

    fixnum		;; display 
    fixnum		;; atom 

)( fixnum  "XGetAtomName"))


(defentry  XGetDefault(

    fixnum		;; display 
    object		;; program 
    object		;; option 		  

)( fixnum  "XGetDefault"))


(defentry  XDisplayName(

    object		;; string 

)( fixnum  "XDisplayName"))


(defentry  XKeysymToString(

    fixnum		;; keysym 

)( fixnum  "XKeysymToString"))




(defentry XInternAtom(

    fixnum		;; display 
    object		;; atom_name 
    fixnum		;; only_if_exists 		 

)( fixnum "XInternAtom"))


(defentry XCopyColormapAndFree(

    fixnum		;; display 
    fixnum		;; colormap 

)( fixnum "XCopyColormapAndFree"))


(defentry XCreateColormap(

    fixnum		;; display 
    fixnum		;; w 
    fixnum		;; visual 
    fixnum			;; alloc 			 

)( fixnum "XCreateColormap"))


(defentry XCreatePixmapCursor(

    fixnum		;; display 
    fixnum		;; source 
    fixnum		;; mask 
    fixnum		;; foreground_color 
    fixnum		;; background_color 
     fixnum	;; x 
     fixnum	;; y 			   

)( fixnum "XCreatePixmapCursor"))


(defentry XCreateGlyphCursor(

    fixnum		;; display 
    fixnum		;; source_font 
    fixnum		;; mask_font 
     fixnum	;; source_char 
     fixnum	;; mask_char 
    fixnum		;; foreground_color 
    fixnum		;; background_color 

)( fixnum "XCreateGlyphCursor"))


(defentry XCreateFontCursor(

    fixnum		;; display 
     fixnum	;; shape 

)( fixnum "XCreateFontCursor"))


(defentry XLoadFont(

    fixnum		;; display 
    object		;; name 

)( fixnum "XLoadFont"))


(defentry XCreateGC(

    fixnum		;; display 
    fixnum		;; d 
     fixnum	;; valuemask 
    fixnum		;; values 

)( fixnum "XCreateGC"))


(defentry XGContextFromGC(

    fixnum			;; gc 

)( fixnum "XGContextFromGC"))


(defentry XCreatePixmap(

    fixnum		;; display 
    fixnum		;; d 
     fixnum	;; width 
     fixnum	;; height 
     fixnum	;; depth 		        

)( fixnum "XCreatePixmap"))


(defentry XCreateBitmapFromData(

    fixnum		;; display 
    fixnum		;; d 
    object		;; data 
     fixnum	;; width 
     fixnum	;; height 

)( fixnum "XCreateBitmapFromData"))


(defentry XCreatePixmapFromBitmapData(

    fixnum		;; display 
    fixnum		;; d 
   object		;; data 
     fixnum	;; width 
     fixnum	;; height 
     fixnum	;; fg 
     fixnum	;; bg 
     fixnum	;; depth 

)( fixnum "XCreatePixmapFromBitmapData"))


(defentry XCreateSimpleWindow(

    fixnum		;; display 
    fixnum		;; parent 
    fixnum			;; x 
    fixnum			;; y 
     fixnum	;; width 
     fixnum	;; height 
     fixnum	;; border_width 
     fixnum	;; border 
     fixnum	;; background 

)( fixnum "XCreateSimpleWindow"))


(defentry XGetSelectionOwner(

    fixnum		;; display 
    fixnum		;; selection 

)( fixnum "XGetSelectionOwner"))


(defentry XCreateWindow(

    fixnum		;; display 
    fixnum		;; parent 
    fixnum			;; x 
    fixnum			;; y 
     fixnum	;; width 
     fixnum	;; height 
     fixnum	;; border_width 
    fixnum			;; depth 
     fixnum	;; class 
    fixnum		;; visual 
     fixnum	;; valuemask 
    fixnum	;; attributes 

)( fixnum "XCreateWindow")) 


(defentry  XListInstalledColormaps(

    fixnum		;; display 
    fixnum		;; w 
    fixnum		;; num_return 

)( fixnum  "XListInstalledColormaps"))


(defentry XListFonts(

    fixnum		;; display 
    object		;; pattern 
    fixnum			;; maxnames 
    fixnum		;; actual_count_return 

)( fixnum "XListFonts"))


(defentry XListFontsWithInfo(

    fixnum		;; display 
    object		;; pattern 
    fixnum			;; maxnames 
    fixnum		;; count_return 
    fixnum		;; info_return 

)( fixnum "XListFontsWithInfo"))


(defentry XGetFontPath(

    fixnum		;; display 
    fixnum		;; npaths_return 

)( fixnum "XGetFontPath"))


(defentry XListExtensions(

    fixnum		;; display 
    fixnum		;; nextensions_return 

)( fixnum "XListExtensions"))


(defentry  XListProperties(

    fixnum		;; display 
    fixnum		;; w 
    fixnum		;; num_prop_return 

)( fixnum  "XListProperties"))


(defentry XListHosts(

    fixnum		;; display 
    fixnum		;; nhosts_return 
    fixnum		;; state_return 

)( fixnum "XListHosts"))


(defentry XKeycodeToKeysym(

    fixnum		;; display 

    fixnum		;; fixnum 

    fixnum			;; index 

)( fixnum "XKeycodeToKeysym"))


(defentry XLookupKeysym(

    fixnum		;; key_event 
    fixnum			;; index 

)( fixnum "XLookupKeysym"))


(defentry  XGetKeyboardMapping(

    fixnum		;; display 

    fixnum		;; first_keycode

    fixnum			;; keycode_count 
    fixnum		;; keysyms_per_keycode_return 

)( fixnum  "XGetKeyboardMapping"))


(defentry XStringToKeysym(

    object		;; string 

)( fixnum "XStringToKeysym"))


(defentry XMaxRequestSize(

    fixnum		;; display 

)( fixnum "XMaxRequestSize"))


(defentry XResourceManagerString(

    fixnum		;; display 

)( fixnum "XResourceManagerString"))


(defentry XDisplayMotionBufferSize(

    fixnum		;; display 

)( fixnum "XDisplayMotionBufferSize"))


(defentry XVisualIDFromVisual(

    fixnum		;; visual 

)( fixnum "XVisualIDFromVisual"))

;; routines for dealing with extensions 



(defentry  XInitExtension(

    fixnum		;; display 
    object		;; name 

)( fixnum  "XInitExtension"))



(defentry  XAddExtension(

    fixnum		;; display 

)( fixnum  "XAddExtension"))


(defentry  XFindOnExtensionList(

    fixnum		;; structure 
    fixnum			;; number 

)( fixnum  "XFindOnExtensionList"))



;;;fix


;(defentry XEHeadOfExtensionList(

;    fixnum	;;object 

;)( fixnum "XEHeadOfExtensionList"))

;; these are routines for which there are also macros 


(defentry XRootWindow(

    fixnum		;; display 
    fixnum			;; screen_number 

)( fixnum "XRootWindow"))


(defentry XDefaultRootWindow(

    fixnum		;; display 

)( fixnum "XDefaultRootWindow"))


(defentry XRootWindowOfScreen(

    fixnum		;; screen 

)( fixnum "XRootWindowOfScreen"))


(defentry  XDefaultVisual(

    fixnum		;; display 
    fixnum			;; screen_number 

)( fixnum  "XDefaultVisual"))


(defentry  XDefaultVisualOfScreen(

    fixnum		;; screen 

)( fixnum  "XDefaultVisualOfScreen"))


(defentry XDefaultGC(

    fixnum		;; display 
    fixnum			;; screen_number 

)( fixnum "XDefaultGC"))


(defentry XDefaultGCOfScreen(

    fixnum		;; screen 

)( fixnum "XDefaultGCOfScreen"))


(defentry XBlackPixel(

    fixnum		;; display 
    fixnum			;; screen_number 

)( fixnum "XBlackPixel"))


(defentry XWhitePixel(

    fixnum		;; display 
    fixnum			;; screen_number 

)( fixnum "XWhitePixel"))


(defentry XAllPlanes(

;;    void

)( fixnum "XAllPlanes"))


(defentry XBlackPixelOfScreen(

    fixnum		;; screen 

)( fixnum "XBlackPixelOfScreen"))


(defentry XWhitePixelOfScreen(

    fixnum		;; screen 

)( fixnum "XWhitePixelOfScreen"))


(defentry XNextRequest(

    fixnum		;; display 

)( fixnum "XNextRequest"))


(defentry XLastKnownRequestProcessed(

    fixnum		;; display 

)( fixnum "XLastKnownRequestProcessed"))


(defentry  XServerVendor(

    fixnum		;; display 

)( fixnum  "XServerVendor"))


(defentry  XDisplayString(

    fixnum		;; display 

)( fixnum  "XDisplayString"))


(defentry XDefaultColormap(

    fixnum		;; display 
    fixnum			;; screen_number 

)( fixnum "XDefaultColormap"))


(defentry XDefaultColormapOfScreen(

    fixnum		;; screen 

)( fixnum "XDefaultColormapOfScreen"))


(defentry  XDisplayOfScreen(

    fixnum		;; screen 

)( fixnum  "XDisplayOfScreen"))


(defentry  XScreenOfDisplay(

    fixnum		;; display 
    fixnum			;; screen_number 

)( fixnum  "XScreenOfDisplay"))


(defentry  XDefaultScreenOfDisplay(

    fixnum		;; display 

)( fixnum  "XDefaultScreenOfDisplay"))


(defentry XEventMaskOfScreen(

    fixnum		;; screen 

)( fixnum "XEventMaskOfScreen"))



(defentry XScreenNumberOfScreen(

    fixnum		;; screen 

)( fixnum "XScreenNumberOfScreen"))



(defentry XSetErrorHandler (

    fixnum	;; handler 

)( fixnum "XSetErrorHandler" ))


;;fix


(defentry XSetIOErrorHandler (

    fixnum	;; handler 

)( fixnum "XSetIOErrorHandler" ))




(defentry XListPixmapFormats(

    fixnum		;; display 
    fixnum		;; count_return 

)( fixnum "XListPixmapFormats"))


(defentry  XListDepths(

    fixnum		;; display 
    fixnum			;; screen_number 
    fixnum		;; count_return 

)( fixnum  "XListDepths"))

;; ICCCM routines for things that don't require special include files; 
;; other declarations are given in Xutil.h                             


(defentry XReconfigureWMWindow(

    fixnum		;; display 
    fixnum		;; w 
    fixnum			;; screen_number 
     fixnum	;; mask 
    fixnum		;; changes 

)( fixnum "XReconfigureWMWindow"))



(defentry XGetWMProtocols(

    fixnum		;; display 
    fixnum		;; w 
    fixnum		;; protocols_return 
    fixnum		;; count_return 

)( fixnum "XGetWMProtocols"))


(defentry XSetWMProtocols(

    fixnum		;; display 
    fixnum		;; w 
    fixnum		;; protocols 
    fixnum			;; count 

)( fixnum "XSetWMProtocols"))


(defentry XIconifyWindow(

    fixnum		;; display 
    fixnum		;; w 
    fixnum			;; screen_number 

)( fixnum "XIconifyWindow"))


(defentry XWithdrawWindow(

    fixnum		;; display 
    fixnum		;; w 
    fixnum			;; screen_number 

)( fixnum "XWithdrawWindow"))

;;;fix


(defentry XGetCommand(

    fixnum		;; display 
    fixnum		;; w 
   fixnum 		;; argv_return 
    fixnum		;; argc_return 

)( fixnum "XGetCommand"))


(defentry XGetWMColormapWindows(

    fixnum		;; display 
    fixnum		;; w 
    fixnum		;; windows_return 
    fixnum		;; count_return 

)( fixnum "XGetWMColormapWindows"))


(defentry XSetWMColormapWindows(

    fixnum		;; display 
    fixnum		;; w 
    fixnum		;; colormap_windows 
    fixnum			;; count 

)( fixnum "XSetWMColormapWindows"))


(defentry XFreeStringList(

   fixnum 	;; list 

)( void "XFreeStringList"))


(defentry XSetTransientForHint(

    fixnum		;; display 
    fixnum		;; w 
    fixnum		;; prop_window 

)( void "XSetTransientForHint"))

;; The following are given in alphabetical order 



(defentry XActivateScreenSaver(

    fixnum		;; display 

)( void "XActivateScreenSaver"))



(defentry XAddHost(

    fixnum		;; display 
    fixnum	;; host 

)( void "XAddHost"))



(defentry XAddHosts(

    fixnum		;; display 
    fixnum		;; hosts 
    fixnum			;; num_hosts     

)( void "XAddHosts"))



(defentry XAddToExtensionList(

    fixnum	;; structure 
    fixnum		;; ext_data 

)( void "XAddToExtensionList"))



(defentry XAddToSaveSet(

    fixnum		;; display 
    fixnum		;; w 

)( void "XAddToSaveSet"))



(defentry XAllocColor(

    fixnum		;; display 
    fixnum		;; colormap 
    fixnum		;; screen_in_out 

)( fixnum "XAllocColor"))

;;;fix


(defentry XAllocColorCells(

    fixnum		;; display 
    fixnum		;; colormap 
    fixnum	        ;; contig 
     fixnum ;; plane_masks_return 
     fixnum	;; nplanes 
     fixnum ;; pixels_return 
     fixnum 	;; npixels 

)( fixnum "XAllocColorCells"))



(defentry XAllocColorPlanes(

    fixnum		;; display 
    fixnum		;; colormap 
    fixnum		;; contig 
     fixnum ;; pixels_return 
    fixnum			;; ncolors 
    fixnum			;; nreds 
    fixnum			;; ngreens 
    fixnum			;; nblues 
     fixnum ;; rmask_return 
     fixnum ;; gmask_return 
     fixnum ;; bmask_return 

)( fixnum "XAllocColorPlanes"))



(defentry XAllocNamedColor(

    fixnum		;; display 
    fixnum		;; colormap 
    object		;; color_name 
    fixnum		;; screen_def_return 
    fixnum		;; exact_def_return 

)( fixnum "XAllocNamedColor"))



(defentry XAllowEvents(

    fixnum		;; display 
    fixnum			;; event_mode 
    fixnum		;; time

)( void "XAllowEvents"))



(defentry XAutoRepeatOff(

    fixnum		;; display 

)( void "XAutoRepeatOff"))



(defentry XAutoRepeatOn(

    fixnum		;; display 

)( void "XAutoRepeatOn"))



(defentry XBell(

    fixnum		;; display 
    fixnum			;; percent 

)( void "XBell"))



(defentry XBitmapBitOrder(

    fixnum		;; display 

)( fixnum "XBitmapBitOrder"))



(defentry XBitmapPad(

    fixnum		;; display 

)( fixnum "XBitmapPad"))



(defentry XBitmapUnit(

    fixnum		;; display 

)( fixnum "XBitmapUnit"))



(defentry XCellsOfScreen(

    fixnum		;; screen 

)( fixnum "XCellsOfScreen"))



(defentry XChangeActivePointerGrab(

    fixnum		;; display 
     fixnum	;; event_mask 
    fixnum		;; cursor 
    fixnum		;; time

)( void "XChangeActivePointerGrab"))



(defentry XChangeGC(

    fixnum		;; display 
    fixnum			;; gc 
     fixnum	;; valuemask 
    fixnum		;; values 

)( void "XChangeGC"))



(defentry XChangeKeyboardControl(

    fixnum		;; display 
     fixnum	;; value_mask 
    fixnum 	;; values 

)( void "XChangeKeyboardControl"))



(defentry XChangeKeyboardMapping(

    fixnum		;; display 
    fixnum			;; first_keycode 
    fixnum			;; keysyms_per_keycode 
    fixnum			;; keysyms 
    fixnum			;; num_codes 

)( void "XChangeKeyboardMapping"))



(defentry XChangePointerControl(

    fixnum		;; display 
    fixnum		;; do_accel 
    fixnum		;; do_threshold 
    fixnum			;; accel_numerator 
    fixnum			;; accel_denominator 
    fixnum			;; threshold 

)( void "XChangePointerControl"))



(defentry XChangeProperty(

    fixnum		;; display 
    fixnum		;; w 
    fixnum		;; property 
    fixnum		;; type 
    fixnum			;; format 
    fixnum			;; mode 
    fixnum 	;; data 
    fixnum			;; nelements 

)( void "XChangeProperty"))



(defentry XChangeSaveSet(

    fixnum		;; display 
    fixnum		;; w 
    fixnum			;; change_mode 

)( void "XChangeSaveSet"))



(defentry XChangeWindowAttributes(

    fixnum		;; display 
    fixnum		;; w 
     fixnum	;; valuemask 
    fixnum ;; attributes 

)( void "XChangeWindowAttributes"))



(defentry XCheckMaskEvent(

    fixnum		;; display 
    fixnum		;; event_mask 
    fixnum		;; event_return 

)( fixnum "XCheckMaskEvent"))



(defentry XCheckTypedEvent(

    fixnum		;; display 
    fixnum			;; event_type 
    fixnum		;; event_return 

)( fixnum "XCheckTypedEvent"))



(defentry XCheckTypedWindowEvent(

    fixnum		;; display 
    fixnum		;; w 
    fixnum			;; event_type 
    fixnum		;; event_return 

)( fixnum "XCheckTypedWindowEvent"))



(defentry XCheckWindowEvent(

    fixnum		;; display 
    fixnum		;; w 
    fixnum		;; event_mask 
    fixnum		;; event_return 

)( fixnum "XCheckWindowEvent"))



(defentry XCirculateSubwindows(

    fixnum		;; display 
    fixnum		;; w 
    fixnum			;; direction 

)( void "XCirculateSubwindows"))



(defentry XCirculateSubwindowsDown(

    fixnum		;; display 
    fixnum		;; w 

)( void "XCirculateSubwindowsDown"))



(defentry XCirculateSubwindowsUp(

    fixnum		;; display 
    fixnum		;; w 

)( void "XCirculateSubwindowsUp"))



(defentry XClearArea(

    fixnum		;; display 
    fixnum		;; w 
    fixnum			;; x 
    fixnum			;; y 
     fixnum	;; width 
     fixnum	;; height 
    fixnum		;; exposures 

)( void "XClearArea"))



(defentry XClearWindow(

    fixnum		;; display 
    fixnum		;; w 

)( void "XClearWindow"))



(defentry XCloseDisplay(

    fixnum		;; display 

)( void "XCloseDisplay"))



(defentry XConfigureWindow(

    fixnum		;; display 
    fixnum		;; w 
     fixnum	;; value_mask 
    fixnum ;; values 		 

)( void "XConfigureWindow"))



(defentry XConnectionNumber(

    fixnum		;; display 

)( fixnum "XConnectionNumber"))



(defentry XConvertSelection(

    fixnum		;; display 
    fixnum		;; selection 
    fixnum 		;; target 
    fixnum		;; property 
    fixnum		;; requestor 
    fixnum		;; time

)( void "XConvertSelection"))



(defentry XCopyArea(

    fixnum		;; display 
    fixnum		;; src 
    fixnum		;; dest 
    fixnum			;; gc 
    fixnum			;; src_x 
    fixnum			;; src_y 
     fixnum	;; width 
     fixnum	;; height 
    fixnum			;; dest_x 
    fixnum			;; dest_y 

)( void "XCopyArea"))



(defentry XCopyGC(

    fixnum		;; display 
    fixnum			;; src 
     fixnum	;; valuemask 
    fixnum			;; dest 

)( void "XCopyGC"))



(defentry XCopyPlane(

    fixnum		;; display 
    fixnum		;; src 
    fixnum		;; dest 
    fixnum			;; gc 
    fixnum			;; src_x 
    fixnum			;; src_y 
     fixnum	;; width 
     fixnum	;; height 
    fixnum			;; dest_x 
    fixnum			;; dest_y 
     fixnum	;; plane 

)( void "XCopyPlane"))



(defentry XDefaultDepth(

    fixnum		;; display 
    fixnum			;; screen_number 

)( fixnum "XDefaultDepth"))



(defentry XDefaultDepthOfScreen(

    fixnum		;; screen 

)( fixnum "XDefaultDepthOfScreen"))



(defentry XDefaultScreen(

    fixnum		;; display 

)( fixnum "XDefaultScreen"))



(defentry XDefineCursor(

    fixnum		;; display 
    fixnum		;; w 
    fixnum		;; cursor 

)( void "XDefineCursor"))



(defentry XDeleteProperty(

    fixnum		;; display 
    fixnum		;; w 
    fixnum		;; property 

)( void "XDeleteProperty"))



(defentry XDestroyWindow(

    fixnum		;; display 
    fixnum		;; w 

)( void "XDestroyWindow"))



(defentry XDestroySubwindows(

    fixnum		;; display 
    fixnum		;; w 

)( void "XDestroySubwindows"))



(defentry XDoesBackingStore(

    fixnum		;; screen     

)( fixnum "XDoesBackingStore"))



(defentry XDoesSaveUnders(

    fixnum		;; screen 

)( fixnum "XDoesSaveUnders"))



(defentry XDisableAccessControl(

    fixnum		;; display 

)( void "XDisableAccessControl"))




(defentry XDisplayCells(

    fixnum		;; display 
    fixnum			;; screen_number 

)( fixnum "XDisplayCells"))



(defentry XDisplayHeight(

    fixnum		;; display 
    fixnum			;; screen_number 

)( fixnum "XDisplayHeight"))



(defentry XDisplayHeightMM(

    fixnum		;; display 
    fixnum			;; screen_number 

)( fixnum "XDisplayHeightMM"))



(defentry XDisplayKeycodes(

    fixnum		;; display 
    fixnum		;; min_keycodes_return 
    fixnum		;; max_keycodes_return 

)( void "XDisplayKeycodes"))



(defentry XDisplayPlanes(

    fixnum		;; display 
    fixnum			;; screen_number 

)( fixnum "XDisplayPlanes"))



(defentry XDisplayWidth(

    fixnum		;; display 
    fixnum			;; screen_number 

)( fixnum "XDisplayWidth"))



(defentry XDisplayWidthMM(

    fixnum		;; display 
    fixnum			;; screen_number 

)( fixnum "XDisplayWidthMM"))



(defentry XDrawArc(

    fixnum		;; display 
    fixnum		;; d 
    fixnum			;; gc 
    fixnum			;; x 
    fixnum			;; y 
     fixnum	;; width 
     fixnum	;; height 
    fixnum			;; angle1 
    fixnum			;; angle2 

)( void "XDrawArc"))



(defentry XDrawArcs(

    fixnum		;; display 
    fixnum		;; d 
    fixnum			;; gc 
    fixnum 	;; arcs 
    fixnum			;; narcs 

)( void "XDrawArcs"))



(defentry XDrawImageString(

    fixnum		;; display 
    fixnum		;; d 
    fixnum			;; gc 
    fixnum			;; x 
    fixnum			;; y 
    object		;; string 
    fixnum			;; length 

)( void "XDrawImageString"))



(defentry XDrawImageString16(

    fixnum		;; display 
    fixnum		;; d 
    fixnum			;; gc 
    fixnum			;; x 
    fixnum			;; y 
     fixnum	;; string 
    fixnum			;; length 

)( void "XDrawImageString16"))



(defentry XDrawLine(

    fixnum		;; display 
    fixnum		;; d 
    fixnum			;; gc 
    fixnum			;; x1 
    fixnum			;; x2 
    fixnum			;; y1 
    fixnum			;; y2 

)( void "XDrawLine"))



(defentry XDrawLines(

    fixnum		;; display 
    fixnum		;; d 
    fixnum			;; gc 
    fixnum		;; points 
    fixnum			;; npoints 
    fixnum			;; mode 

)( void "XDrawLines"))



(defentry XDrawPoint(

    fixnum		;; display 
    fixnum		;; d 
    fixnum			;; gc 
    fixnum			;; x 
    fixnum			;; y 

)( void "XDrawPoint"))



(defentry XDrawPoints(

    fixnum		;; display 
    fixnum		;; d 
    fixnum			;; gc 
    fixnum		;; points 
    fixnum			;; npoints 
    fixnum			;; mode 

)( void "XDrawPoints"))



(defentry XDrawRectangle(

    fixnum		;; display 
    fixnum		;; d 
    fixnum			;; gc 
    fixnum			;; x 
    fixnum			;; y 
     fixnum	;; width 
     fixnum	;; height 

)( void "XDrawRectangle"))



(defentry XDrawRectangles(

    fixnum		;; display 
    fixnum		;; d 
    fixnum			;; gc 
    fixnum 	;; rectangles 
    fixnum			;; nrectangles 

)( void "XDrawRectangles"))



(defentry XDrawSegments(

    fixnum		;; display 
    fixnum		;; d 
    fixnum			;; gc 
    fixnum 	;; segments 
    fixnum			;; nsegments 

)( void "XDrawSegments"))



(defentry XDrawString(

    fixnum		;; display 
    fixnum		;; d 
    fixnum			;; gc 
    fixnum			;; x 
    fixnum			;; y 
    object		;; string 
    fixnum			;; length 

)( void "XDrawString"))



(defentry XDrawString16(

    fixnum		;; display 
    fixnum		;; d 
    fixnum			;; gc 
    fixnum			;; x 
    fixnum			;; y 
     fixnum	;; string 
    fixnum			;; length 

)( void "XDrawString16"))



(defentry XDrawText(

    fixnum		;; display 
    fixnum		;; d 
    fixnum			;; gc 
    fixnum			;; x 
    fixnum			;; y 
    fixnum 	;; items 
    fixnum			;; nitems 

)( void "XDrawText"))



(defentry XDrawText16(

    fixnum		;; display 
    fixnum		;; d 
    fixnum			;; gc 
    fixnum			;; x 
    fixnum			;; y 
    fixnum ;; items 
    fixnum			;; nitems 

)( void "XDrawText16"))



(defentry XEnableAccessControl(

    fixnum		;; display 

)( void "XEnableAccessControl"))



(defentry XEventsQueued(

    fixnum		;; display 
    fixnum			;; mode 

)( fixnum "XEventsQueued"))



(defentry XFetchName(

    fixnum		;; display 
    fixnum		;; w 
   fixnum 	;; window_name_return 

)( fixnum "XFetchName"))



(defentry XFillArc(

    fixnum		;; display 
    fixnum		;; d 
    fixnum			;; gc 
    fixnum			;; x 
    fixnum			;; y 
     fixnum	;; width 
     fixnum	;; height 
    fixnum			;; angle1 
    fixnum			;; angle2 

)( void "XFillArc"))



(defentry XFillArcs(

    fixnum		;; display 
    fixnum		;; d 
    fixnum			;; gc 
    fixnum 	;; arcs 
    fixnum			;; narcs 

)( void "XFillArcs"))



(defentry XFillPolygon(

    fixnum		;; display 
    fixnum		;; d 
    fixnum			;; gc 
    fixnum		;; points 
    fixnum			;; npoints 
    fixnum			;; shape 
    fixnum			;; mode 

)( void "XFillPolygon"))



(defentry XFillRectangle(

    fixnum		;; display 
    fixnum		;; d 
    fixnum			;; gc 
    fixnum			;; x 
    fixnum			;; y 
     fixnum	;; width 
     fixnum	;; height 

)( void "XFillRectangle"))



(defentry XFillRectangles(

    fixnum		;; display 
    fixnum		;; d 
    fixnum			;; gc 
    fixnum 	;; rectangles 
    fixnum			;; nrectangles 

)( void "XFillRectangles"))



(defentry XFlush(

    fixnum		;; display 

)( void "XFlush"))



(defentry XForceScreenSaver(

    fixnum		;; display 
    fixnum			;; mode 

)( void "XForceScreenSaver"))



(defentry XFree(

   object		;; data 

)( void "XFree"))



(defentry XFreeColormap(

    fixnum		;; display 
    fixnum		;; colormap 

)( void "XFreeColormap"))



(defentry XFreeColors(

    fixnum		;; display 
    fixnum		;; colormap 
     fixnum ;; pixels 
    fixnum			;; npixels 
     fixnum	;; planes 

)( void "XFreeColors"))



(defentry XFreeCursor(

    fixnum		;; display 
    fixnum		;; cursor 

)( void "XFreeCursor"))



(defentry XFreeExtensionList(

   fixnum 	;; list     

)( void "XFreeExtensionList"))



(defentry XFreeFont(

    fixnum		;; display 
    fixnum	;; font_struct 

)( void "XFreeFont"))



(defentry XFreeFontInfo(

   fixnum 	;; names 
    fixnum	;; free_info 
    fixnum			;; actual_count 

)( void "XFreeFontInfo"))



(defentry XFreeFontNames(

   fixnum 	;; list 

)( void "XFreeFontNames"))



(defentry XFreeFontPath(

   fixnum 	;; list 

)( void "XFreeFontPath"))



(defentry XFreeGC(

    fixnum		;; display 
    fixnum			;; gc 

)( void "XFreeGC"))



(defentry XFreeModifiermap(

    fixnum	;; modmap 

)( void "XFreeModifiermap"))



(defentry XFreePixmap(

    fixnum		;; display 
    fixnum		;; fixnum 

)( void "XFreePixmap"))



(defentry XGeometry(

    fixnum		;; display 
    fixnum			;; screen 
    object		;; position 
    object		;; default_position 
     fixnum	;; bwidth 
     fixnum	;; fwidth 
     fixnum	;; fheight 
    fixnum			;; xadder 
    fixnum			;; yadder 
    fixnum		;; x_return 
    fixnum		;; y_return 
    fixnum		;; width_return 
    fixnum		;; height_return 

)( fixnum "XGeometry"))



(defentry XGetErrorDatabaseText(

    fixnum		;; display 
    object		;; name 
    object		;; message 
    object		;; default_string 
   object		;; buffer_return 
    fixnum			;; length 

)( void "XGetErrorDatabaseText"))



(defentry XGetErrorText(

    fixnum		;; display 
    fixnum			;; code 
   object		;; buffer_return 
    fixnum			;; length 

)( void "XGetErrorText"))



(defentry XGetFontProperty(

    fixnum	;; font_struct 
    fixnum		;; atom 
     fixnum ;; value_return 

)( fixnum "XGetFontProperty"))



(defentry XGetGCValues(

    fixnum		;; display 
    fixnum			;; gc 
     fixnum	;; valuemask 
    fixnum 	;; values_return 

)( fixnum "XGetGCValues"))



(defentry XGetGeometry(

    fixnum		;; display 
    fixnum		;; d 
    fixnum		;; root_return 
    fixnum		;; x_return 
    fixnum		;; y_return 
     fixnum	;; width_return 
     fixnum	;; height_return 
     fixnum	;; border_width_return 
     fixnum	;; depth_return 

)( fixnum "XGetGeometry"))



(defentry XGetIconName(

    fixnum		;; display 
    fixnum		;; w 
   fixnum 	;; icon_name_return 

)( fixnum "XGetIconName"))



(defentry XGetInputFocus(

    fixnum		;; display 
    fixnum		;; focus_return 
    fixnum		;; revert_to_return 

)( void "XGetInputFocus"))



(defentry XGetKeyboardControl(

    fixnum		;; display 
    fixnum ;; values_return 

)( void "XGetKeyboardControl"))



(defentry XGetPointerControl(

    fixnum		;; display 
    fixnum		;; accel_numerator_return 
    fixnum		;; accel_denominator_return 
    fixnum		;; threshold_return 

)( void "XGetPointerControl"))



(defentry XGetPointerMapping(

    fixnum		;; display 
    object	;; map_return 
    fixnum			;; nmap 

)( fixnum "XGetPointerMapping"))



(defentry XGetScreenSaver(

    fixnum		;; display 
    fixnum		;; intout_return 
    fixnum		;; interval_return 
    fixnum		;; prefer_blanking_return 
    fixnum		;; allow_exposures_return 

)( void "XGetScreenSaver"))



(defentry XGetTransientForHint(

    fixnum		;; display 
    fixnum		;; w 
    fixnum		;; prop_window_return 

)( fixnum "XGetTransientForHint"))



(defentry XGetWindowProperty(

    fixnum		;; display 
    fixnum		;; w 
    fixnum		;; property 
    fixnum		;; int_offset 
    fixnum		;; int_length 
    fixnum		;; delete 
    fixnum		;; req_type 
    fixnum 	;; actual_type_return 
    fixnum		;; actual_format_return 
     fixnum ;; nitems_return 
     fixnum ;; bytes_after_return 
    fixnum ;; prop_return 

)( fixnum "XGetWindowProperty"))



(defentry XGetWindowAttributes(

    fixnum		;; display 
    fixnum		;; w 
    fixnum ;; Window_attributes_return 

)( fixnum "XGetWindowAttributes"))



(defentry XGrabButton(

    fixnum		;; display 
     fixnum	;; button 
     fixnum	;; modifiers 
    fixnum		;; grab_window 
    fixnum		;; owner_events 
     fixnum	;; event_mask 
    fixnum			;; pointer_mode 
    fixnum			;; keyboard_mode 
    fixnum		;; confine_to 
    fixnum		;; cursor 

)( void "XGrabButton"))



(defentry XGrabKey(

    fixnum		;; display 
    fixnum			;; keycode
     fixnum	;; modifiers 
    fixnum		;; grab_window 
    fixnum		;; owner_events 
    fixnum			;; pointer_mode 
    fixnum			;; keyboard_mode 

)( void "XGrabKey"))



(defentry XGrabKeyboard(

    fixnum		;; display 
    fixnum		;; grab_window 
    fixnum		;; owner_events 
    fixnum			;; pointer_mode 
    fixnum			;; keyboard_mode 
    fixnum		;; fixnum 

)( fixnum "XGrabKeyboard"))



(defentry XGrabPointer(

    fixnum		;; display 
    fixnum		;; grab_window 
    fixnum		;; owner_events 
     fixnum	;; event_mask 
    fixnum			;; pointer_mode 
    fixnum			;; keyboard_mode 
    fixnum		;; confine_to 
    fixnum		;; cursor 
    fixnum		;; fixnum 

)( fixnum "XGrabPointer"))



(defentry XGrabServer(

    fixnum		;; display 

)( void "XGrabServer"))



(defentry XHeightMMOfScreen(

    fixnum		;; screen 

)( fixnum "XHeightMMOfScreen"))



(defentry XHeightOfScreen(

    fixnum		;; screen 

)( fixnum "XHeightOfScreen"))



(defentry XImageByteOrder(

    fixnum		;; display 

)( fixnum "XImageByteOrder"))



(defentry XInstallColormap(

    fixnum		;; display 
    fixnum		;; colormap 

)( void "XInstallColormap"))



(defentry XKeysymToKeycode(

    fixnum		;; display 
    fixnum		;; keysym 

)( fixnum "XKeysymToKeycode"))



(defentry XKillClient(

    fixnum		;; display 
    fixnum			;; resource 

)( void "XKillClient"))


(defentry XLookupColor(

    fixnum		;; display 
    fixnum		;; colormap 
    object		;; color_name 
    fixnum		;; exact_def_return 
    fixnum		;; screen_def_return 

)( fixnum "XLookupColor"))



(defentry XLowerWindow(

    fixnum		;; display 
    fixnum		;; w 

)( void "XLowerWindow"))



(defentry XMapRaised(

    fixnum		;; display 
    fixnum		;; w 

)( void "XMapRaised"))



(defentry XMapSubwindows(

    fixnum		;; display 
    fixnum		;; w 

)( void "XMapSubwindows"))



(defentry XMapWindow(

    fixnum		;; display 
    fixnum		;; w 

)( void "XMapWindow"))



(defentry XMaskEvent(

    fixnum		;; display 
    fixnum		;; event_mask 
    fixnum		;; event_return 

)( void "XMaskEvent"))



(defentry XMaxCmapsOfScreen(

    fixnum		;; screen 

)( fixnum "XMaxCmapsOfScreen"))



(defentry XMinCmapsOfScreen(

    fixnum		;; screen 

)( fixnum "XMinCmapsOfScreen"))



(defentry XMoveResizeWindow(

    fixnum		;; display 
    fixnum		;; w 
    fixnum			;; x 
    fixnum			;; y 
     fixnum	;; width 
     fixnum	;; height 

)( void "XMoveResizeWindow"))



(defentry XMoveWindow(

    fixnum		;; display 
    fixnum		;; w 
    fixnum			;; x 
    fixnum			;; y 

)( void "XMoveWindow"))



(defentry XNextEvent(

    fixnum		;; display 
    fixnum		;; event_return 

)( void "XNextEvent"))



(defentry XNoOp(

    fixnum		;; display 

)( void "XNoOp"))



(defentry XParseColor(

    fixnum		;; display 
    fixnum		;; colormap 
    object		;; spec 
    fixnum		;; exact_def_return 

)( fixnum "XParseColor"))



(defentry XParseGeometry(

    object		;; parsestring 
    fixnum		;; x_return 
    fixnum		;; y_return 
     fixnum	;; width_return 
     fixnum	;; height_return 

)( fixnum "XParseGeometry"))



(defentry XPeekEvent(

    fixnum		;; display 
    fixnum		;; event_return 

)( void "XPeekEvent"))




(defentry XPending(

    fixnum		;; display 

)( fixnum "XPending"))



(defentry XPlanesOfScreen(

    fixnum		;; screen 
    

)( fixnum "XPlanesOfScreen"))



(defentry XProtocolRevision(

    fixnum		;; display 

)( fixnum "XProtocolRevision"))



(defentry XProtocolVersion(

    fixnum		;; display 

)( fixnum "XProtocolVersion"))




(defentry XPutBackEvent(

    fixnum		;; display 
    fixnum		;; event 

)( void "XPutBackEvent"))



(defentry XPutImage(

    fixnum		;; display 
    fixnum		;; d 
    fixnum			;; gc 
    fixnum 	;; image 
    fixnum			;; src_x 
    fixnum			;; src_y 
    fixnum			;; dest_x 
    fixnum			;; dest_y 
     fixnum	;; width 
     fixnum	;; height 	  

)( void "XPutImage"))



(defentry XQLength(

    fixnum		;; display 

)( fixnum "XQLength"))



(defentry XQueryBestCursor(

    fixnum		;; display 
    fixnum		;; d 
     fixnum        ;; width 
     fixnum	;; height 
     fixnum	;; width_return 
     fixnum	;; height_return 

)( fixnum "XQueryBestCursor"))



(defentry XQueryBestSize(

    fixnum		;; display 
    fixnum			;; class 
    fixnum		;; which_screen 
     fixnum	;; width 
     fixnum	;; height 
     fixnum	;; width_return 
     fixnum	;; height_return 

)( fixnum "XQueryBestSize"))



(defentry XQueryBestStipple(

    fixnum		;; display 
    fixnum		;; which_screen 
     fixnum	;; width 
     fixnum	;; height 
     fixnum	;; width_return 
     fixnum	;; height_return 

)( fixnum "XQueryBestStipple"))



(defentry XQueryBestTile(

    fixnum		;; display 
    fixnum		;; which_screen 
     fixnum	;; width 
     fixnum	;; height 
     fixnum	;; width_return 
     fixnum	;; height_return 

)( fixnum "XQueryBestTile"))



(defentry XQueryColor(

    fixnum		;; display 
    fixnum		;; colormap 
    fixnum		;; def_in_out 

)( void "XQueryColor"))



(defentry XQueryColors(

    fixnum		;; display 
    fixnum		;; colormap 
    fixnum		;; defs_in_out 
    fixnum			;; ncolors 

)( void "XQueryColors"))



(defentry XQueryExtension(

    fixnum		;; display 
    object		;; name 
    fixnum		;; major_opcode_return 
    fixnum		;; first_event_return 
    fixnum		;; first_error_return 

)( fixnum "XQueryExtension"))


;;fix
(defentry XQueryKeymap(

    fixnum		;; display 
    fixnum		;; keys_return 

)( void "XQueryKeymap"))



(defentry XQueryPointer(

    fixnum		;; display 
    fixnum		;; w 
    fixnum		;; root_return 
    fixnum		;; child_return 
    fixnum		;; root_x_return 
    fixnum		;; root_y_return 
    fixnum		;; win_x_return 
    fixnum		;; win_y_return 
     fixnum       ;; mask_return 

)( fixnum "XQueryPointer"))



(defentry XQueryTextExtents(

    fixnum		;; display 
    fixnum			;; font_ID 
    object		;; string 
    fixnum			;; nchars 
    fixnum		;; direction_return 
    fixnum		;; font_ascent_return 
    fixnum		;; font_descent_return 
    fixnum	;; overall_return     

)( void "XQueryTextExtents"))



(defentry XQueryTextExtents16(

    fixnum		;; display 
    fixnum			;; font_ID 
     fixnum	;; string 
    fixnum			;; nchars 
    fixnum		;; direction_return 
    fixnum		;; font_ascent_return 
    fixnum		;; font_descent_return 
    fixnum	;; overall_return 

)( void "XQueryTextExtents16"))



(defentry XQueryTree(

    fixnum		;; display 
    fixnum		;; w 
    fixnum		;; root_return 
    fixnum		;; parent_return 
    fixnum		;; children_return 
     fixnum	;; nchildren_return 

)( fixnum "XQueryTree"))



(defentry XRaiseWindow(

    fixnum		;; display 
    fixnum		;; w 

)( void "XRaiseWindow"))



(defentry XReadBitmapFile(

    fixnum		;; display 
    fixnum 		;; d 
    object		;; filename 
     fixnum	;; width_return 
     fixnum	;; height_return 
    fixnum		;; bitmap_return 
    fixnum		;; x_hot_return 
    fixnum		;; y_hot_return 

)( fixnum "XReadBitmapFile"))



(defentry XRebindKeysym(

    fixnum		;; display 
    fixnum		;; keysym 
    fixnum		;; list 
    fixnum			;; mod_count 
     object	;; string 
    fixnum			;; bytes_string 

)( void "XRebindKeysym"))



(defentry XRecolorCursor(

    fixnum		;; display 
    fixnum		;; cursor 
    fixnum		;; foreground_color 
    fixnum		;; background_color 

)( void "XRecolorCursor"))



(defentry XRefreshKeyboardMapping(

    fixnum	;; event_map     

)( void "XRefreshKeyboardMapping"))



(defentry XRemoveFromSaveSet(

    fixnum		;; display 
    fixnum		;; w 

)( void "XRemoveFromSaveSet"))



(defentry XRemoveHost(

    fixnum		;; display 
    fixnum ;; host 

)( void "XRemoveHost"))



(defentry XRemoveHosts(

    fixnum		;; display 
    fixnum	;; hosts 
    fixnum			;; num_hosts 

)( void "XRemoveHosts"))



(defentry XReparentWindow(

    fixnum		;; display 
    fixnum		;; w 
    fixnum		;; parent 
    fixnum			;; x 
    fixnum			;; y 

)( void "XReparentWindow"))



(defentry XResetScreenSaver(

    fixnum		;; display 

)( void "XResetScreenSaver"))



(defentry XResizeWindow(

    fixnum		;; display 
    fixnum		;; w 
     fixnum	;; width 
     fixnum	;; height 

)( void "XResizeWindow"))



(defentry XRestackWindows(

    fixnum		;; display 
    fixnum		;; windows 
    fixnum			;; nwindows 

)( void "XRestackWindows"))



(defentry XRotateBuffers(

    fixnum		;; display 
    fixnum			;; rotate 

)( void "XRotateBuffers"))



(defentry XRotateWindowProperties(

    fixnum		;; display 
    fixnum		;; w 
    fixnum 	;; properties 
    fixnum			;; num_prop 
    fixnum			;; npositions 

)( void "XRotateWindowProperties"))



(defentry XScreenCount(

    fixnum		;; display 

)( fixnum "XScreenCount"))



(defentry XSelectInput(

    fixnum		;; display 
    fixnum		;; w 
    fixnum		;; event_mask 

)( void "XSelectInput"))



(defentry XSendEvent(

    fixnum		;; display 
    fixnum		;; w 
    fixnum		;; propagate 
    fixnum		;; event_mask 
    fixnum		;; event_send 

)( fixnum "XSendEvent"))



(defentry XSetAccessControl(

    fixnum		;; display 
    fixnum			;; mode 

)( void "XSetAccessControl"))



(defentry XSetArcMode(

    fixnum		;; display 
    fixnum			;; gc 
    fixnum			;; arc_mode 

)( void "XSetArcMode"))



(defentry XSetBackground(

    fixnum		;; display 
    fixnum			;; gc 
     fixnum	;; background 

)( void "XSetBackground"))



(defentry XSetClipMask(

    fixnum		;; display 
    fixnum			;; gc 
    fixnum		;; fixnum 

)( void "XSetClipMask"))



(defentry XSetClipOrigin(

    fixnum		;; display 
    fixnum			;; gc 
    fixnum			;; clip_x_origin 
    fixnum			;; clip_y_origin 

)( void "XSetClipOrigin"))



(defentry XSetClipRectangles(

    fixnum		;; display 
    fixnum			;; gc 
    fixnum			;; clip_x_origin 
    fixnum			;; clip_y_origin 
    fixnum			;; rectangles 
    fixnum			;; n 
    fixnum			;; ordering 

)( void "XSetClipRectangles"))



(defentry XSetCloseDownMode(

    fixnum		;; display 
    fixnum			;; close_mode 

)( void "XSetCloseDownMode"))



(defentry XSetCommand(

    fixnum		;; display 
    fixnum		;; w 
   fixnum 	;; argv 
    fixnum			;; argc 

)( void "XSetCommand"))



(defentry XSetDashes(

    fixnum		;; display 
    fixnum			;; gc 
    fixnum			;; dash_offset 
    object		;; dash_list 
    fixnum			;; n 

)( void "XSetDashes"))



(defentry XSetFillRule(

    fixnum		;; display 
    fixnum			;; gc 
    fixnum			;; fill_rule 

)( void "XSetFillRule"))



(defentry XSetFillStyle(

    fixnum		;; display 
    fixnum			;; gc 
    fixnum			;; fill_style 

)( void "XSetFillStyle"))



(defentry XSetFont(

    fixnum		;; display 
    fixnum			;; gc 
    fixnum		;; font 

)( void "XSetFont"))



(defentry XSetFontPath(

    fixnum		;; display 
   fixnum 	;; directories 
    fixnum			;; ndirs 	     

)( void "XSetFontPath"))



(defentry XSetForeground(

    fixnum		;; display 
    fixnum			;; gc 
     fixnum	;; foreground 

)( void "XSetForeground"))



(defentry XSetFunction(

    fixnum		;; display 
    fixnum			;; gc 
    fixnum			;; function 

)( void "XSetFunction"))



(defentry XSetGraphicsExposures(

    fixnum		;; display 
    fixnum			;; gc 
    fixnum		;; graphics_exposures 

)( void "XSetGraphicsExposures"))



(defentry XSetIconName(

    fixnum		;; display 
    fixnum		;; w 
    object		;; icon_name 

)( void "XSetIconName"))



(defentry XSetInputFocus(

    fixnum		;; display 
    fixnum		;; focus 
    fixnum			;; revert_to 
    fixnum		;; fixnum 

)( void "XSetInputFocus"))



(defentry XSetLineAttributes(

    fixnum		;; display 
    fixnum			;; gc 
     fixnum	;; line_width 
    fixnum			;; line_style 
    fixnum			;; cap_style 
    fixnum			;; join_style 

)( void "XSetLineAttributes"))



(defentry XSetModifierMapping(

    fixnum		;; display 
    fixnum	;; modmap 

)( fixnum "XSetModifierMapping"))



(defentry XSetPlaneMask(

    fixnum		;; display 
    fixnum			;; gc 
     fixnum	;; plane_mask 

)( void "XSetPlaneMask"))



(defentry XSetPointerMapping(

    fixnum		;; display 
     object	;; map 
    fixnum			;; nmap 

)( fixnum "XSetPointerMapping"))



(defentry XSetScreenSaver(

    fixnum		;; display 
    fixnum			;; intout 
    fixnum			;; interval 
    fixnum			;; prefer_blanking 
    fixnum			;; allow_exposures 

)( void "XSetScreenSaver"))



(defentry XSetSelectionOwner(

    fixnum		;; display 
    fixnum	        ;; selection 
    fixnum		;; owner 
    fixnum		;; fixnum 

)( void "XSetSelectionOwner"))



(defentry XSetState(

    fixnum		;; display 
    fixnum			;; gc 
     fixnum 	;; foreground 
     fixnum	;; background 
    fixnum			;; function 
     fixnum	;; plane_mask 

)( void "XSetState"))



(defentry XSetStipple(

    fixnum		;; display 
    fixnum			;; gc 
    fixnum		;; stipple 

)( void "XSetStipple"))



(defentry XSetSubwindowMode(

    fixnum		;; display 
    fixnum			;; gc 
    fixnum			;; subwindow_mode 

)( void "XSetSubwindowMode"))



(defentry XSetTSOrigin(

    fixnum		;; display 
    fixnum			;; gc 
    fixnum			;; ts_x_origin 
    fixnum			;; ts_y_origin 

)( void "XSetTSOrigin"))



(defentry XSetTile(

    fixnum		;; display 
    fixnum			;; gc 
    fixnum		;; tile 

)( void "XSetTile"))



(defentry XSetWindowBackground(

    fixnum		;; display 
    fixnum		;; w 
     fixnum	;; background_pixel 

)( void "XSetWindowBackground"))



(defentry XSetWindowBackgroundPixmap(

    fixnum		;; display 
    fixnum		;; w 
    fixnum		;; background_pixmap 

)( void "XSetWindowBackgroundPixmap"))



(defentry XSetWindowBorder(

    fixnum		;; display 
    fixnum		;; w 
     fixnum	;; border_pixel 

)( void "XSetWindowBorder"))



(defentry XSetWindowBorderPixmap(

    fixnum		;; display 
    fixnum		;; w 
    fixnum		;; border_pixmap

)( void "XSetWindowBorderPixmap"))



(defentry XSetWindowBorderWidth(

    fixnum		;; display 
    fixnum		;; w 
     fixnum	;; width 

)( void "XSetWindowBorderWidth"))



(defentry XSetWindowColormap(

    fixnum		;; display 
    fixnum		;; w 
    fixnum		;; colormap 

)( void "XSetWindowColormap"))



(defentry XStoreBuffer(

    fixnum		;; display 
    object		;; bytes 
    fixnum			;; nbytes 
    fixnum			;; buffer 

)( void "XStoreBuffer"))



(defentry XStoreBytes(

    fixnum		;; display 
    object		;; bytes 
    fixnum			;; nbytes 

)( void "XStoreBytes"))



(defentry XStoreColor(

    fixnum		;; display 
    fixnum		;; colormap 
    fixnum		;; color 

)( void "XStoreColor"))



(defentry XStoreColors(

    fixnum		;; display 
    fixnum		;; colormap 
    fixnum		;; color 
    fixnum			;; ncolors 

)( void "XStoreColors"))



(defentry XStoreName(

    fixnum		;; display 
    fixnum		;; w 
    object		;; window_name 

)( void "XStoreName"))



(defentry XStoreNamedColor(

    fixnum		;; display 
    fixnum		;; colormap 
    object		;; color 
     fixnum	;; pixel 
    fixnum			;; flags 

)( void "XStoreNamedColor"))



(defentry XSync(

    fixnum		;; display 
    fixnum		;; discard 

)( void "XSync"))



(defentry XTextExtents(

    fixnum	;; font_struct 
    object		;; string 
    fixnum			;; nchars 
    fixnum		;; direction_return 
    fixnum		;; font_ascent_return 
    fixnum		;; font_descent_return 
    fixnum	;; overall_return 

)( void "XTextExtents"))



(defentry XTextExtents16(

    fixnum	;; font_struct 
     fixnum	;; string 
    fixnum			;; nchars 
    fixnum		;; direction_return 
    fixnum		;; font_ascent_return 
    fixnum		;; font_descent_return 
    fixnum	;; overall_return 

)( void "XTextExtents16"))



(defentry XTextWidth(

    fixnum	;; font_struct 
    object		;; string 
    fixnum			;; count 

)( fixnum "XTextWidth"))



(defentry XTextWidth16(

    fixnum	;; font_struct 
    fixnum		;; string 
    fixnum			;; count 

)( fixnum "XTextWidth16"))



(defentry XTranslateCoordinates(

    fixnum		;; display 
    fixnum		;; src_w 
    fixnum		;; dest_w 
    fixnum		;; src_x 
    fixnum		;; src_y 
    fixnum		;; dest_x_return 
    fixnum		;; dest_y_return 
    fixnum		;; child_return 

)( fixnum "XTranslateCoordinates"))



(defentry XUndefineCursor(

    fixnum		;; display 
    fixnum		;; w 

)( void "XUndefineCursor"))



(defentry XUngrabButton(

    fixnum		;; display 
     fixnum	;; button 
     fixnum	;; modifiers 
    fixnum		;; grab_window 

)( void "XUngrabButton"))



(defentry XUngrabKey(

    fixnum		;; display 
    fixnum		;; keycode
     fixnum	;; modifiers 
    fixnum		;; grab_window 

)( void "XUngrabKey"))



(defentry XUngrabKeyboard(

    fixnum		;; display 
    fixnum		;; fixnum 

)( void "XUngrabKeyboard"))



(defentry XUngrabPointer(

    fixnum		;; display 
    fixnum		;; fixnum 

)( void "XUngrabPointer"))



(defentry XUngrabServer(

    fixnum		;; display 

)( void "XUngrabServer"))



(defentry XUninstallColormap(

    fixnum		;; display 
    fixnum		;; colormap 

)( void "XUninstallColormap"))



(defentry XUnloadFont(

    fixnum		;; display 
    fixnum	;; font 

)( void "XUnloadFont"))



(defentry XUnmapSubwindows(

    fixnum		;; display 
    fixnum		;; w 

)( void "XUnmapSubwindows"))



(defentry XUnmapWindow(

    fixnum		;; display 
    fixnum		;; w 

)( void "XUnmapWindow"))



(defentry XVendorRelease(

    fixnum		;; display 

)( fixnum "XVendorRelease"))



(defentry XWarpPointer(

    fixnum		;; display 
    fixnum		;; src_w 
    fixnum		;; dest_w 
    fixnum			;; src_x 
    fixnum			;; src_y 
     fixnum	;; src_width 
     fixnum	;; src_height 
    fixnum			;; dest_x 
    fixnum			;; dest_y 	     

)( void "XWarpPointer"))



(defentry XWidthMMOfScreen(

    fixnum		;; screen 

)( fixnum "XWidthMMOfScreen"))



(defentry XWidthOfScreen(

    fixnum		;; screen 

)( fixnum "XWidthOfScreen"))



(defentry XWindowEvent(

    fixnum		;; display 
    fixnum		;; w 
    fixnum		;; event_mask 
    fixnum		;; event_return 

)( void "XWindowEvent"))



(defentry XWriteBitmapFile(

    fixnum		;; display 
    object		;; filename 
    fixnum		;; bitmap 
     fixnum	;; width 
     fixnum	;; height 
    fixnum			;; x_hot 
    fixnum			;; y_hot 		     

)( fixnum "XWriteBitmapFile"))



;;;;;;;;;problems




;;(defentry fixnum (int Synchronize(

;;    fixnum		;; display 
;;    fixnum		;; onoff 

;;))()())
;;(defentry fixnum (int SetAfterFunction(

;;    fixnum		;; display 
;;    fixnum (int  ( fixnum			;; display 
;;            )		;; procedure 

;;))()())					


;;(defentry void XPeekIfEvent(

;;    fixnum		;; display 
;;    fixnum		;; event_return 
;;    fixnum (int  ( fixnum		;; display 
;;               fixnum		;; event 
;;              object		;; arg 
;;             )		;; predicate 
;;   object		;; arg 

;;)())

;;(defentry fixnum XCheckIfEvent(

;;    fixnum		;; display 
;;    fixnum		;; event_return 
;;    fixnum (int  ( fixnum			;; display 
;;               fixnum			;; event 
;;              object			;; arg 
;;             )		;; predicate 
;;   object		;; arg 

;;)())

;;(defentry void XIfEvent(

;;    fixnum		;; display 
;;    fixnum		;; event_return 
;;    fixnum (int  ( fixnum			;; display 
;;               fixnum			;; event 
;;              object			;; arg 
;;             )		;; predicate 
;;   object		;; arg 

;;)())
