(in-package :XLIB)
; Xinit.lsp         Hiep Huu Nguyen       27 Aug 92; GSN 07 Mar 95

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

;;a word about Xakcl:
;;Since Xakcl is a direct translation of the X library in C to lisp to a
;;large extent.  it would be beneficial to use a X 11 version 4, manual
;;in order to look up functions.  the only unique functions of Xakcl are those
;;that involove manipulating C structs. all functions involved in creating
;;a C struct in X starts with a 'make' followed by the structure name.  All
;;functions involved in getting a field of a C struct strats with the
;;name of the C struct followed by the name of the field.  the
;;parameters it excepts is the variable containing the structure.  All
;;functions to set a field of a C struct starts with 'set' followed by
;;the C struct name followed by the field name.  these functions accept
;;as parameter, the variable containing the struct and the value to be
;;put in the field.

;;;;
;;contents of this file:
;;;;
;;this files has examples of initializing the display, screen,
;;root-window, pixel value, gc, and colormap.
;;;;
;;gives an example of opening windows, setting size's and sizehints for
;;the window manager getting drawbles' geometry
;;;;
;;drawing lines , drawing in color, changing line, attributes
;;;;
;;tracking the mouse and handling events and manipulating the event
;;queue
;;;;
;;there is also some basic text handling stuff
;;;;

;;globals
(defvar  *default-display* )
(defvar *default-screen* )
(defvar *default-colormap*)
(defvar  *root-window* )
(defvar  *black-pixel* ) 
(defvar  *white-pixel* )
(defvar *default-size-hints* (make-XsizeHints) )
(defvar *default-GC* )
(defvar *default-event* (make-XEvent))
(defvar *pos-x* 10)
(defvar *pos-y* 20)
(defvar *win-width* 225)
(defvar *win-height* 400)
(defvar *border-width* 1)
(defvar *root-return* (int-array 1))
(defvar *x-return* (int-array 1))
(defvar *y-return* (int-array 1) )
(defvar *width-return* (int-array 1))
(defvar *height-return* (int-array 1))
(defvar *border-width-return* (int-array 1))
(defvar *depth-return* (int-array 1))
(defvar *GC-Values* (make-XGCValues))

;;an example window
(defvar a-window)


;;;;;;;;;;;;;;;;;;;;;; 
;;this function initializes all variables needed by most applications.
;;it uses all defaults which is inherited from the root window, and
;;screen.

(defun Xinit()
  (setq *default-display* (XOpenDisplay (get-c-string "")))
  (setq *default-screen* (XdefaultScreen *default-display*))
  (setq *root-window* (XRootWindow *default-display* *default-screen*))
  (setq *black-pixel* (XBlackPixel *default-display*  
				   *default-screen*))
  (setq *white-pixel* (XWhitePixel *default-display*  
				   *default-screen*))
  (setq *default-GC* (XDefaultGC  *default-display*  *default-screen*))
  (setq *default-colormap* ( XDefaultColormap *default-display* *default-screen*))
  (Xflush *default-display* ))




;;;;;;;;;;;;;;;;;;;;;;
;;This is an example of creating a window.  This function takes care of
;;positioning, size and other attributes of the window.

(defun open-window(&key (pos-x  *pos-x* ) (pos-y  *pos-y*) (win-width *win-width*) 
			(win-height *win-height* ) 
			(border-width *border-width*) (window-name "My Window") 
			(icon-name  "My Icon"))
;;create the window

  (let (( a-window (XCreateSimpleWindow
		    *default-display*  *root-window*
		    pos-x pos-y win-width win-height border-width  *black-pixel*  *white-pixel*))) 

;; all children of the root window needs a XSizeHints to tell the window manager 
;; how to position it, etc

    (set-Xsizehints-x *default-size-hints* pos-x)
    (set-xsizehints-y *default-size-hints* pos-y)
    (set-xsizehints-width *default-size-hints* win-width)
    (set-xsizehints-height *default-size-hints* win-height)
    (set-xsizehints-flags *default-size-hints* (+ Psize Pposition))
    (XsetStandardProperties  *default-display*  a-window (get-c-string window-name)
			     (get-c-string icon-name) none 0 0 *default-size-hints*)

;; the events or input a window can have are set with Xselectinput
;;    (Xselectinput *default-display* a-window 
;;		  (+ ButtonpressMask PointerMotionMask ExposureMask))

;; the window needs to be mapped
    (Xmapwindow *default-display* a-window)

;;the X server needs to have the output buffer sent to it before it can
;;process requests.  this is accomplished with XFlush or functions that
;;read and manipulate the event queue.  remember to do this after
;;operations that won't be calling an eventhandling function

    (Xflush *default-display* )

;;after flushing the request buffer the X server draws window as requested

    a-window))


