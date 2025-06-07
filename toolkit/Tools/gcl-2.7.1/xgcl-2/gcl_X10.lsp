(in-package :XLIB)
; X10.lsp        modified by Hiep Huu Nguyen                      27 Aug 92

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


(defconstant VertexRelative		#x01		) ;; else absolute 
(defconstant VertexDontDraw		#x02		) ;; else draw 
(defconstant VertexCurved		#x04		) ;; else straight 
(defconstant VertexStartClosed	#x08		) ;; else not 
(defconstant VertexEndClosed		#x10		) ;; else not 
