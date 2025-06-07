; draw-gates.lsp                  Gordon S. Novak Jr.              20 Oct 94

; Copyright (c) 1995 Gordon S. Novak Jr. and The University of Texas at Austin.

; See the file gnu.license .

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

; Written by: Gordon S. Novak Jr., Department of Computer Sciences,
; University of Texas at Austin  78712.    novak@cs.utexas.edu

(defun draw-nand (w x y)
  (window-draw-arc-xy w (+ x 24) (+ y 16) 16 16 -90 180)
  (window-draw-circle-xy w (+ x 45) (+ y 16) 4)
  (window-draw-line-xy w (+ x 24) (+ y 32) x (+ y 32))
  (window-draw-line-xy w x (+ y 32) x y)
  (window-draw-line-xy w x y (+ x 24) y)
  (window-force-output w)) 

(setf (get 'nand 'picmenu-spec)
      '(picmenu-spec 52 32 ((in1 (0 26)) (in2 (0 6)) (out (50 16))) t
           draw-nand 9x15)) 

(defun draw-and (w x y)
  (window-draw-arc-xy w (+ x 24) (+ y 16) 16 16 -90 180)
  (window-draw-line-xy w (+ x 24) (+ y 32) x (+ y 32))
  (window-draw-line-xy w x (+ y 32) x y)
  (window-draw-line-xy w x y (+ x 24) y)
  (window-force-output w)) 

(setf (get 'and 'picmenu-spec)
      '(picmenu-spec 40 32 ((in1 (0 26)) (in2 (0 6)) (out (40 16))) t
           draw-and 9x15)) 

(defun draw-not (w x y)
  (window-draw-line-xy w x (+ y 24) (+ x 21) (+ y 12))
  (window-draw-line-xy w x y (+ x 21) (+ y 12))
  (window-draw-line-xy w x y x (+ y 24))
  (window-draw-circle-xy w (+ x 23) (+ y 12) 3)
  (window-force-output w)) 

(setf (get 'not 'picmenu-spec)
      '(picmenu-spec 27 24 ((in (0 12)) (out (27 12))) t
           draw-not 9x15)) 

(defun draw-or (w x y)
  (window-draw-arc-xy w x (- y 26) 58 58 46.4 43.6)
  (window-draw-arc-xy w x (+ y 58) 58 58 270.0 43.6)
  (window-draw-arc-xy w (- x 16) (+ y 16) 23 23 315 90)
  (window-force-output w) )

(setf (get 'or 'picmenu-spec)
      '(picmenu-spec  40 32 ((in1 (6 26)) (in2 (6 6)) (out (40 16))) t
           draw-or 9x15))

(defun draw-xor (w x y)
  (window-draw-arc-xy w (- x 16) (+ y 16) 23 23 315 90)
  (draw-or w (+ x 6) y))

(setf (get 'xor 'picmenu-spec)
      '(picmenu-spec  46 32 ((in1 (6 26)) (in2 (6 6)) (out (46 16))) t
           draw-xor 9x15))

(defun draw-nor (w x y)
  (window-draw-circle-xy w (+ x 44) (+ y 16) 4)
  (draw-or w x y))

(setf (get 'nor 'picmenu-spec)
      '(picmenu-spec  48 32 ((in1 (0 26)) (in2 (0 6)) (out (48 16))) t
           draw-nor 9x15))


(defun draw-nor2 (w x y)
  (window-draw-circle-xy w (+ x 4) (+ y 6) 4)
  (window-draw-circle-xy w (+ x 4) (+ y 26) 4)
  (draw-and w (+ x 8) y))

(setf (get 'nor2 'picmenu-spec)
      '(picmenu-spec  48 32 ((in1 (0 26)) (in2 (0 6)) (out (48 16))) t
           draw-nor2 9x15))

(defun draw-nand2 (w x y)
  (window-draw-circle-xy w (+ x 4) (+ y 6) 4)
  (window-draw-circle-xy w (+ x 4) (+ y 26) 4)
  (draw-or w (+ x 4) y))

(setf (get 'nand2 'picmenu-spec)
      '(picmenu-spec  44 32 ((in1 (0 26)) (in2 (0 6)) (out (44 16))) t
           draw-nand2 9x15))
