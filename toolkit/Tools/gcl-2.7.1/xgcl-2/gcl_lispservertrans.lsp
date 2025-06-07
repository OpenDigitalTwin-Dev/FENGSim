; 27 Jan 2006 14:38:08 CST
; This program is free software; you can redistribute it and/or modify
; it under the terms of the GNU General Public License as published by
; the Free Software Foundation; either version 2 of the License, or
; (at your option) any later version.

; This program is distributed in the hope that it will be useful,
; but WITHOUT ANY WARRANTY; without even the implied warranty of
; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
; GNU General Public License for more details.

; You should have received a copy of the GNU General Public License
; along with this program; if not, write to the Free Software
; Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA


(DEFVAR *WIO-WINDOW* NIL)

(DEFVAR *WIO-WINDOW-WIDTH* 500)

(DEFVAR *WIO-WINDOW-HEIGHT* 300)

(DEFVAR *WIO-MENU-SET* NIL)

(DEFVAR *WIO-FONT* '8X13)

(DEFVAR *WIO-WINDOW*)
(SETF (GET '*WIO-WINDOW* 'GLISPGLOBALVAR) T)
(SETF (GET '*WIO-WINDOW* 'GLISPGLOBALVARTYPE) 'WINDOW)
(DEFVAR *WIO-WINDOW-WIDTH*)
(SETF (GET '*WIO-WINDOW-WIDTH* 'GLISPGLOBALVAR) T)
(SETF (GET '*WIO-WINDOW-WIDTH* 'GLISPGLOBALVARTYPE) 'INTEGER)
(DEFVAR *WIO-WINDOW-HEIGHT*)
(SETF (GET '*WIO-WINDOW-HEIGHT* 'GLISPGLOBALVAR) T)
(SETF (GET '*WIO-WINDOW-HEIGHT* 'GLISPGLOBALVARTYPE) 'INTEGER)
(DEFVAR *WIO-MENU-SET*)
(SETF (GET '*WIO-MENU-SET* 'GLISPGLOBALVAR) T)
(SETF (GET '*WIO-MENU-SET* 'GLISPGLOBALVARTYPE) 'MENU-SET)


(DEFMACRO WHILE (TEST &REST FORMS)
  (LIST* 'LOOP (LIST 'UNLESS TEST '(RETURN)) FORMS))

(SETF (GET 'WIO-WINDOW 'GLFNRESULTTYPE) 'WINDOW)

(DEFUN WIO-WINDOW (&OPTIONAL TITLE WIDTH HEIGHT (POSX 0) (POSY 0) FONT)
  (IF WIDTH (SETQ *WIO-WINDOW-WIDTH* WIDTH))
  (IF HEIGHT (SETQ *WIO-WINDOW-HEIGHT* HEIGHT))
  (OR *WIO-WINDOW*
      (SETQ *WIO-WINDOW*
            (WINDOW-CREATE *WIO-WINDOW-WIDTH* *WIO-WINDOW-HEIGHT* TITLE
                NIL POSX POSY FONT))))

(DEFUN WIO-INIT-MENUS (W COMMANDS)
  (LET ()
    (WINDOW-CLEAR W)
    (SETQ *WIO-MENU-SET* (MENU-SET-CREATE W NIL))
    (MENU-SET-ADD-MENU *WIO-MENU-SET* 'COMMAND NIL "Commands" COMMANDS
        (LIST 0 0))
    (MENU-SET-ADJUST *WIO-MENU-SET* 'COMMAND 'TOP NIL 2)
    (MENU-SET-ADJUST *WIO-MENU-SET* 'COMMAND 'RIGHT NIL 2)))

(DEFUN LISP-SERVER ()
  (LET (W INPUTM DONE SEL (REDRAW T) STR RESULT)
    (SETQ W (WIO-WINDOW "Lisp Server"))
    (WINDOW-OPEN W)
    (WINDOW-CLEAR W)
    (WINDOW-SET-FONT W *WIO-FONT*)
    (WIO-INIT-MENUS W '(("Quit" . QUIT)))
    (WINDOW-PRINT-LINES W
        '("Click mouse in the input box, then enter"
          "a Lisp expression followed by Return." ""
          "Input:   e.g.  (+ 3 4)  or  (sqrt 2)")
        10 (+ -20 *WIO-WINDOW-HEIGHT*))
    (WINDOW-PRINTAT-XY W "Result:" 10 (+ -150 *WIO-WINDOW-HEIGHT*))
    (SETQ INPUTM
          (TEXTMENU-CREATE (+ -100 *WIO-WINDOW-WIDTH*) 30 NIL W 20
              (+ -110 *WIO-WINDOW-HEIGHT*) T T '9X15 T))
    (MENU-SET-ADD-ITEM *WIO-MENU-SET* 'INPUT NIL INPUTM)
    (WHILE (NOT DONE)
           (SETQ SEL (MENU-SET-SELECT *WIO-MENU-SET* REDRAW))
           (SETQ REDRAW NIL)
           (CASE (CADR SEL)
             (COMMAND (CASE (CAR SEL) (QUIT (SETQ DONE T))))
             (INPUT (SETQ STR (CAR SEL))
                    (SETQ RESULT
                          (CATCH 'ERROR
                            (EVAL (SAFE-READ-FROM-STRING STR))))
                    (WINDOW-ERASE-AREA-XY W 20 2
                        (+ -20 *WIO-WINDOW-WIDTH*)
                        (+ -160 *WIO-WINDOW-HEIGHT*))
                    (WINDOW-PRINT-LINE W
                        (WRITE-TO-STRING RESULT :PRETTY T) 20
                        (+ -170 *WIO-WINDOW-HEIGHT*)))))
    (WINDOW-CLOSE W)))

(DEFUN SAFE-READ-FROM-STRING (STR)
  (IF (AND (STRINGP STR) (> (LENGTH STR) 0))
      (READ-FROM-STRING STR NIL 'READ-ERROR)))

(DEFUN COMPILE-LISPSERVER ()
  (GLCOMPFILES *DIRECTORY* '("glisp/vector.lsp")
      '("glisp/lispserver.lsp") "glisp/lispservertrans.lsp"
      "glisp/gpl.txt"))

(DEFUN COMPILE-LISPSERVERB ()
  (GLCOMPFILES *DIRECTORY*
      '("glisp/vector.lsp" "X/dwindow.lsp" "X/dwnoopen.lsp")
      '("glisp/lispserver.lsp") "glisp/lispservertrans.lsp"
      "glisp/gpl.txt"))
