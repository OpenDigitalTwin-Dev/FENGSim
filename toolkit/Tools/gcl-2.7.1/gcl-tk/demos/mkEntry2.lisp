;;# mkEntry2 -
;;
;; Create a top-level window that displays a bunch of entries with
;; scrollbars.
;;
;; Arguments:
;;    w -	Name to use for new top-level window.
(IN-package "TK")
(defun mkEntry2 (&optional (w '.e2)) 
    (if (winfo :exists w :return 'boolean) (destroy w))
    (toplevel w)
    (dpos w)
    (wm :title w "Entry Demonstration")
    (wm :iconname w "Entries")
    (message (conc w '.msg) :font :Adobe-times-medium-r-normal--*-180* :aspect 200 
	    :text "Three different entries are displayed below, with a scrollbar for each entry.  You can add characters by pointing, clicking and typing.  You can delete by selecting and typing Control-d.  Backspace, Control-h, and Delete may be typed to erase the character just before the insertion point, Control-W erases the word just before the insertion point, and Control-u clears the entry.  For entries that are too large to fit in the window all at once, you can scan through the entries using the scrollbars, or by dragging with mouse button 2 pressed.  Click the \"OK\" button when you've seen enough.")
    (frame (conc w '.frame) :borderwidth 10)
    (button (conc w '.ok) :text "OK" :command (tk-conc "destroy " w))
    (pack (conc w '.msg) (conc w '.frame) (conc w '.ok) :side "top" :fill "both")

    (entry (conc w '.frame.e1) :relief "sunken" :xscrollcommand (tk-conc w ".frame.s1 set"))
    (scrollbar (conc w '.frame.s1) :relief "sunken" :orient "horiz" :command 
	    (tk-conc w ".frame.e1 xview"))
    (frame (conc w '.frame.f1) :width 20 :height 10)
    (entry (conc w '.frame.e2) :relief "sunken" :xscrollcommand (tk-conc w ".frame.s2 set"))
    (scrollbar (conc w '.frame.s2) :relief "sunken" :orient "horiz" :command 
	    (tk-conc w ".frame.e2 xview"))
    (frame (conc w '.frame.f2) :width 20 :height 10)
    (entry (conc w '.frame.e3) :relief "sunken" :xscrollcommand (tk-conc w ".frame.s3 set"))
    (scrollbar (conc w '.frame.s3) :relief "sunken" :orient "horiz" :command 
	    (tk-conc w ".frame.e3 xview"))
    (pack (conc w '.frame.e1) (conc w '.frame.s1) (conc w '.frame.f1) (conc w '.frame.e2) (conc w '.frame.s2) 
	    (conc w '.frame.f2) (conc w '.frame.e3) (conc w '.frame.s3) :side "top" :fill "x")

    (funcall (conc w '.frame.e1) :insert 0 "Initial value")
    (funcall (conc w '.frame.e2) :insert 'end "This entry contains a long value, much too long ")
    (funcall (conc w '.frame.e2) :insert 'end "to fit in the window at one time, so long in fact ")
    (funcall (conc w '.frame.e2) :insert 'end "that you'll have to scan or scroll to see the end.")
)
