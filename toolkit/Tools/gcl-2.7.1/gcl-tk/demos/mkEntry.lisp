;;# mkEntry w
;;
;; Create a top-level window that displays a bunch of entries.
;;
;; Arguments:
;;    w -	Name to use for new top-level window.

(in-package "TK")
(defun mkEntry (&optional (w '.e1)) 
    (if (winfo :exists w :return 'boolean) (destroy w))
    (toplevel w)
    (dpos w)
    (wm :title w "Entry Demonstration")
    (wm :iconname w "Entries")
    (message (conc w '.msg) :font :Adobe-times-medium-r-normal--*-180* :aspect 200 
	    :text "Three different entries are displayed below.  You can add characters by pointing, clicking and typing.  The usual emacs control characters control editing.   Thus control-b back a char, control-f forward a char, control-a begin line, control-k kill rest of line, control-y yank.   For entries that are too large to fit in the window all at once, you can scan through the entries by dragging with mouse button 2 pressed.  Click the \"OK\" button when you've seen enough.")
    (frame (conc w '.frame) :borderwidth 10)
    (button (conc w '.ok) :text "OK" :command (tk-conc "destroy " w))
    (pack (conc w '.msg) (conc w '.frame) (conc w '.ok) :side "top" :fill "both")

    (entry (conc w '.frame.e1) :relief "sunken")
    (entry (conc w '.frame.e2) :relief "sunken")
    (entry (conc w '.frame.e3) :relief "sunken")
    (pack (conc w '.frame.e1) (conc w '.frame.e2) (conc w '.frame.e3) :side "top" :pady 5 :fill "x")

    (funcall (conc w '.frame.e1) :insert 0 "Initial value")
    (funcall (conc w '.frame.e2) :insert "end" "This entry contains a long value, much too long ")
    (funcall (conc w '.frame.e2) :insert "end" "to fit in the window at one time, so long in fact ")
    (funcall (conc w '.frame.e2) :insert "end" "that you'll have to scan or scroll to see the end.")
)
