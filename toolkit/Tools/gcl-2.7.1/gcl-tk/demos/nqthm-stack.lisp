(in-package "TK")
;; turn on history;
;(MAINTAIN-REWRITE-PATH  t)


(defun nqthm-stack (&optional (w '.nqthm))
  (toplevel w)
  (dpos w)
  (wm :title w "Nqthm Stack Frames")
  (wm :iconname w "Nqthm Stack")
  (wm :minsize w 1 1)
  (message (conc w '.msg) :font :Adobe-times-medium-r-normal--*-180* :aspect 300
	   :text "A listbox containing the 50 states is displayed below, along with a scrollbar.  You can scan the list either using the scrollbar or by dragging in the listbox window with button 2 pressed.  Click the OK button when you've seen enough.")
  (frame (conc w '.frame) :borderwidth 10)
  (button (conc w '.ok) :text "OK" :command `(destroy ',w))
  (button (conc w '.redo) :text "Show Frames" :command
	  `(show-frames))
  (checkbutton (conc w '.rew) :text "Maintain Frames"
	  :variable '(boolean #+anci-cl cl-user::do-frames #-ansi-cl user::do-frames)
	  :command #+ansi-cl '(cl-user::MAINTAIN-REWRITE-PATH cl-user::do-frames) #-ansi-cl '(user::MAINTAIN-REWRITE-PATH user::do-frames))
  (pack (conc w '.frame) :side "top" :expand "yes" :fill "y")
  (pack (conc w '.rew)(conc w '.redo) (conc w '.ok)  :side "bottom" :fill "x")
  (scrollbar (conc w '.frame '.scroll) :relief "sunken"
	     :command
	     (tk-conc w ".frame.list yview"))
  (listbox (conc w '.frame.list) :yscroll (tk-conc w ".frame.scroll set")
	   :relief "sunken"
	   :setgrid 1)
  (pack (conc w '.frame.scroll) :side "right" :fill "y")
  (pack (conc w '.frame.list) :side "left" :expand "yes" :fill "both")
  (setq *list-box* (conc w '.frame.list)))

#+ansi-cl(in-package "CL-USER")
#-ansi-cl(in-package "USER")

(defun tk::show-frames()
  (funcall tk::*list-box* :delete 0 "end")
  (apply tk::*list-box* :insert 0
	 (sloop::sloop for i below #+ansi-cl cl-user::REWRITE-PATH-STK-PTR #-ansi-cl user::REWRITE-PATH-STK-PTR
	    do (setq tem (aref #+ansi-cl cl-user::REWRITE-PATH-STK #-ansi-cl user::REWRITE-PATH-STK i))
	    (setq tem 
	    (display-rewrite-path-token
	     (nth 0 tem)
	     (nth 3 tem)))
	    (cond ((consp tem) (setq tem (format nil "~a" tem))))
	    collect tem)))
	     


(defun display-rewrite-path-token (prog term)
  (case prog
        (ADD-EQUATIONS-TO-POT-LST
         (access linear-lemma name term))
        (REWRITE-WITH-LEMMAS
         (access rewrite-rule name term))
        ((REWRITE REWRITE-WITH-LINEAR)
         (ffn-symb term))
        ((SET-SIMPLIFY-CLAUSE-POT-LST SIMPLIFY-CLAUSE)
         "clause")
        (t (er hard (prog term)
               |Unexpected| |prog| |in| |call| |of| display-rewrite-path-token
               |on| (!ppr prog nil) |and| (!ppr term (quote |.|))))))