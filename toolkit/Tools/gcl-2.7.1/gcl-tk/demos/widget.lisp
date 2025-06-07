
(in-package "TK")
;;
;; This "script" demonstrates the various widgets provided by Tk,
;; along with many of the features of the Tk toolkit.  This file
;; only contains code to generate the main window for the
;; application, which invokes individual demonstrations.  The
;; code for the actual demonstrations is contained in separate
;; ".tcl" files is this directory, which are auto-loaded by Tcl
;; when they are needed.  To find the code for a particular
;; demo, look below for the procedure that's invoked by its menu
;; entry, then grep for the file that contains the procedure
;; definition.

(tk-do (concatenate 'string
		    "set auto_path  \"" *tk-library* "/demos " "$auto_path\""))

;; add teh current path to the auto_path so that we find the
;; .tcl demos for older demos not in new releases..
(tk-do  (concatenate 'string
		    "lappend auto_path  [file dirname "  (namestring  (truename si::*load-pathname*)) "]"))


;(setq si::*load-path* (cons (tk-conc si::*lib-directory* "gcl-tk/demos/") si::*load-path*))
(load (merge-pathnames "index.lsp" si::*load-pathname*))

(wm :title '|.| "Widget Demonstration")

;;-------------------------------------------------------
;; The code below create the main window, consisting of a
;; menu bar and a message explaining the basic operation
;; of the program.
;;-------------------------------------------------------

(frame '.menu :relief "raised" :borderwidth 1)
(message '.msg :font :Adobe-times-medium-r-normal--*-180* :relief "raised" :width 500 
:borderwidth 1 :text "This application demonstrates the widgets provided by the GCL Tk toolkit.  The menus above are organized by widget type:  each menu contains one or more demonstrations of a particular type of widget.  To invoke a demonstration, press mouse button 1 over one of the menu buttons above, drag the mouse to the desired entry in the menu, then release the mouse button.)
(To exit this demonstration, invoke the \"Quit\" entry in the \"Misc\" menu.")

(pack '.menu :side "top" :fill "x")
(pack '.msg :side "bottom" :expand "yes" :fill "both")

;;-------------------------------------------------------
;; The code below creates all the menus, which invoke procedures
;; to create particular demonstrations of various widgets.
;;-------------------------------------------------------

(menubutton '.menu.button :text "Labels/Buttons" :menu '.menu.button.m 
    :underline 7)
(menu '.menu.button.m)
(.menu.button.m :add 'command :label "Labels" :command "mkLabel" :underline 0)
(.menu.button.m :add 'command :label "Buttons" :command "mkButton" :underline 0)
(.menu.button.m :add 'command :label "Checkbuttons" :command "mkCheck" 
    :underline 0)
(.menu.button.m :add 'command :label "Radiobuttons" :command 'mkRadio
    :underline 0)
(.menu.button.m :add 'command :label "15-puzzle" :command "mkPuzzle" :underline 0)
(.menu.button.m :add 'command :label "Iconic buttons" :command "mkIcon" 
    :underline 0)

(menubutton '.menu.listbox :text "Listboxes" :menu '.menu.listbox.m 
	:underline 0)
(menu '.menu.listbox.m)
(.menu.listbox.m :add 'command :label "States" :command 'mkListbox :underline 0)
(.menu.listbox.m :add 'command :label "Colors" :command "mkListbox2" :underline 0)
(.menu.listbox.m :add 'command :label "Well-known sayings" :command "mkListbox3" 
    :underline 0)

(menubutton '.menu.entry :text "Entries" :menu '.menu.entry.m 
	:underline 0)
(menu '.menu.entry.m)
(.menu.entry.m :add 'command :label "Without scrollbars" :command 'mkentry
    :underline 4)
(.menu.entry.m :add 'command :label "With scrollbars" :command 'mkEntry2
    :underline 0)
(.menu.entry.m :add 'command :label "Simple form" :command 'mkForm 
    :underline 0)

(menubutton '.menu.text :text "Text" :menu '.menu.text.m :underline 0)
(menu '.menu.text.m)
(.menu.text.m :add 'command :label "Basic text" :command 'mkBasic 
    :underline 0)
(.menu.text.m :add 'command :label "Display styles" :command 'mkStyles 
    :underline 0)
(.menu.text.m :add 'command :label "Command bindings" :command 'mkTextBind 
    :underline 0)
(.menu.text.m :add 'command :label "Search" :command "mkTextSearch" 
    :underline 0)

(menubutton '.menu.scroll :text "Scrollbars" :menu '.menu.scroll.m 
	:underline 0)
(menu '.menu.scroll.m)
(.menu.scroll.m :add 'command :label "Vertical" :command "mkListbox2" :underline 0)
(.menu.scroll.m :add 'command :label "Horizontal" :command "mkEntry2" :underline 0)

(menubutton '.menu.scale :text "Scales" :menu '.menu.scale.m :underline 2)
(menu '.menu.scale.m)
(.menu.scale.m :add 'command :label "Vertical" :command 'mkVScale :underline 0)
(.menu.scale.m :add 'command :label "Horizontal" :command 'mkHScale :underline 0)

(menubutton '.menu.canvas :text "Canvases" :menu '.menu.canvas.m 
	:underline 0)
(menu '.menu.canvas.m)
(.menu.canvas.m :add 'command :label "Item types" :command 'mkItems :underline 0)
(.menu.canvas.m :add 'command :label "2-D plot" :command 'mkPlot :underline 0)
(.menu.canvas.m :add 'command :label "Text" :command "mkCanvText" :underline 0)
(.menu.canvas.m :add 'command :label "Arrow shapes" :command "mkArrow" :underline 0)
(.menu.canvas.m :add 'command :label "Ruler" :command 'mkRuler :underline 0)
(.menu.canvas.m :add 'command :label "Scrollable canvas" :command "mkScroll" 
    :underline 0)
(.menu.canvas.m :add 'command :label "Floor plan" :command "mkFloor" 
    :underline 0)

(menubutton '.menu.menu :text "Menus" :menu '.menu.menu.m :underline 0)
(menu '.menu.menu.m)
(.menu.menu.m :add 'command :label "Print hello" :command '(print "Hello")
    :accelerator "Control+a" :underline 6)
(bind '|.| "<Control-a>" '(print "Hello"))
(.menu.menu.m :add 'command :label "Print goodbye" :command 
   '(print "Goodbye") :accelerator "Control+b" :underline 6)
(bind '|.| "<Control-b>" '(format t "Goodbye"))
(.menu.menu.m :add 'command :label "Light blue background" 
    :command '(.msg :configure :bg "LightBlue1") :underline 0)
(.menu.menu.m :add 'command :label "Info on tear-off menus" :command "mkTear" 
    :underline 0)
(.menu.menu.m :add 'cascade :label "Check buttons" :menu '.menu.menu.m.check 
    :underline 0)
(.menu.menu.m :add 'cascade :label "Radio buttons" :menu '.menu.menu.m.radio 
    :underline 0)
(.menu.menu.m :add 'command :bitmap "@": *tk-library* :"/demos/bitmaps/pattern" 
    :command '
	(mkDialog '.pattern '(:text "The menu entry you invoked displays a bitmap rather than a text string.  Other than this, it is just like any other menu entry." :aspect 250 )))
    

(menu '.menu.menu.m.check)
(.menu.menu.m.check :add 'check :label "Oil checked" :variable 'oil)
(.menu.menu.m.check :add 'check :label "Transmission checked" :variable 'trans)
(.menu.menu.m.check :add 'check :label "Brakes checked" :variable 'brakes)
(.menu.menu.m.check :add 'check :label "Lights checked" :variable 'lights)
(.menu.menu.m.check :add 'separator)
(.menu.menu.m.check :add 'command :label "Show current values" 
    :command '(showVars '.menu.menu.dialog '(oil trans brakes lights)))
(.menu.menu.m.check :invoke 1)
(.menu.menu.m.check :invoke 3)

(menu '.menu.menu.m.radio)
(.menu.menu.m.radio :add 'radio :label "10 point" :variable 'pointSize :value 10)
(.menu.menu.m.radio :add 'radio :label "14 point" :variable 'pointSize :value 14)
(.menu.menu.m.radio :add 'radio :label "18 point" :variable 'pointSize :value 18)
(.menu.menu.m.radio :add 'radio :label "24 point" :variable 'pointSize :value 24)
(.menu.menu.m.radio :add 'radio :label "32 point" :variable 'pointSize :value 32)
(.menu.menu.m.radio :add 'sep)
(.menu.menu.m.radio :add 'radio :label "Roman" :variable 'style :value "roman")
(.menu.menu.m.radio :add 'radio :label "Bold" :variable 'style :value "bold")
(.menu.menu.m.radio :add 'radio :label "Italic" :variable 'style :value "italic")
(.menu.menu.m.radio :add 'sep)
(.menu.menu.m.radio :add 'command :label "Show current values" :command 
    '(showVars '.menu.menu.dialog '(pointSize style)))
(.menu.menu.m.radio :invoke 1)
(.menu.menu.m.radio :invoke 7)

(menubutton '.menu.misc :text "Misc" :menu '.menu.misc.m :underline 1)
(menu '.menu.misc.m)
(.menu.misc.m :add 'command :label "Modal dialog (local grab)" :command '
    (progn
      (mkDialog '.modal '(:text "This dialog box is a modal one.  It uses Tk's \"grab\" command to create a \"local grab\" on the dialog box.  The grab prevents any pointer related events from getting to any other windows in the application.  If you press the \"OK\" button below (or hit the Return key) then the dialog box will go away and things will return to normal." :aspect 250 :justify "left") '("OK" nil) '("Hi" (print "hi")))
      (wm :geometry '.modal "+10+10")
	(tk-wait-til-exists '.modal)
   ;   (tkwait :visibility '.modal)
      (grab '.modal)
      (tkwait :window '.modal)
      )
    :underline 0)
(.menu.misc.m
 :add 'command :label "Modal dialog (global grab)"
 :command
 '(progn
    (mkDialog '.modal '(:text "This is another modal dialog box.  However, in this case a \"global grab\" is used, which locks up the display so you can't talk to any windows in any applications anywhere, except for the dialog.  If you press the \"OK\" button below (or hit the Return key) then the dialog box will go away and things will return to normal." :aspect 250 :justify "left") '("OK" nil) '("Hi" (print "hi1")))

    (wm :geometry '.modal "+10+10")
    (tk-wait-til-exists '.modal)
					;(tkwait :visibility '.modal)
    (grab :set :global '.modal)
    (tkwait :window '.modal)
    )
  :underline 0)
(.menu.misc.m :add 'command :label "Built-in bitmaps" :command "mkBitmaps" 
	:underline 0)
(.menu.misc.m :add 'command :label "GC monitor"
   :command 'mkgcmonitor :underline 0)
(.menu.misc.m :add 'command :label "Quit" :command "destroy ." :underline 0)

(pack '.menu.button '.menu.listbox '.menu.entry '.menu.text '.menu.scroll 
	'.menu.scale '.menu.canvas '.menu.menu '.menu.misc :side "left")

;; Set up for keyboard-based menu traversal

(bind '|.| "<Any-FocusIn>" 
    '(progn 
       (if (and (equal |%d| "NotifyVirtual")
	     (equal |%m| "NotifyNormal"))
	(focus '.menu)
    )))

;; make the meta key do traversal bindings
(bind '.menu "<Any-M-KeyPress>" "tk_traverseToMenu %W %A")

(tk-menu-bar '.menu '.menu.button '.menu.listbox '.menu.entry '.menu.text 
	'.menu.scroll '.menu.scale '.menu.canvas '.menu.menu '.menu.misc)

;; Position a dialog box at a reasonable place on the screen.

(defun dpos (w) 
    (wm :geometry w "+60+25")
)

;; some of the widgets are tcl and need this.
(tk-do "proc dpos w {
    wm geometry $w +300+300
}")
