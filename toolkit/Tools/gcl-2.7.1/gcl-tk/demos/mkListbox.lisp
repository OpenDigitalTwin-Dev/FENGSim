(in-package "TK")

(defun mklistbox (&optional (w '.listbox))
  (toplevel w )
  (dpos w)
  (wm :title w "Listbox Demonstration (50 states)")
  (wm :iconname w "Listbox")
  (wm :minsize w 1 1)
  (message (conc w '.msg) :font :Adobe-times-medium-r-normal--*-180* :aspect 300
	   :text "A listbox containing the 50 states is displayed below, along with a scrollbar.  You can scan the list either using the scrollbar or by dragging in the listbox window with button 2 pressed.  Click the OK button when you've seen enough.")
  (frame (conc w '.frame) :borderwidth 10)
  (button (conc w '.ok) :text "OK" :command `(destroy ',w))
  (pack (conc w '.frame) :side "top" :expand "yes" :fill "y")
  (pack (conc w '.ok) :side "bottom" :fill "x")
  (scrollbar (conc w '.frame '.scroll) :relief "sunken" :command
	     (tk-conc w ".frame.list yview"))
  (listbox (conc w '.frame.list) :yscroll (tk-conc w ".frame.scroll set")
	   :relief "sunken"
	   :setgrid 1)
  (pack (conc w '.frame.scroll) :side "right" :fill "y")
  (pack (conc w '.frame.list) :side "left" :expand "yes" :fill "both")
  (funcall (conc w '.frame.list)
	   :insert 0
	   "Alabama" "Alaska" "Arizona" "Arkansas" "California" 
	   "Colorado" "Connecticut" "Delaware" "Florida" "Georgia" "Hawaii" "Idaho" "Illinois" 
	   "Indiana" "Iowa" "Kansas" "Kentucky" "Louisiana" "Maine" "Maryland" 
	   "Massachusetts" "Michigan" "Minnesota" "Mississippi" "Missouri" 
	   "Montana" "Nebraska" "Nevada" "New Hampshire" "New Jersey" "New Mexico" 
	   "New York" "North Carolina" "North Dakota" 
	   "Ohio" "Oklahoma" "Oregon" "Pennsylvania" "Rhode Island" 
	   "South Carolina" "South Dakota" 
	   "Tennessee" "Texas" "Utah" "Vermont" "Virginia" "Washington" 
	   "West Virginia" "Wisconsin" "Wyoming")
  w)
    

   
   
   
   
   
	    
