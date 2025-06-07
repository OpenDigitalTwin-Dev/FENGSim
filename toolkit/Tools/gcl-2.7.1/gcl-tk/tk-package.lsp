(unless (find-package "TK")
  (make-package "TK" :use '("LISP" "SLOOP")))

(in-package "SI")
(import '(
string begin end header name
				 info-subfile
				 file tags
end-waiting
si::match-beginning si::idescribe
                              si::setup-info
                              si::autoload
			      si::idescribe
			      si::*default-info-files*
                              si::*info-paths*
			      si::*info-window*
                              si::info
			      si::get-match
			      si::print-node
			      si::offer-choices
                              si::match-end si::string-match
			      si::*case-fold-search*
				si::*current-info-data*
				si::info-data
			      si::node
				si::info-aux
			si::info-error
			      si::*tk-library*
			      si::*tk-connection*
			      si::show-info
			      si::tkconnect
                              si::*match-data*)
   "TK")


