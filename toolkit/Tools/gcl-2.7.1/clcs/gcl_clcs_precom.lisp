;;; -*- Mode: Lisp; Syntax: Common-Lisp; Package: "CONDITIONS"; Base: 10 -*-

(unless (find-package :conditions)
  (make-package :conditions :use '("LISP" "PCL")))

(in-package "CONDITIONS")

#+pcl
(pcl::precompile-random-code-segments clcs)
