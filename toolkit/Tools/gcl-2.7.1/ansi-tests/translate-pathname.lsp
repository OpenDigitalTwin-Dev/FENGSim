;-*- Mode:     Lisp -*-

(in-package :cl-test)

(deftest translate-pathname.1 (translate-pathname "foobar" "foobar" "foobar") #P"foobar")
(deftest translate-pathname.2 (translate-pathname "foobar" "foobar" "foo*")   #P"foo")
(deftest translate-pathname.3 (translate-pathname "foobar" "foobar" "*")      #P"foobar")
(deftest translate-pathname.4 (translate-pathname "foobar" "foobar" "")       #P"foobar")

(deftest translate-pathname.5 (translate-pathname "foobar" "foo*r"  "foobar") #P"foobar")
(deftest translate-pathname.6 (translate-pathname "foobar" "foo*r"  "foo*")   #P"fooba")
(deftest translate-pathname.7 (translate-pathname "foobar" "foo*r"  "*")      #P"foobar")
(deftest translate-pathname.8 (translate-pathname "foobar" "foo*r"  "")       #P"foobar")

(deftest translate-pathname.9  (translate-pathname "foobar" "*"  "foobar") #P"foobar")
(deftest translate-pathname.10 (translate-pathname "foobar" "*"  "foo*")   #P"foofoobar")
(deftest translate-pathname.11 (translate-pathname "foobar" "*"  "*")      #P"foobar")
(deftest translate-pathname.12 (translate-pathname "foobar" "*"  "")       #P"foobar")

(deftest translate-pathname.13 (translate-pathname "foobar" ""  "foobar") #P"foobar")
(deftest translate-pathname.14 (translate-pathname "foobar" ""  "foo*")   #P"foofoobar")
(deftest translate-pathname.15 (translate-pathname "foobar" ""  "*")      #P"foobar")
(deftest translate-pathname.16 (translate-pathname "foobar" ""  "")       #P"foobar")

(deftest translate-pathname.17 (translate-pathname "/a/bbfb/c/d/" "/a/bbfb/c/d/" "/a/qc/c/d/")   #P"/a/qc/c/d/")
(deftest translate-pathname.18 (translate-pathname "/a/bbfb/c/d/" "/a/bbfb/c/d/" "/a/q*c*/c/d/") #P"/a/qc/c/d/")
(deftest translate-pathname.19 (translate-pathname "/a/bbfb/c/d/" "/a/bbfb/c/d/" "/a/*/c/d/")    #P"/a/c/d/")
(deftest translate-pathname.20 (translate-pathname "/a/bbfb/c/d/" "/a/bbfb/c/d/" "/a/**/d/")     #P"/a/d/")

(deftest translate-pathname.21 (translate-pathname "/a/bbfb/c/d/" "/a/b*f*/c/d/" "/a/qc/c/d/")   #P"/a/qc/c/d/")
(deftest translate-pathname.22 (translate-pathname "/a/bbfb/c/d/" "/a/b*f*/c/d/" "/a/q*c*/c/d/") #P"/a/qbcb/c/d/")
(deftest translate-pathname.23 (translate-pathname "/a/bbfb/c/d/" "/a/b*f*/c/d/" "/a/*/c/d/")    #P"/a/bbfb/c/d/")
(deftest translate-pathname.24 (translate-pathname "/a/bbfb/c/d/" "/a/b*f*/c/d/" "/a/**/d/")     #P"/a/bbfb/d/")

(deftest translate-pathname.25 (translate-pathname "/a/bbfb/c/d/" "/a/**/d/" "/a/qc/c/d/")   #P"/a/qc/c/d/")
(deftest translate-pathname.26 (translate-pathname "/a/bbfb/c/d/" "/a/**/d/" "/a/q*c*/c/d/") #P"/a/qc/c/d/")
(deftest translate-pathname.27 (translate-pathname "/a/bbfb/c/d/" "/a/**/d/" "/a/*/d/")      #P"/a/bbfb/d/")
(deftest translate-pathname.28 (translate-pathname "/a/bbfb/c/d/" "/a/**/d/" "/a/**/d/")     #P"/a/bbfb/c/d/")

(deftest translate-pathname.29 (translate-pathname "/a/bbfb/c/d/" "/a/**/d/" "a/qc/c/d/")    #P"a/qc/c/d/")
(deftest translate-pathname.30 (translate-pathname "/a/bbfb/c/d/" "/a/**/d/" "a/q*c*/c/d/")  #P"a/qc/c/d/")
(deftest translate-pathname.31 (translate-pathname "/a/bbfb/c/d/" "/a/**/d/" "a/*/d/")       #P"a/bbfb/d/")
(deftest translate-pathname.32 (translate-pathname "/a/bbfb/c/d/" "/a/**/d/" "a/**/d/")      #P"a/bbfb/c/d/")

(deftest translate-pathname.33 (translate-pathname "/a/bbfb/c/d/" "/a/bbfb/c/d/" "a")        #P"/a/bbfb/c/d/a")
(deftest translate-pathname.34 (translate-pathname "/a/bbfb/c/d/" "/a/b*f*/c/d/" "a")        #P"/a/bbfb/c/d/a")
(deftest translate-pathname.35 (translate-pathname "/a/bbfb/c/d/" "/a/*/c/d/"    "a")        #P"/a/bbfb/c/d/a")
(deftest translate-pathname.36 (translate-pathname "/a/bbfb/c/d/" "/a/**/d/"     "a")        #P"/a/bbfb/c/d/a")


