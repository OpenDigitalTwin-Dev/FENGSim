;(load "/stage/ftp/pub/novak/xgcl-4/gcl_dwtrans.lsp")
(in-package :xlib)
(load (merge-pathnames "gcl_drawtrans.lsp" *load-pathname*))
(load (merge-pathnames "gcl_editorstrans.lsp" *load-pathname*))
(load (merge-pathnames "gcl_lispservertrans.lsp" *load-pathname*))
(load (merge-pathnames "gcl_menu-settrans.lsp" *load-pathname*))
(load (merge-pathnames "gcl_dwtest.lsp" *load-pathname*))
(load (merge-pathnames "gcl_draw-gates.lsp" *load-pathname*))

(wtesta)
(wtestb)
(wtestc)
(wtestd)
(wteste)
(wtestf)
(wtestg)
(wtesth)
(wtesti)
(wtestj)
(wtestk)

(window-clear myw)
(edit-color myw)

(lisp-server)

(draw 'foo)

(window-draw-box-xy myw 48 48 204 204)
(window-edit myw 50 50 200 200 '("(edit this, ^Q to quit)" "Now is the time" "for all" "good"))

(draw-nand myw 50 50)
