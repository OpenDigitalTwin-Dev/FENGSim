(in-package :XLIB)
; keysymdef.lsp        modified by Hiep Huu Nguyen                27 Aug 92

; Copyright (c) 1994 Hiep Huu Nguyen and The University of Texas at Austin.

; See the files gnu.license and dec.copyright .

; This program is free software; you can redistribute it and/or modify
; it under the terms of the GNU General Public License as published by
; the Free Software Foundation; either version 1, or (at your option)
; any later version.

; This program is distributed in the hope that it will be useful,
; but WITHOUT ANY WARRANTY; without even the implied warranty of
; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
; GNU General Public License for more details.

; You should have received a copy of the GNU General Public License
; along with this program; if not, write to the Free Software
; Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

; Some of the files that interface to the Xlib are adapted from DEC/MIT files.
; See the file dec.copyright for details.

;; $XConsortium: keysymdef.h,v 1.13 89/12/12 16:23:30 rws Exp $ 

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defconstant XK_VoidSymbol		#xFFFFFF	;; void symbol 

;;#ifdef XK_MISCELLANY
;;
 ; TTY Functions, cleverly chosen to map to ascii, for convenience of
 ; programming, but could have been arbitrary at the cost of lookup
 ; tables in client code.
 

)(defconstant XK_BackSpace		#xFF08	;; back space, back char 
)(defconstant XK_Tab			#xFF09
)(defconstant XK_Linefeed		#xFF0A	;; Linefeed, LF 
)(defconstant XK_Clear		#xFF0B
)(defconstant XK_Return		#xFF0D	;; Return, enter 
)(defconstant XK_Pause		#xFF13	;; Pause, hold 
)(defconstant XK_Scroll_Lock		#xFF14
)(defconstant XK_Escape		#xFF1B
)(defconstant XK_Delete		#xFFFF	;; Delete, rubout 



;; International & multi-key character composition 

)(defconstant XK_Multi_key		#xFF20  ;; Multi-key character compose 

;; Japanese keyboard support 

)(defconstant XK_Kanji		#xFF21	;; Kanji, Kanji convert 
)(defconstant XK_Muhenkan		#xFF22  ;; Cancel Conversion 
)(defconstant XK_Henkan_Mode		#xFF23  ;; Start/Stop Conversion 
)(defconstant XK_Henkan		#xFF23  ;; Alias for Henkan_Mode 
)(defconstant XK_Romaji		#xFF24  ;; to Romaji 
)(defconstant XK_Hiragana		#xFF25  ;; to Hiragana 
)(defconstant XK_Katakana		#xFF26  ;; to Katakana 
)(defconstant XK_Hiragana_Katakana	#xFF27  ;; Hiragana/Katakana toggle 
)(defconstant XK_Zenkaku		#xFF28  ;; to Zenkaku 
)(defconstant XK_Hankaku		#xFF29  ;; to Hankaku 
)(defconstant XK_Zenkaku_Hankaku	#xFF2A  ;; Zenkaku/Hankaku toggle 
)(defconstant XK_Touroku		#xFF2B  ;; Add to Dictionary 
)(defconstant XK_Massyo		#xFF2C  ;; Delete from Dictionary 
)(defconstant XK_Kana_Lock		#xFF2D  ;; Kana Lock 
)(defconstant XK_Kana_Shift		#xFF2E  ;; Kana Shift 
)(defconstant XK_Eisu_Shift		#xFF2F  ;; Alphanumeric Shift 
)(defconstant XK_Eisu_toggle		#xFF30  ;; Alphanumeric toggle 

;; Cursor control & motion 

)(defconstant XK_Home			#xFF50
)(defconstant XK_Left			#xFF51	;; Move left, left arrow 
)(defconstant XK_Up			#xFF52	;; Move up, up arrow 
)(defconstant XK_Right		#xFF53	;; Move right, right arrow 
)(defconstant XK_Down			#xFF54	;; Move down, down arrow 
)(defconstant XK_Prior		#xFF55	;; Prior, previous 
)(defconstant XK_Next			#xFF56	;; Next 
)(defconstant XK_End			#xFF57	;; EOL 
)(defconstant XK_Begin		#xFF58	;; BOL 


;; Misc Functions 

)(defconstant XK_Select		#xFF60	;; Select, mark 
)(defconstant XK_Print		#xFF61
)(defconstant XK_Execute		#xFF62	;; Execute, run, do 
)(defconstant XK_Insert		#xFF63	;; Insert, insert here 
)(defconstant XK_Undo			#xFF65	;; Undo, oops 
)(defconstant XK_Redo			#xFF66	;; redo, again 
)(defconstant XK_Menu			#xFF67
)(defconstant XK_Find			#xFF68	;; Find, search 
)(defconstant XK_Cancel		#xFF69	;; Cancel, stop, abort, exit 
)(defconstant XK_Help			#xFF6A	;; Help, ? 
)(defconstant XK_Break		#xFF6B
)(defconstant XK_Mode_switch		#xFF7E	;; Character set switch 
)(defconstant XK_script_switch        #xFF7E  ;; Alias for mode_switch 
)(defconstant XK_Num_Lock		#xFF7F

;; Keypad Functions, keypad numbers cleverly chosen to map to ascii 

)(defconstant XK_KP_Space		#xFF80	;; space 
)(defconstant XK_KP_Tab		#xFF89
)(defconstant XK_KP_Enter		#xFF8D	;; enter 
)(defconstant XK_KP_F1		#xFF91	;; PF1, KP_A, ... 
)(defconstant XK_KP_F2		#xFF92
)(defconstant XK_KP_F3		#xFF93
)(defconstant XK_KP_F4		#xFF94
)(defconstant XK_KP_Equal		#xFFBD	;; equals 
)(defconstant XK_KP_Multiply		#xFFAA
)(defconstant XK_KP_Add		#xFFAB
)(defconstant XK_KP_Separator		#xFFAC	;; separator, often comma 
)(defconstant XK_KP_Subtract		#xFFAD
)(defconstant XK_KP_Decimal		#xFFAE
)(defconstant XK_KP_Divide		#xFFAF
)(defconstant XK_KP_0			#xFFB0
)(defconstant XK_KP_1			#xFFB1
)(defconstant XK_KP_2			#xFFB2
)(defconstant XK_KP_3			#xFFB3
)(defconstant XK_KP_4			#xFFB4
)(defconstant XK_KP_5			#xFFB5
)(defconstant XK_KP_6			#xFFB6
)(defconstant XK_KP_7			#xFFB7
)(defconstant XK_KP_8			#xFFB8
)(defconstant XK_KP_9			#xFFB9



;;
 ; Auxiliary Functions; note the duplicate definitions for left and right
 ; function keys;  Sun keyboards and a few other manufactures have such
 ; function key groups on the left and/or right sides of the keyboard.
 ; We've not found a keyboard with more than 35 function keys total.
 

)(defconstant XK_F1			#xFFBE
)(defconstant XK_F2			#xFFBF
)(defconstant XK_F3			#xFFC0
)(defconstant XK_F4			#xFFC1
)(defconstant XK_F5			#xFFC2
)(defconstant XK_F6			#xFFC3
)(defconstant XK_F7			#xFFC4
)(defconstant XK_F8			#xFFC5
)(defconstant XK_F9			#xFFC6
)(defconstant XK_F10			#xFFC7
)(defconstant XK_F11			#xFFC8
)(defconstant XK_L1			#xFFC8
)(defconstant XK_F12			#xFFC9
)(defconstant XK_L2			#xFFC9
)(defconstant XK_F13			#xFFCA
)(defconstant XK_L3			#xFFCA
)(defconstant XK_F14			#xFFCB
)(defconstant XK_L4			#xFFCB
)(defconstant XK_F15			#xFFCC
)(defconstant XK_L5			#xFFCC
)(defconstant XK_F16			#xFFCD
)(defconstant XK_L6			#xFFCD
)(defconstant XK_F17			#xFFCE
)(defconstant XK_L7			#xFFCE
)(defconstant XK_F18			#xFFCF
)(defconstant XK_L8			#xFFCF
)(defconstant XK_F19			#xFFD0
)(defconstant XK_L9			#xFFD0
)(defconstant XK_F20			#xFFD1
)(defconstant XK_L10			#xFFD1
)(defconstant XK_F21			#xFFD2
)(defconstant XK_R1			#xFFD2
)(defconstant XK_F22			#xFFD3
)(defconstant XK_R2			#xFFD3
)(defconstant XK_F23			#xFFD4
)(defconstant XK_R3			#xFFD4
)(defconstant XK_F24			#xFFD5
)(defconstant XK_R4			#xFFD5
)(defconstant XK_F25			#xFFD6
)(defconstant XK_R5			#xFFD6
)(defconstant XK_F26			#xFFD7
)(defconstant XK_R6			#xFFD7
)(defconstant XK_F27			#xFFD8
)(defconstant XK_R7			#xFFD8
)(defconstant XK_F28			#xFFD9
)(defconstant XK_R8			#xFFD9
)(defconstant XK_F29			#xFFDA
)(defconstant XK_R9			#xFFDA
)(defconstant XK_F30			#xFFDB
)(defconstant XK_R10			#xFFDB
)(defconstant XK_F31			#xFFDC
)(defconstant XK_R11			#xFFDC
)(defconstant XK_F32			#xFFDD
)(defconstant XK_R12			#xFFDD
)(defconstant XK_R13			#xFFDE
)(defconstant XK_F33			#xFFDE
)(defconstant XK_F34			#xFFDF
)(defconstant XK_R14			#xFFDF
)(defconstant XK_F35			#xFFE0
)(defconstant XK_R15			#xFFE0

;; Modifiers 

)(defconstant XK_Shift_L		#xFFE1	;; Left shift 
)(defconstant XK_Shift_R		#xFFE2	;; Right shift 
)(defconstant XK_Control_L		#xFFE3	;; Left control 
)(defconstant XK_Control_R		#xFFE4	;; Right control 
)(defconstant XK_Caps_Lock		#xFFE5	;; Caps lock 
)(defconstant XK_Shift_Lock		#xFFE6	;; Shift lock 

)(defconstant XK_Meta_L		#xFFE7	;; Left meta 
)(defconstant XK_Meta_R		#xFFE8	;; Right meta 
)(defconstant XK_Alt_L		#xFFE9	;; Left alt 
)(defconstant XK_Alt_R		#xFFEA	;; Right alt 
)(defconstant XK_Super_L		#xFFEB	;; Left super 
)(defconstant XK_Super_R		#xFFEC	;; Right super 
)(defconstant XK_Hyper_L		#xFFED	;; Left hyper 
)(defconstant XK_Hyper_R		#xFFEE	;; Right hyper 
;;#endif ;; XK_MISCELLANY 

;;
 ;  Latin 1
 ;  Byte 3 = 0
 
;;ifdef XK_LATIN1
)(defconstant XK_space               #x020
)(defconstant XK_exclam              #x021
)(defconstant XK_quotedbl            #x022
)(defconstant XK_numbersign          #x023
)(defconstant XK_dollar              #x024
)(defconstant XK_percent             #x025
)(defconstant XK_ampersand           #x026
)(defconstant XK_apostrophe          #x027
)(defconstant XK_quoteright          #x027	;; deprecated 
)(defconstant XK_parenleft           #x028
)(defconstant XK_parenright          #x029
)(defconstant XK_asterisk            #x02a
)(defconstant XK_plus                #x02b
)(defconstant XK_comma               #x02c
)(defconstant XK_minus               #x02d
)(defconstant XK_period              #x02e
)(defconstant XK_slash               #x02f
)(defconstant XK_0                   #x030
)(defconstant XK_1                   #x031
)(defconstant XK_2                   #x032
)(defconstant XK_3                   #x033
)(defconstant XK_4                   #x034
)(defconstant XK_5                   #x035
)(defconstant XK_6                   #x036
)(defconstant XK_7                   #x037
)(defconstant XK_8                   #x038
)(defconstant XK_9                   #x039
)(defconstant XK_colon               #x03a
)(defconstant XK_semicolon           #x03b
)(defconstant XK_less                #x03c
)(defconstant XK_equal               #x03d
)(defconstant XK_greater             #x03e
)(defconstant XK_question            #x03f
)(defconstant XK_at                  #x040
)(defconstant XK_A                   #x041
)(defconstant XK_B                   #x042
)(defconstant XK_C                   #x043
)(defconstant XK_D                   #x044
)(defconstant XK_E                   #x045
)(defconstant XK_F                   #x046
)(defconstant XK_G                   #x047
)(defconstant XK_H                   #x048
)(defconstant XK_I                   #x049
)(defconstant XK_J                   #x04a
)(defconstant XK_K                   #x04b
)(defconstant XK_L                   #x04c
)(defconstant XK_M                   #x04d
)(defconstant XK_N                   #x04e
)(defconstant XK_O                   #x04f
)(defconstant XK_P                   #x050
)(defconstant XK_Q                   #x051
)(defconstant XK_R                   #x052
)(defconstant XK_S                   #x053
)(defconstant XK_T                   #x054
)(defconstant XK_U                   #x055
)(defconstant XK_V                   #x056
)(defconstant XK_W                   #x057
)(defconstant XK_X                   #x058
)(defconstant XK_Y                   #x059
)(defconstant XK_Z                   #x05a
)(defconstant XK_bracketleft         #x05b
)(defconstant XK_backslash           #x05c
)(defconstant XK_bracketright        #x05d
)(defconstant XK_asciicircum         #x05e
)(defconstant XK_underscore          #x05f
)(defconstant XK_grave               #x060
)(defconstant XK_quoteleft           #x060	;; deprecated 
)(defconstant XK_a                   #x061
)(defconstant XK_b                   #x062
)(defconstant XK_c                   #x063
)(defconstant XK_d                   #x064
)(defconstant XK_e                   #x065
)(defconstant XK_f                   #x066
)(defconstant XK_g                   #x067
)(defconstant XK_h                   #x068
)(defconstant XK_i                   #x069
)(defconstant XK_j                   #x06a
)(defconstant XK_k                   #x06b
)(defconstant XK_l                   #x06c
)(defconstant XK_m                   #x06d
)(defconstant XK_n                   #x06e
)(defconstant XK_o                   #x06f
)(defconstant XK_p                   #x070
)(defconstant XK_q                   #x071
)(defconstant XK_r                   #x072
)(defconstant XK_s                   #x073
)(defconstant XK_t                   #x074
)(defconstant XK_u                   #x075
)(defconstant XK_v                   #x076
)(defconstant XK_w                   #x077
)(defconstant XK_x                   #x078
)(defconstant XK_y                   #x079
)(defconstant XK_z                   #x07a
)(defconstant XK_braceleft           #x07b
)(defconstant XK_bar                 #x07c
)(defconstant XK_braceright          #x07d
)(defconstant XK_asciitilde          #x07e

)(defconstant XK_nobreakspace        #x0a0
)(defconstant XK_exclamdown          #x0a1
)(defconstant XK_cent        	       #x0a2
)(defconstant XK_sterling            #x0a3
)(defconstant XK_currency            #x0a4
)(defconstant XK_yen                 #x0a5
)(defconstant XK_brokenbar           #x0a6
)(defconstant XK_section             #x0a7
)(defconstant XK_diaeresis           #x0a8
)(defconstant XK_copyright           #x0a9
)(defconstant XK_ordfeminine         #x0aa
)(defconstant XK_guillemotleft       #x0ab	;; left angle quotation mark 
)(defconstant XK_notsign             #x0ac
)(defconstant XK_hyphen              #x0ad
)(defconstant XK_registered          #x0ae
)(defconstant XK_macron              #x0af
)(defconstant XK_degree              #x0b0
)(defconstant XK_plusminus           #x0b1
)(defconstant XK_twosuperior         #x0b2
)(defconstant XK_threesuperior       #x0b3
)(defconstant XK_acute               #x0b4
)(defconstant XK_mu                  #x0b5
)(defconstant XK_paragraph           #x0b6
)(defconstant XK_periodcentered      #x0b7
)(defconstant XK_cedilla             #x0b8
)(defconstant XK_onesuperior         #x0b9
)(defconstant XK_masculine           #x0ba
)(defconstant XK_guillemotright      #x0bb	;; right angle quotation mark 
)(defconstant XK_onequarter          #x0bc
)(defconstant XK_onehalf             #x0bd
)(defconstant XK_threequarters       #x0be
)(defconstant XK_questiondown        #x0bf
)(defconstant XK_Agrave              #x0c0
)(defconstant XK_Aacute              #x0c1
)(defconstant XK_Acircumflex         #x0c2
)(defconstant XK_Atilde              #x0c3
)(defconstant XK_Adiaeresis          #x0c4
)(defconstant XK_Aring               #x0c5
)(defconstant XK_AE                  #x0c6
)(defconstant XK_Ccedilla            #x0c7
)(defconstant XK_Egrave              #x0c8
)(defconstant XK_Eacute              #x0c9
)(defconstant XK_Ecircumflex         #x0ca
)(defconstant XK_Ediaeresis          #x0cb
)(defconstant XK_Igrave              #x0cc
)(defconstant XK_Iacute              #x0cd
)(defconstant XK_Icircumflex         #x0ce
)(defconstant XK_Idiaeresis          #x0cf
)(defconstant XK_ETH                 #x0d0
)(defconstant XK_Eth                 #x0d0	;; deprecated 
)(defconstant XK_Ntilde              #x0d1
)(defconstant XK_Ograve              #x0d2
)(defconstant XK_Oacute              #x0d3
)(defconstant XK_Ocircumflex         #x0d4
)(defconstant XK_Otilde              #x0d5
)(defconstant XK_Odiaeresis          #x0d6
)(defconstant XK_multiply            #x0d7
)(defconstant XK_Ooblique            #x0d8
)(defconstant XK_Ugrave              #x0d9
)(defconstant XK_Uacute              #x0da
)(defconstant XK_Ucircumflex         #x0db
)(defconstant XK_Udiaeresis          #x0dc
)(defconstant XK_Yacute              #x0dd
)(defconstant XK_THORN               #x0de
)(defconstant XK_Thorn               #x0de	;; deprecated 
)(defconstant XK_ssharp              #x0df
)(defconstant XK_agrave              #x0e0
)(defconstant XK_aacute              #x0e1
)(defconstant XK_acircumflex         #x0e2
)(defconstant XK_atilde              #x0e3
)(defconstant XK_adiaeresis          #x0e4
)(defconstant XK_aring               #x0e5
)(defconstant XK_ae                  #x0e6
)(defconstant XK_ccedilla            #x0e7
)(defconstant XK_egrave              #x0e8
)(defconstant XK_eacute              #x0e9
)(defconstant XK_ecircumflex         #x0ea
)(defconstant XK_ediaeresis          #x0eb
)(defconstant XK_igrave              #x0ec
)(defconstant XK_iacute              #x0ed
)(defconstant XK_icircumflex         #x0ee
)(defconstant XK_idiaeresis          #x0ef
)(defconstant XK_eth                 #x0f0
)(defconstant XK_ntilde              #x0f1
)(defconstant XK_ograve              #x0f2
)(defconstant XK_oacute              #x0f3
)(defconstant XK_ocircumflex         #x0f4
)(defconstant XK_otilde              #x0f5
)(defconstant XK_odiaeresis          #x0f6
)(defconstant XK_division            #x0f7
)(defconstant XK_oslash              #x0f8
)(defconstant XK_ugrave              #x0f9
)(defconstant XK_uacute              #x0fa
)(defconstant XK_ucircumflex         #x0fb
)(defconstant XK_udiaeresis          #x0fc
)(defconstant XK_yacute              #x0fd
)(defconstant XK_thorn               #x0fe
)(defconstant XK_ydiaeresis          #x0ff
;;endif ;; XK_LATIN1 

;;
 ;   Latin 2
 ;   Byte 3 = 1
 

;;ifdef XK_LATIN2
)(defconstant XK_Aogonek             #x1a1
)(defconstant XK_breve               #x1a2
)(defconstant XK_Lstroke             #x1a3
)(defconstant XK_Lcaron              #x1a5
)(defconstant XK_Sacute              #x1a6
)(defconstant XK_Scaron              #x1a9
)(defconstant XK_Scedilla            #x1aa
)(defconstant XK_Tcaron              #x1ab
)(defconstant XK_Zacute              #x1ac
)(defconstant XK_Zcaron              #x1ae
)(defconstant XK_Zabovedot           #x1af
)(defconstant XK_aogonek             #x1b1
)(defconstant XK_ogonek              #x1b2
)(defconstant XK_lstroke             #x1b3
)(defconstant XK_lcaron              #x1b5
)(defconstant XK_sacute              #x1b6
)(defconstant XK_caron               #x1b7
)(defconstant XK_scaron              #x1b9
)(defconstant XK_scedilla            #x1ba
)(defconstant XK_tcaron              #x1bb
)(defconstant XK_zacute              #x1bc
)(defconstant XK_doubleacute         #x1bd
)(defconstant XK_zcaron              #x1be
)(defconstant XK_zabovedot           #x1bf
)(defconstant XK_Racute              #x1c0
)(defconstant XK_Abreve              #x1c3
)(defconstant XK_Lacute              #x1c5
)(defconstant XK_Cacute              #x1c6
)(defconstant XK_Ccaron              #x1c8
)(defconstant XK_Eogonek             #x1ca
)(defconstant XK_Ecaron              #x1cc
)(defconstant XK_Dcaron              #x1cf
)(defconstant XK_Dstroke             #x1d0
)(defconstant XK_Nacute              #x1d1
)(defconstant XK_Ncaron              #x1d2
)(defconstant XK_Odoubleacute        #x1d5
)(defconstant XK_Rcaron              #x1d8
)(defconstant XK_Uring               #x1d9
)(defconstant XK_Udoubleacute        #x1db
)(defconstant XK_Tcedilla            #x1de
)(defconstant XK_racute              #x1e0
)(defconstant XK_abreve              #x1e3
)(defconstant XK_lacute              #x1e5
)(defconstant XK_cacute              #x1e6
)(defconstant XK_ccaron              #x1e8
)(defconstant XK_eogonek             #x1ea
)(defconstant XK_ecaron              #x1ec
)(defconstant XK_dcaron              #x1ef
)(defconstant XK_dstroke             #x1f0
)(defconstant XK_nacute              #x1f1
)(defconstant XK_ncaron              #x1f2
)(defconstant XK_odoubleacute        #x1f5
)(defconstant XK_udoubleacute        #x1fb
)(defconstant XK_rcaron              #x1f8
)(defconstant XK_uring               #x1f9
)(defconstant XK_tcedilla            #x1fe
)(defconstant XK_abovedot            #x1ff
;;endif ;; XK_LATIN2 

;;
 ;   Latin 3
 ;   Byte 3 = 2
 

;;ifdef XK_LATIN3
)(defconstant XK_Hstroke             #x2a1
)(defconstant XK_Hcircumflex         #x2a6
)(defconstant XK_Iabovedot           #x2a9
)(defconstant XK_Gbreve              #x2ab
)(defconstant XK_Jcircumflex         #x2ac
)(defconstant XK_hstroke             #x2b1
)(defconstant XK_hcircumflex         #x2b6
)(defconstant XK_idotless            #x2b9
)(defconstant XK_gbreve              #x2bb
)(defconstant XK_jcircumflex         #x2bc
)(defconstant XK_Cabovedot           #x2c5
)(defconstant XK_Ccircumflex         #x2c6
)(defconstant XK_Gabovedot           #x2d5
)(defconstant XK_Gcircumflex         #x2d8
)(defconstant XK_Ubreve              #x2dd
)(defconstant XK_Scircumflex         #x2de
)(defconstant XK_cabovedot           #x2e5
)(defconstant XK_ccircumflex         #x2e6
)(defconstant XK_gabovedot           #x2f5
)(defconstant XK_gcircumflex         #x2f8
)(defconstant XK_ubreve              #x2fd
)(defconstant XK_scircumflex         #x2fe
;;endif ;; XK_LATIN3 


;;
 ;   Latin 4
 ;   Byte 3 = 3
 

;;ifdef XK_LATIN4
)(defconstant XK_kra                 #x3a2
)(defconstant XK_kappa               #x3a2	;; deprecated 
)(defconstant XK_Rcedilla            #x3a3
)(defconstant XK_Itilde              #x3a5
)(defconstant XK_Lcedilla            #x3a6
)(defconstant XK_Emacron             #x3aa
)(defconstant XK_Gcedilla            #x3ab
)(defconstant XK_Tslash              #x3ac
)(defconstant XK_rcedilla            #x3b3
)(defconstant XK_itilde              #x3b5
)(defconstant XK_lcedilla            #x3b6
)(defconstant XK_emacron             #x3ba
)(defconstant XK_gcedilla            #x3bb
)(defconstant XK_tslash              #x3bc
)(defconstant XK_ENG                 #x3bd
)(defconstant XK_eng                 #x3bf
)(defconstant XK_Amacron             #x3c0
)(defconstant XK_Iogonek             #x3c7
)(defconstant XK_Eabovedot           #x3cc
)(defconstant XK_Imacron             #x3cf
)(defconstant XK_Ncedilla            #x3d1
)(defconstant XK_Omacron             #x3d2
)(defconstant XK_Kcedilla            #x3d3
)(defconstant XK_Uogonek             #x3d9
)(defconstant XK_Utilde              #x3dd
)(defconstant XK_Umacron             #x3de
)(defconstant XK_amacron             #x3e0
)(defconstant XK_iogonek             #x3e7
)(defconstant XK_eabovedot           #x3ec
)(defconstant XK_imacron             #x3ef
)(defconstant XK_ncedilla            #x3f1
)(defconstant XK_omacron             #x3f2
)(defconstant XK_kcedilla            #x3f3
)(defconstant XK_uogonek             #x3f9
)(defconstant XK_utilde              #x3fd
)(defconstant XK_umacron             #x3fe
;;endif ;; XK_LATIN4 

;;
 ; Katakana
 ; Byte 3 = 4
 

;;ifdef XK_KATAKANA
)(defconstant XK_overline				       #x47e
)(defconstant XK_kana_fullstop                               #x4a1
)(defconstant XK_kana_openingbracket                         #x4a2
)(defconstant XK_kana_closingbracket                         #x4a3
)(defconstant XK_kana_comma                                  #x4a4
)(defconstant XK_kana_conjunctive                            #x4a5
)(defconstant XK_kana_middledot                              #x4a5  ;; deprecated 
)(defconstant XK_kana_WO                                     #x4a6
)(defconstant XK_kana_a                                      #x4a7
)(defconstant XK_kana_i                                      #x4a8
)(defconstant XK_kana_u                                      #x4a9
)(defconstant XK_kana_e                                      #x4aa
)(defconstant XK_kana_o                                      #x4ab
)(defconstant XK_kana_ya                                     #x4ac
)(defconstant XK_kana_yu                                     #x4ad
)(defconstant XK_kana_yo                                     #x4ae
)(defconstant XK_kana_tsu                                    #x4af
)(defconstant XK_kana_tu                                     #x4af  ;; deprecated 
)(defconstant XK_prolongedsound                              #x4b0
)(defconstant XK_kana_A                                      #x4b1
)(defconstant XK_kana_I                                      #x4b2
)(defconstant XK_kana_U                                      #x4b3
)(defconstant XK_kana_E                                      #x4b4
)(defconstant XK_kana_O                                      #x4b5
)(defconstant XK_kana_KA                                     #x4b6
)(defconstant XK_kana_KI                                     #x4b7
)(defconstant XK_kana_KU                                     #x4b8
)(defconstant XK_kana_KE                                     #x4b9
)(defconstant XK_kana_KO                                     #x4ba
)(defconstant XK_kana_SA                                     #x4bb
)(defconstant XK_kana_SHI                                    #x4bc
)(defconstant XK_kana_SU                                     #x4bd
)(defconstant XK_kana_SE                                     #x4be
)(defconstant XK_kana_SO                                     #x4bf
)(defconstant XK_kana_TA                                     #x4c0
)(defconstant XK_kana_CHI                                    #x4c1
)(defconstant XK_kana_TI                                     #x4c1  ;; deprecated 
)(defconstant XK_kana_TSU                                    #x4c2
)(defconstant XK_kana_TU                                     #x4c2  ;; deprecated 
)(defconstant XK_kana_TE                                     #x4c3
)(defconstant XK_kana_TO                                     #x4c4
)(defconstant XK_kana_NA                                     #x4c5
)(defconstant XK_kana_NI                                     #x4c6
)(defconstant XK_kana_NU                                     #x4c7
)(defconstant XK_kana_NE                                     #x4c8
)(defconstant XK_kana_NO                                     #x4c9
)(defconstant XK_kana_HA                                     #x4ca
)(defconstant XK_kana_HI                                     #x4cb
)(defconstant XK_kana_FU                                     #x4cc
)(defconstant XK_kana_HU                                     #x4cc  ;; deprecated 
)(defconstant XK_kana_HE                                     #x4cd
)(defconstant XK_kana_HO                                     #x4ce
)(defconstant XK_kana_MA                                     #x4cf
)(defconstant XK_kana_MI                                     #x4d0
)(defconstant XK_kana_MU                                     #x4d1
)(defconstant XK_kana_ME                                     #x4d2
)(defconstant XK_kana_MO                                     #x4d3
)(defconstant XK_kana_YA                                     #x4d4
)(defconstant XK_kana_YU                                     #x4d5
)(defconstant XK_kana_YO                                     #x4d6
)(defconstant XK_kana_RA                                     #x4d7
)(defconstant XK_kana_RI                                     #x4d8
)(defconstant XK_kana_RU                                     #x4d9
)(defconstant XK_kana_RE                                     #x4da
)(defconstant XK_kana_RO                                     #x4db
)(defconstant XK_kana_WA                                     #x4dc
)(defconstant XK_kana_N                                      #x4dd
)(defconstant XK_voicedsound                                 #x4de
)(defconstant XK_semivoicedsound                             #x4df
)(defconstant XK_kana_switch          #xFF7E  ;; Alias for mode_switch 
;;endif ;; XK_KATAKANA 

;;
 ;  Arabic
 ;  Byte 3 = 5
 

;;ifdef XK_ARABIC
)(defconstant XK_Arabic_comma                                #x5ac
)(defconstant XK_Arabic_semicolon                            #x5bb
)(defconstant XK_Arabic_question_mark                        #x5bf
)(defconstant XK_Arabic_hamza                                #x5c1
)(defconstant XK_Arabic_maddaonalef                          #x5c2
)(defconstant XK_Arabic_hamzaonalef                          #x5c3
)(defconstant XK_Arabic_hamzaonwaw                           #x5c4
)(defconstant XK_Arabic_hamzaunderalef                       #x5c5
)(defconstant XK_Arabic_hamzaonyeh                           #x5c6
)(defconstant XK_Arabic_alef                                 #x5c7
)(defconstant XK_Arabic_beh                                  #x5c8
)(defconstant XK_Arabic_tehmarbuta                           #x5c9
)(defconstant XK_Arabic_teh                                  #x5ca
)(defconstant XK_Arabic_theh                                 #x5cb
)(defconstant XK_Arabic_jeem                                 #x5cc
)(defconstant XK_Arabic_hah                                  #x5cd
)(defconstant XK_Arabic_khah                                 #x5ce
)(defconstant XK_Arabic_dal                                  #x5cf
)(defconstant XK_Arabic_thal                                 #x5d0
)(defconstant XK_Arabic_ra                                   #x5d1
)(defconstant XK_Arabic_zain                                 #x5d2
)(defconstant XK_Arabic_seen                                 #x5d3
)(defconstant XK_Arabic_sheen                                #x5d4
)(defconstant XK_Arabic_sad                                  #x5d5
)(defconstant XK_Arabic_dad                                  #x5d6
)(defconstant XK_Arabic_tah                                  #x5d7
)(defconstant XK_Arabic_zah                                  #x5d8
)(defconstant XK_Arabic_ain                                  #x5d9
)(defconstant XK_Arabic_ghain                                #x5da
)(defconstant XK_Arabic_tatweel                              #x5e0
)(defconstant XK_Arabic_feh                                  #x5e1
)(defconstant XK_Arabic_qaf                                  #x5e2
)(defconstant XK_Arabic_kaf                                  #x5e3
)(defconstant XK_Arabic_lam                                  #x5e4
)(defconstant XK_Arabic_meem                                 #x5e5
)(defconstant XK_Arabic_noon                                 #x5e6
)(defconstant XK_Arabic_ha                                   #x5e7
)(defconstant XK_Arabic_heh                                  #x5e7  ;; deprecated 
)(defconstant XK_Arabic_waw                                  #x5e8
)(defconstant XK_Arabic_alefmaksura                          #x5e9
)(defconstant XK_Arabic_yeh                                  #x5ea
)(defconstant XK_Arabic_fathatan                             #x5eb
)(defconstant XK_Arabic_dammatan                             #x5ec
)(defconstant XK_Arabic_kasratan                             #x5ed
)(defconstant XK_Arabic_fatha                                #x5ee
)(defconstant XK_Arabic_damma                                #x5ef
)(defconstant XK_Arabic_kasra                                #x5f0
)(defconstant XK_Arabic_shadda                               #x5f1
)(defconstant XK_Arabic_sukun                                #x5f2
)(defconstant XK_Arabic_switch        #xFF7E  ;; Alias for mode_switch 
;;endif ;; XK_ARABIC 

;;
 ; Cyrillic
 ; Byte 3 = 6
 
;;ifdef XK_CYRILLIC
)(defconstant XK_Serbian_dje                                 #x6a1
)(defconstant XK_Macedonia_gje                               #x6a2
)(defconstant XK_Cyrillic_io                                 #x6a3
)(defconstant XK_Ukrainian_ie                                #x6a4
)(defconstant XK_Ukranian_je                                 #x6a4  ;; deprecated 
)(defconstant XK_Macedonia_dse                               #x6a5
)(defconstant XK_Ukrainian_i                                 #x6a6
)(defconstant XK_Ukranian_i                                  #x6a6  ;; deprecated 
)(defconstant XK_Ukrainian_yi                                #x6a7
)(defconstant XK_Ukranian_yi                                 #x6a7  ;; deprecated 
)(defconstant XK_Cyrillic_je                                 #x6a8
)(defconstant XK_Serbian_je                                  #x6a8  ;; deprecated 
)(defconstant XK_Cyrillic_lje                                #x6a9
)(defconstant XK_Serbian_lje                                 #x6a9  ;; deprecated 
)(defconstant XK_Cyrillic_nje                                #x6aa
)(defconstant XK_Serbian_nje                                 #x6aa  ;; deprecated 
)(defconstant XK_Serbian_tshe                                #x6ab
)(defconstant XK_Macedonia_kje                               #x6ac
)(defconstant XK_Byelorussian_shortu                         #x6ae
)(defconstant XK_Cyrillic_dzhe                               #x6af
)(defconstant XK_Serbian_dze                                 #x6af  ;; deprecated 
)(defconstant XK_numerosign                                  #x6b0
)(defconstant XK_Serbian_DJE                                 #x6b1
)(defconstant XK_Macedonia_GJE                               #x6b2
)(defconstant XK_Cyrillic_IO                                 #x6b3
)(defconstant XK_Ukrainian_IE                                #x6b4
)(defconstant XK_Ukranian_JE                                 #x6b4  ;; deprecated 
)(defconstant XK_Macedonia_DSE                               #x6b5
)(defconstant XK_Ukrainian_I                                 #x6b6
)(defconstant XK_Ukranian_I                                  #x6b6  ;; deprecated 
)(defconstant XK_Ukrainian_YI                                #x6b7
)(defconstant XK_Ukranian_YI                                 #x6b7  ;; deprecated 
)(defconstant XK_Cyrillic_JE                                 #x6b8
)(defconstant XK_Serbian_JE                                  #x6b8  ;; deprecated 
)(defconstant XK_Cyrillic_LJE                                #x6b9
)(defconstant XK_Serbian_LJE                                 #x6b9  ;; deprecated 
)(defconstant XK_Cyrillic_NJE                                #x6ba
)(defconstant XK_Serbian_NJE                                 #x6ba  ;; deprecated 
)(defconstant XK_Serbian_TSHE                                #x6bb
)(defconstant XK_Macedonia_KJE                               #x6bc
)(defconstant XK_Byelorussian_SHORTU                         #x6be
)(defconstant XK_Cyrillic_DZHE                               #x6bf
)(defconstant XK_Serbian_DZE                                 #x6bf  ;; deprecated 
)(defconstant XK_Cyrillic_yu                                 #x6c0
)(defconstant XK_Cyrillic_a                                  #x6c1
)(defconstant XK_Cyrillic_be                                 #x6c2
)(defconstant XK_Cyrillic_tse                                #x6c3
)(defconstant XK_Cyrillic_de                                 #x6c4
)(defconstant XK_Cyrillic_ie                                 #x6c5
)(defconstant XK_Cyrillic_ef                                 #x6c6
)(defconstant XK_Cyrillic_ghe                                #x6c7
)(defconstant XK_Cyrillic_ha                                 #x6c8
)(defconstant XK_Cyrillic_i                                  #x6c9
)(defconstant XK_Cyrillic_shorti                             #x6ca
)(defconstant XK_Cyrillic_ka                                 #x6cb
)(defconstant XK_Cyrillic_el                                 #x6cc
)(defconstant XK_Cyrillic_em                                 #x6cd
)(defconstant XK_Cyrillic_en                                 #x6ce
)(defconstant XK_Cyrillic_o                                  #x6cf
)(defconstant XK_Cyrillic_pe                                 #x6d0
)(defconstant XK_Cyrillic_ya                                 #x6d1
)(defconstant XK_Cyrillic_er                                 #x6d2
)(defconstant XK_Cyrillic_es                                 #x6d3
)(defconstant XK_Cyrillic_te                                 #x6d4
)(defconstant XK_Cyrillic_u                                  #x6d5
)(defconstant XK_Cyrillic_zhe                                #x6d6
)(defconstant XK_Cyrillic_ve                                 #x6d7
)(defconstant XK_Cyrillic_softsign                           #x6d8
)(defconstant XK_Cyrillic_yeru                               #x6d9
)(defconstant XK_Cyrillic_ze                                 #x6da
)(defconstant XK_Cyrillic_sha                                #x6db
)(defconstant XK_Cyrillic_e                                  #x6dc
)(defconstant XK_Cyrillic_shcha                              #x6dd
)(defconstant XK_Cyrillic_che                                #x6de
)(defconstant XK_Cyrillic_hardsign                           #x6df
)(defconstant XK_Cyrillic_YU                                 #x6e0
)(defconstant XK_Cyrillic_A                                  #x6e1
)(defconstant XK_Cyrillic_BE                                 #x6e2
)(defconstant XK_Cyrillic_TSE                                #x6e3
)(defconstant XK_Cyrillic_DE                                 #x6e4
)(defconstant XK_Cyrillic_IE                                 #x6e5
)(defconstant XK_Cyrillic_EF                                 #x6e6
)(defconstant XK_Cyrillic_GHE                                #x6e7
)(defconstant XK_Cyrillic_HA                                 #x6e8
)(defconstant XK_Cyrillic_I                                  #x6e9
)(defconstant XK_Cyrillic_SHORTI                             #x6ea
)(defconstant XK_Cyrillic_KA                                 #x6eb
)(defconstant XK_Cyrillic_EL                                 #x6ec
)(defconstant XK_Cyrillic_EM                                 #x6ed
)(defconstant XK_Cyrillic_EN                                 #x6ee
)(defconstant XK_Cyrillic_O                                  #x6ef
)(defconstant XK_Cyrillic_PE                                 #x6f0
)(defconstant XK_Cyrillic_YA                                 #x6f1
)(defconstant XK_Cyrillic_ER                                 #x6f2
)(defconstant XK_Cyrillic_ES                                 #x6f3
)(defconstant XK_Cyrillic_TE                                 #x6f4
)(defconstant XK_Cyrillic_U                                  #x6f5
)(defconstant XK_Cyrillic_ZHE                                #x6f6
)(defconstant XK_Cyrillic_VE                                 #x6f7
)(defconstant XK_Cyrillic_SOFTSIGN                           #x6f8
)(defconstant XK_Cyrillic_YERU                               #x6f9
)(defconstant XK_Cyrillic_ZE                                 #x6fa
)(defconstant XK_Cyrillic_SHA                                #x6fb
)(defconstant XK_Cyrillic_E                                  #x6fc
)(defconstant XK_Cyrillic_SHCHA                              #x6fd
)(defconstant XK_Cyrillic_CHE                                #x6fe
)(defconstant XK_Cyrillic_HARDSIGN                           #x6ff
;;endif ;; XK_CYRILLIC 

;;
 ; Greek
 ; Byte 3 = 7
 

;;ifdef XK_GREEK
)(defconstant XK_Greek_ALPHAaccent                           #x7a1
)(defconstant XK_Greek_EPSILONaccent                         #x7a2
)(defconstant XK_Greek_ETAaccent                             #x7a3
)(defconstant XK_Greek_IOTAaccent                            #x7a4
)(defconstant XK_Greek_IOTAdiaeresis                         #x7a5
)(defconstant XK_Greek_OMICRONaccent                         #x7a7
)(defconstant XK_Greek_UPSILONaccent                         #x7a8
)(defconstant XK_Greek_UPSILONdieresis                       #x7a9
)(defconstant XK_Greek_OMEGAaccent                           #x7ab
)(defconstant XK_Greek_accentdieresis                        #x7ae
)(defconstant XK_Greek_horizbar                              #x7af
)(defconstant XK_Greek_alphaaccent                           #x7b1
)(defconstant XK_Greek_epsilonaccent                         #x7b2
)(defconstant XK_Greek_etaaccent                             #x7b3
)(defconstant XK_Greek_iotaaccent                            #x7b4
)(defconstant XK_Greek_iotadieresis                          #x7b5
)(defconstant XK_Greek_iotaaccentdieresis                    #x7b6
)(defconstant XK_Greek_omicronaccent                         #x7b7
)(defconstant XK_Greek_upsilonaccent                         #x7b8
)(defconstant XK_Greek_upsilondieresis                       #x7b9
)(defconstant XK_Greek_upsilonaccentdieresis                 #x7ba
)(defconstant XK_Greek_omegaaccent                           #x7bb
)(defconstant XK_Greek_ALPHA                                 #x7c1
)(defconstant XK_Greek_BETA                                  #x7c2
)(defconstant XK_Greek_GAMMA                                 #x7c3
)(defconstant XK_Greek_DELTA                                 #x7c4
)(defconstant XK_Greek_EPSILON                               #x7c5
)(defconstant XK_Greek_ZETA                                  #x7c6
)(defconstant XK_Greek_ETA                                   #x7c7
)(defconstant XK_Greek_THETA                                 #x7c8
)(defconstant XK_Greek_IOTA                                  #x7c9
)(defconstant XK_Greek_KAPPA                                 #x7ca
)(defconstant XK_Greek_LAMDA                                 #x7cb
)(defconstant XK_Greek_LAMBDA                                #x7cb
)(defconstant XK_Greek_MU                                    #x7cc
)(defconstant XK_Greek_NU                                    #x7cd
)(defconstant XK_Greek_XI                                    #x7ce
)(defconstant XK_Greek_OMICRON                               #x7cf
)(defconstant XK_Greek_PI                                    #x7d0
)(defconstant XK_Greek_RHO                                   #x7d1
)(defconstant XK_Greek_SIGMA                                 #x7d2
)(defconstant XK_Greek_TAU                                   #x7d4
)(defconstant XK_Greek_UPSILON                               #x7d5
)(defconstant XK_Greek_PHI                                   #x7d6
)(defconstant XK_Greek_CHI                                   #x7d7
)(defconstant XK_Greek_PSI                                   #x7d8
)(defconstant XK_Greek_OMEGA                                 #x7d9
)(defconstant XK_Greek_alpha                                 #x7e1
)(defconstant XK_Greek_beta                                  #x7e2
)(defconstant XK_Greek_gamma                                 #x7e3
)(defconstant XK_Greek_delta                                 #x7e4
)(defconstant XK_Greek_epsilon                               #x7e5
)(defconstant XK_Greek_zeta                                  #x7e6
)(defconstant XK_Greek_eta                                   #x7e7
)(defconstant XK_Greek_theta                                 #x7e8
)(defconstant XK_Greek_iota                                  #x7e9
)(defconstant XK_Greek_kappa                                 #x7ea
)(defconstant XK_Greek_lamda                                 #x7eb
)(defconstant XK_Greek_lambda                                #x7eb
)(defconstant XK_Greek_mu                                    #x7ec
)(defconstant XK_Greek_nu                                    #x7ed
)(defconstant XK_Greek_xi                                    #x7ee
)(defconstant XK_Greek_omicron                               #x7ef
)(defconstant XK_Greek_pi                                    #x7f0
)(defconstant XK_Greek_rho                                   #x7f1
)(defconstant XK_Greek_sigma                                 #x7f2
)(defconstant XK_Greek_finalsmallsigma                       #x7f3
)(defconstant XK_Greek_tau                                   #x7f4
)(defconstant XK_Greek_upsilon                               #x7f5
)(defconstant XK_Greek_phi                                   #x7f6
)(defconstant XK_Greek_chi                                   #x7f7
)(defconstant XK_Greek_psi                                   #x7f8
)(defconstant XK_Greek_omega                                 #x7f9
)(defconstant XK_Greek_switch         #xFF7E  ;; Alias for mode_switch 
;;endif ;; XK_GREEK 

;;
 ; Technical
 ; Byte 3 = 8
 

;;ifdef XK_TECHNICAL
)(defconstant XK_leftradical                                 #x8a1
)(defconstant XK_topleftradical                              #x8a2
)(defconstant XK_horizconnector                              #x8a3
)(defconstant XK_topintegral                                 #x8a4
)(defconstant XK_botintegral                                 #x8a5
)(defconstant XK_vertconnector                               #x8a6
)(defconstant XK_topleftsqbracket                            #x8a7
)(defconstant XK_botleftsqbracket                            #x8a8
)(defconstant XK_toprightsqbracket                           #x8a9
)(defconstant XK_botrightsqbracket                           #x8aa
)(defconstant XK_topleftparens                               #x8ab
)(defconstant XK_botleftparens                               #x8ac
)(defconstant XK_toprightparens                              #x8ad
)(defconstant XK_botrightparens                              #x8ae
)(defconstant XK_leftmiddlecurlybrace                        #x8af
)(defconstant XK_rightmiddlecurlybrace                       #x8b0
)(defconstant XK_topleftsummation                            #x8b1
)(defconstant XK_botleftsummation                            #x8b2
)(defconstant XK_topvertsummationconnector                   #x8b3
)(defconstant XK_botvertsummationconnector                   #x8b4
)(defconstant XK_toprightsummation                           #x8b5
)(defconstant XK_botrightsummation                           #x8b6
)(defconstant XK_rightmiddlesummation                        #x8b7
)(defconstant XK_lessthanequal                               #x8bc
)(defconstant XK_notequal                                    #x8bd
)(defconstant XK_greaterthanequal                            #x8be
)(defconstant XK_integral                                    #x8bf
)(defconstant XK_therefore                                   #x8c0
)(defconstant XK_variation                                   #x8c1
)(defconstant XK_infinity                                    #x8c2
)(defconstant XK_nabla                                       #x8c5
)(defconstant XK_approximate                                 #x8c8
)(defconstant XK_similarequal                                #x8c9
)(defconstant XK_ifonlyif                                    #x8cd
)(defconstant XK_implies                                     #x8ce
)(defconstant XK_identical                                   #x8cf
)(defconstant XK_radical                                     #x8d6
)(defconstant XK_includedin                                  #x8da
)(defconstant XK_includes                                    #x8db
)(defconstant XK_intersection                                #x8dc
)(defconstant XK_union                                       #x8dd
)(defconstant XK_logicaland                                  #x8de
)(defconstant XK_logicalor                                   #x8df
)(defconstant XK_partialderivative                           #x8ef
)(defconstant XK_function                                    #x8f6
)(defconstant XK_leftarrow                                   #x8fb
)(defconstant XK_uparrow                                     #x8fc
)(defconstant XK_rightarrow                                  #x8fd
)(defconstant XK_downarrow                                   #x8fe
;;endif ;; XK_TECHNICAL 

;;
 ;  Special
 ;  Byte 3 = 9
 

;;ifdef XK_SPECIAL
)(defconstant XK_blank                                       #x9df
)(defconstant XK_soliddiamond                                #x9e0
)(defconstant XK_checkerboard                                #x9e1
)(defconstant XK_ht                                          #x9e2
)(defconstant XK_ff                                          #x9e3
)(defconstant XK_cr                                          #x9e4
)(defconstant XK_lf                                          #x9e5
)(defconstant XK_nl                                          #x9e8
)(defconstant XK_vt                                          #x9e9
)(defconstant XK_lowrightcorner                              #x9ea
)(defconstant XK_uprightcorner                               #x9eb
)(defconstant XK_upleftcorner                                #x9ec
)(defconstant XK_lowleftcorner                               #x9ed
)(defconstant XK_crossinglines                               #x9ee
)(defconstant XK_horizlinescan1                              #x9ef
)(defconstant XK_horizlinescan3                              #x9f0
)(defconstant XK_horizlinescan5                              #x9f1
)(defconstant XK_horizlinescan7                              #x9f2
)(defconstant XK_horizlinescan9                              #x9f3
)(defconstant XK_leftt                                       #x9f4
)(defconstant XK_rightt                                      #x9f5
)(defconstant XK_bott                                        #x9f6
)(defconstant XK_topt                                        #x9f7
)(defconstant XK_vertbar                                     #x9f8
;;endif ;; XK_SPECIAL 

;;
 ;  Publishing
 ;  Byte 3 = a
 

;;ifdef XK_PUBLISHING
)(defconstant XK_emspace                                     #xaa1
)(defconstant XK_enspace                                     #xaa2
)(defconstant XK_em3space                                    #xaa3
)(defconstant XK_em4space                                    #xaa4
)(defconstant XK_digitspace                                  #xaa5
)(defconstant XK_punctspace                                  #xaa6
)(defconstant XK_thinspace                                   #xaa7
)(defconstant XK_hairspace                                   #xaa8
)(defconstant XK_emdash                                      #xaa9
)(defconstant XK_endash                                      #xaaa
)(defconstant XK_signifblank                                 #xaac
)(defconstant XK_ellipsis                                    #xaae
)(defconstant XK_doubbaselinedot                             #xaaf
)(defconstant XK_onethird                                    #xab0
)(defconstant XK_twothirds                                   #xab1
)(defconstant XK_onefifth                                    #xab2
)(defconstant XK_twofifths                                   #xab3
)(defconstant XK_threefifths                                 #xab4
)(defconstant XK_fourfifths                                  #xab5
)(defconstant XK_onesixth                                    #xab6
)(defconstant XK_fivesixths                                  #xab7
)(defconstant XK_careof                                      #xab8
)(defconstant XK_figdash                                     #xabb
)(defconstant XK_leftanglebracket                            #xabc
)(defconstant XK_decimalpoint                                #xabd
)(defconstant XK_rightanglebracket                           #xabe
)(defconstant XK_marker                                      #xabf
)(defconstant XK_oneeighth                                   #xac3
)(defconstant XK_threeeighths                                #xac4
)(defconstant XK_fiveeighths                                 #xac5
)(defconstant XK_seveneighths                                #xac6
)(defconstant XK_trademark                                   #xac9
)(defconstant XK_signaturemark                               #xaca
)(defconstant XK_trademarkincircle                           #xacb
)(defconstant XK_leftopentriangle                            #xacc
)(defconstant XK_rightopentriangle                           #xacd
)(defconstant XK_emopencircle                                #xace
)(defconstant XK_emopenrectangle                             #xacf
)(defconstant XK_leftsinglequotemark                         #xad0
)(defconstant XK_rightsinglequotemark                        #xad1
)(defconstant XK_leftdoublequotemark                         #xad2
)(defconstant XK_rightdoublequotemark                        #xad3
)(defconstant XK_prescription                                #xad4
)(defconstant XK_minutes                                     #xad6
)(defconstant XK_seconds                                     #xad7
)(defconstant XK_latincross                                  #xad9
)(defconstant XK_hexagram                                    #xada
)(defconstant XK_filledrectbullet                            #xadb
)(defconstant XK_filledlefttribullet                         #xadc
)(defconstant XK_filledrighttribullet                        #xadd
)(defconstant XK_emfilledcircle                              #xade
)(defconstant XK_emfilledrect                                #xadf
)(defconstant XK_enopencircbullet                            #xae0
)(defconstant XK_enopensquarebullet                          #xae1
)(defconstant XK_openrectbullet                              #xae2
)(defconstant XK_opentribulletup                             #xae3
)(defconstant XK_opentribulletdown                           #xae4
)(defconstant XK_openstar                                    #xae5
)(defconstant XK_enfilledcircbullet                          #xae6
)(defconstant XK_enfilledsqbullet                            #xae7
)(defconstant XK_filledtribulletup                           #xae8
)(defconstant XK_filledtribulletdown                         #xae9
)(defconstant XK_leftpointer                                 #xaea
)(defconstant XK_rightpointer                                #xaeb
)(defconstant XK_club                                        #xaec
)(defconstant XK_diamond                                     #xaed
)(defconstant XK_heart                                       #xaee
)(defconstant XK_maltesecross                                #xaf0
)(defconstant XK_dagger                                      #xaf1
)(defconstant XK_doubledagger                                #xaf2
)(defconstant XK_checkmark                                   #xaf3
)(defconstant XK_ballotcross                                 #xaf4
)(defconstant XK_musicalsharp                                #xaf5
)(defconstant XK_musicalflat                                 #xaf6
)(defconstant XK_malesymbol                                  #xaf7
)(defconstant XK_femalesymbol                                #xaf8
)(defconstant XK_telephone                                   #xaf9
)(defconstant XK_telephonerecorder                           #xafa
)(defconstant XK_phonographcopyright                         #xafb
)(defconstant XK_caret                                       #xafc
)(defconstant XK_singlelowquotemark                          #xafd
)(defconstant XK_doublelowquotemark                          #xafe
)(defconstant XK_cursor                                      #xaff
;;endif ;; XK_PUBLISHING 

;;
 ;  APL
 ;  Byte 3 = b
 

;;ifdef XK_APL
)(defconstant XK_leftcaret                                   #xba3
)(defconstant XK_rightcaret                                  #xba6
)(defconstant XK_downcaret                                   #xba8
)(defconstant XK_upcaret                                     #xba9
)(defconstant XK_overbar                                     #xbc0
)(defconstant XK_downtack                                    #xbc2
)(defconstant XK_upshoe                                      #xbc3
)(defconstant XK_downstile                                   #xbc4
)(defconstant XK_underbar                                    #xbc6
)(defconstant XK_jot                                         #xbca
)(defconstant XK_quad                                        #xbcc
)(defconstant XK_uptack                                      #xbce
)(defconstant XK_circle                                      #xbcf
)(defconstant XK_upstile                                     #xbd3
)(defconstant XK_downshoe                                    #xbd6
)(defconstant XK_rightshoe                                   #xbd8
)(defconstant XK_leftshoe                                    #xbda
)(defconstant XK_lefttack                                    #xbdc
)(defconstant XK_righttack                                   #xbfc
;;endif ;; XK_APL 

;;
 ; Hebrew
 ; Byte 3 = c
 

;;ifdef XK_HEBREW
)(defconstant XK_hebrew_doublelowline                        #xcdf
)(defconstant XK_hebrew_aleph                                #xce0
)(defconstant XK_hebrew_bet                                  #xce1
)(defconstant XK_hebrew_beth                                 #xce1  ;; deprecated 
)(defconstant XK_hebrew_gimel                                #xce2
)(defconstant XK_hebrew_gimmel                               #xce2  ;; deprecated 
)(defconstant XK_hebrew_dalet                                #xce3
)(defconstant XK_hebrew_daleth                               #xce3  ;; deprecated 
)(defconstant XK_hebrew_he                                   #xce4
)(defconstant XK_hebrew_waw                                  #xce5
)(defconstant XK_hebrew_zain                                 #xce6
)(defconstant XK_hebrew_zayin                                #xce6  ;; deprecated 
)(defconstant XK_hebrew_chet                                 #xce7
)(defconstant XK_hebrew_het                                  #xce7  ;; deprecated 
)(defconstant XK_hebrew_tet                                  #xce8
)(defconstant XK_hebrew_teth                                 #xce8  ;; deprecated 
)(defconstant XK_hebrew_yod                                  #xce9
)(defconstant XK_hebrew_finalkaph                            #xcea
)(defconstant XK_hebrew_kaph                                 #xceb
)(defconstant XK_hebrew_lamed                                #xcec
)(defconstant XK_hebrew_finalmem                             #xced
)(defconstant XK_hebrew_mem                                  #xcee
)(defconstant XK_hebrew_finalnun                             #xcef
)(defconstant XK_hebrew_nun                                  #xcf0
)(defconstant XK_hebrew_samech                               #xcf1
)(defconstant XK_hebrew_samekh                               #xcf1  ;; deprecated 
)(defconstant XK_hebrew_ayin                                 #xcf2
)(defconstant XK_hebrew_finalpe                              #xcf3
)(defconstant XK_hebrew_pe                                   #xcf4
)(defconstant XK_hebrew_finalzade                            #xcf5
)(defconstant XK_hebrew_finalzadi                            #xcf5  ;; deprecated 
)(defconstant XK_hebrew_zade                                 #xcf6
)(defconstant XK_hebrew_zadi                                 #xcf6  ;; deprecated 
)(defconstant XK_hebrew_qoph                                 #xcf7
)(defconstant XK_hebrew_kuf                                  #xcf7  ;; deprecated 
)(defconstant XK_hebrew_resh                                 #xcf8
)(defconstant XK_hebrew_shin                                 #xcf9
)(defconstant XK_hebrew_taw                                  #xcfa
)(defconstant XK_hebrew_taf                                  #xcfa  ;; deprecated 
)(defconstant XK_Hebrew_switch        #xFF7E  ;; Alias for mode_switch 
;;endif ;; XK_HEBREW 
)
