(in-package :XLIB)
; X.lsp        modified by Hiep Huu Nguyen                      27 Aug 92

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

;;
;;	$XConsortium: X.h,v 1.66 88/09/06 15:55:56 jim Exp $
 

;; Definitions for the X window system likely to be used by applications 


;;**********************************************************
;;Copyright 1987 by Digital Equipment Corporation, Maynard, Massachusetts,
;;and the Massachusetts Institute of Technology, Cambridge, Massachusetts.

;;modified by Hiep H Nguyen 28 Jul 91

;;                        All Rights Reserved

;;Permission to use, copy, modify, and distribute this software and its 
;;documentation for any purpose and without fee is hereby granted, 
;;provided that the above copyright notice appear in all copies and that
;;both that copyright notice and this permission notice appear in 
;;supporting documentation, and that the names of Digital or MIT not be
;;used in advertising or publicity pertaining to distribution of the
;;software without specific, written prior permission.  

;;DIGITAL DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
;;ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL
;;DIGITAL BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
;;ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
;;WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
;;ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
;;SOFTWARE.

;;*****************************************************************
(defconstant X_PROTOCOL	11		) ;; current protocol version 
(defconstant X_PROTOCOL_REVISION 0		) ;; current minor version 

(defconstant  True 1)
(defconstant  False 0)

;; Resources 

;;typedef unsigned long XID) ;

;;typedef XID Window) ;
;;typedef XID Drawable) ;
;;typedef XID Font) ;
;;typedef XID Pixmap) ;
;;typedef XID Cursor) ;
;;typedef XID Colormap) ;
;;typedef XID GContext) ;
;;typedef XID KeySym) ;

;;typedef unsigned long Mask) ;

;;typedef unsigned long Atom) ;

;;typedef unsigned long VisualID) ;

;;typedef unsigned long Time) ;

;;typedef unsigned char KeyCode) ;

;;****************************************************************
;; * RESERVED RESOURCE AND CONSTANT DEFINITIONS
;; ****************************************************************

(defconstant None                 0	) ;; universal null resource or null atom 

(defconstant ParentRelative       1	) ;; background pixmap in CreateWindow
				    ;;and ChangeWindowAttributes 

(defconstant CopyFromParent       0	) ;; border pixmap in CreateWindow
				       ;;and ChangeWindowAttributes
				   ;;special VisualID and special window
				      ;; class passed to CreateWindow 

(defconstant PointerWindow        0	) ;; destination window in SendEvent 
(defconstant InputFocus           1	) ;; destination window in SendEvent 

(defconstant PointerRoot          1	) ;; focus window in SetInputFocus 

(defconstant AnyPropertyType      0	) ;; special Atom, passed to GetProperty 

(defconstant AnyKey		     0	) ;; special Key Code, passed to GrabKey 

(defconstant AnyButton            0	) ;; special Button Code, passed to GrabButton 

(defconstant AllTemporary         0	) ;; special Resource ID passed to KillClient 

(defconstant CurrentTime          0	) ;; special Time 

(defconstant NoSymbol	     0	) ;; special KeySym 

;;**************************************************************** 
;; * EVENT DEFINITIONS 
;; ****************************************************************

;; Input Event Masks. Used as event-mask window attribute and as arguments
;;   to Grab requests.  Not to be confused with event names.  

(defconstant NoEventMask			0)
(defconstant KeyPressMask			(expt 2 0)  )
(defconstant KeyReleaseMask			(expt 2 1)  )
(defconstant ButtonPressMask			(expt 2 2)  )
(defconstant ButtonReleaseMask		(expt 2 3)  )
(defconstant EnterWindowMask			(expt 2 4)  )
(defconstant LeaveWindowMask			(expt 2 5)  )
(defconstant PointerMotionMask		(expt 2 6)  )
(defconstant PointerMotionHintMask		(expt 2 7)  )
(defconstant Button1MotionMask		(expt 2 8)  )
(defconstant Button2MotionMask		(expt 2 9)  )
(defconstant Button3MotionMask		(expt 2 10) )
(defconstant Button4MotionMask		(expt 2 11) )
(defconstant Button5MotionMask		(expt 2 12) )
(defconstant ButtonMotionMask		(expt 2 13) )
(defconstant KeymapStateMask			(expt 2 14))
(defconstant ExposureMask			(expt 2 15) )
(defconstant VisibilityChangeMask		(expt 2 16) )
(defconstant StructureNotifyMask		(expt 2 17) )
(defconstant ResizeRedirectMask		(expt 2 18) )
(defconstant SubstructureNotifyMask		(expt 2 19) )
(defconstant SubstructureRedirectMask	(expt 2 20) )
(defconstant FocusChangeMask			(expt 2 21) )
(defconstant PropertyChangeMask		(expt 2 22) )
(defconstant ColormapChangeMask		(expt 2 23) )
(defconstant OwnerGrabButtonMask		(expt 2 24) )

;; Event names.  Used in "type" field in XEvent structures.  Not to be
;;confused with event masks above.  They start from 2 because 0 and 1
;;are reserved in the protocol for errors and replies. 

(defconstant KeyPress		2)
(defconstant KeyRelease		3)
(defconstant ButtonPress		4)
(defconstant ButtonRelease		5)
(defconstant MotionNotify		6)
(defconstant EnterNotify		7)
(defconstant LeaveNotify		8)
(defconstant FocusIn			9)
(defconstant FocusOut		10)
(defconstant KeymapNotify		11)
(defconstant Expose			12)
(defconstant GraphicsExpose		13)
(defconstant NoExpose		14)
(defconstant VisibilityNotify	15)
(defconstant CreateNotify		16)
(defconstant DestroyNotify		17)
(defconstant UnmapNotify		18)
(defconstant MapNotify		19)
(defconstant MapRequest		20)
(defconstant ReparentNotify		21)
(defconstant ConfigureNotify		22)
(defconstant ConfigureRequest	23)
(defconstant GravityNotify		24)
(defconstant ResizeRequest		25)
(defconstant CirculateNotify		26)
(defconstant CirculateRequest	27)
(defconstant PropertyNotify		28)
(defconstant SelectionClear		29)
(defconstant SelectionRequest	30)
(defconstant SelectionNotify		31)
(defconstant ColormapNotify		32)
(defconstant ClientMessage		33)
(defconstant MappingNotify		34)
(defconstant LASTEvent		35	) ;; must be bigger than any event # 


;; Key masks. Used as modifiers to GrabButton and GrabKey, results of QueryPointer,
;;   state in various key-, mouse-, and button-related events. 

(defconstant ShiftMask		(expt 2 0))
(defconstant LockMask		(expt 2 1))
(defconstant ControlMask		(expt 2 2))
(defconstant Mod1Mask		(expt 2 3))
(defconstant Mod2Mask		(expt 2 4))
(defconstant Mod3Mask		(expt 2 5))
(defconstant Mod4Mask		(expt 2 6))
(defconstant Mod5Mask		(expt 2 7))

;; modifier names.  Used to build a SetModifierMapping request or
;;   to read a GetModifierMapping request.  These correspond to the
;;   masks defined above. 
(defconstant ShiftMapIndex		0)
(defconstant LockMapIndex		1)
(defconstant ControlMapIndex		2)
(defconstant Mod1MapIndex		3)
(defconstant Mod2MapIndex		4)
(defconstant Mod3MapIndex		5)
(defconstant Mod4MapIndex		6)
(defconstant Mod5MapIndex		7)


;; button masks.  Used in same manner as Key masks above. Not to be confused
;;   with button names below. 

(defconstant Button1Mask		(expt 2 8))
(defconstant Button2Mask		(expt 2 9))
(defconstant Button3Mask		(expt 2 10))
(defconstant Button4Mask		(expt 2 11))
(defconstant Button5Mask		(expt 2 12))

(defconstant AnyModifier		(expt 2 15)  ) ;; used in GrabButton, GrabKey 


;; button names. Used as arguments to GrabButton and as detail in ButtonPress
;;   and ButtonRelease events.  Not to be confused with button masks above.
;;   Note that 0 is already defined above as "AnyButton".  

(defconstant Button1			1)
(defconstant Button2			2)
(defconstant Button3			3)
(defconstant Button4			4)
(defconstant Button5			5)

;; Notify modes 

(defconstant NotifyNormal		0)
(defconstant NotifyGrab		1)
(defconstant NotifyUngrab		2)
(defconstant NotifyWhileGrabbed	3)

(defconstant NotifyHint		1	) ;; for MotionNotify events 

;; Notify detail 

(defconstant NotifyAncestor		0)
(defconstant NotifyVirtual		1)
(defconstant NotifyInferior		2)
(defconstant NotifyNonlinear		3)
(defconstant NotifyNonlinearVirtual	4)
(defconstant NotifyPointer		5)
(defconstant NotifyPointerRoot	6)
(defconstant NotifyDetailNone	7)

;; Visibility notify 

(defconstant VisibilityUnobscured		0)
(defconstant VisibilityPartiallyObscured	1)
(defconstant VisibilityFullyObscured		2)

;; Circulation request 

(defconstant PlaceOnTop		0)
(defconstant PlaceOnBottom		1)

;; protocol families 

(defconstant FamilyInternet		0)
(defconstant FamilyDECnet		1)
(defconstant FamilyChaos		2)

;; Property notification 

(defconstant PropertyNewValue	0)
(defconstant PropertyDelete		1)

;; Color Map notification 

(defconstant ColormapUninstalled	0)
(defconstant ColormapInstalled	1)

;; GrabPointer, GrabButton, GrabKeyboard, GrabKey Modes 

(defconstant GrabModeSync		0)
(defconstant GrabModeAsync		1)

;; GrabPointer, GrabKeyboard reply status 

(defconstant GrabSuccess		0)
(defconstant AlreadyGrabbed		1)
(defconstant GrabInvalidTime		2)
(defconstant GrabNotViewable		3)
(defconstant GrabFrozen		4)

;; AllowEvents modes 

(defconstant AsyncPointer		0)
(defconstant SyncPointer		1)
(defconstant ReplayPointer		2)
(defconstant AsyncKeyboard		3)
(defconstant SyncKeyboard		4)
(defconstant ReplayKeyboard		5)
(defconstant AsyncBoth		6)
(defconstant SyncBoth		7)

;; Used in SetInputFocus, GetInputFocus 

(defconstant RevertToNone		None)
(defconstant RevertToPointerRoot	PointerRoot)
(defconstant RevertToParent		2)

;;****************************************************************
;; * ERROR CODES 
;; ****************************************************************

(defconstant Success		   0	) ;; everything's okay 
(defconstant BadRequest	   1	) ;; bad request code 
(defconstant BadValue	   2	) ;; int parameter out of range 
(defconstant BadWindow	   3	) ;; parameter not a Window 
(defconstant BadPixmap	   4	) ;; parameter not a Pixmap 
(defconstant BadAtom		   5	) ;; parameter not an Atom 
(defconstant BadCursor	   6	) ;; parameter not a Cursor 
(defconstant BadFont		   7	) ;; parameter not a Font 
(defconstant BadMatch	   8	) ;; parameter mismatch 
(defconstant BadDrawable	   9	) ;; parameter not a Pixmap or Window 
(defconstant BadAccess	  10	) ;; depending on context:
				 ;;- key/button already grabbed
				 ;;- attempt to free an illegal 
				 ;;  cmap entry 
				;;- attempt to store into a read-only 
				  ;; color map entry.
                                 ;;- attempt to modify the access control
				  ;; list from other than the local host.
				
(defconstant BadAlloc	  11	) ;; insufficient resources 
(defconstant BadColor	  12	) ;; no such colormap 
(defconstant BadGC		  13	) ;; parameter not a GC 
(defconstant BadIDChoice	  14	) ;; choice not in range or already used 
(defconstant BadName		  15	) ;; font or color name doesn't exist 
(defconstant BadLength	  16	) ;; Request length incorrect 
(defconstant BadImplementation 17	) ;; server is defective 

(defconstant FirstExtensionError	128)
(defconstant LastExtensionError	255)

;;****************************************************************
;; * WINDOW DEFINITIONS 
;; ****************************************************************

;; Window classes used by CreateWindow 
;; Note that CopyFromParent is already defined as 0 above 

(defconstant InputOutput		1)
(defconstant InputOnly		2)

;; Window attributes for CreateWindow and ChangeWindowAttributes 

(defconstant CWBackPixmap		(expt 2 0))
(defconstant CWBackPixel		(expt 2 1))
(defconstant CWBorderPixmap		(expt 2 2))
(defconstant CWBorderPixel           (expt 2 3))
(defconstant CWBitGravity		(expt 2 4))
(defconstant CWWinGravity		(expt 2 5))
(defconstant CWBackingStore          (expt 2 6))
(defconstant CWBackingPlanes	        (expt 2 7))
(defconstant CWBackingPixel	        (expt 2 8))
(defconstant CWOverrideRedirect	(expt 2 9))
(defconstant CWSaveUnder		(expt 2 10))
(defconstant CWEventMask		(expt 2 11))
(defconstant CWDontPropagate	        (expt 2 12))
(defconstant CWColormap		(expt 2 13))
(defconstant CWCursor	        (expt 2 14))

;; ConfigureWindow structure 

(defconstant CWX			(expt 2 0))
(defconstant CWY			(expt 2 1))
(defconstant CWWidth			(expt 2 2))
(defconstant CWHeight		(expt 2 3))
(defconstant CWBorderWidth		(expt 2 4))
(defconstant CWSibling		(expt 2 5))
(defconstant CWStackMode		(expt 2 6))


;; Bit Gravity 

(defconstant ForgetGravity		0)
(defconstant NorthWestGravity	1)
(defconstant NorthGravity		2)
(defconstant NorthEastGravity	3)
(defconstant WestGravity		4)
(defconstant CenterGravity		5)
(defconstant EastGravity		6)
(defconstant SouthWestGravity	7)
(defconstant SouthGravity		8)
(defconstant SouthEastGravity	9)
(defconstant StaticGravity		10)

;; Window gravity + bit gravity above 

(defconstant UnmapGravity		0)

;; Used in CreateWindow for backing-store hint 

(defconstant NotUseful               0)
(defconstant WhenMapped              1)
(defconstant Always                  2)

;; Used in GetWindowAttributes reply 

(defconstant IsUnmapped		0)
(defconstant IsUnviewable		1)
(defconstant IsViewable		2)

;; Used in ChangeSaveSet 

(defconstant SetModeInsert           0)
(defconstant SetModeDelete           1)

;; Used in ChangeCloseDownMode 

(defconstant DestroyAll              0)
(defconstant RetainPermanent         1)
(defconstant RetainTemporary         2)

;; Window stacking method (in configureWindow) 

(defconstant Above                   0)
(defconstant Below                   1)
(defconstant TopIf                   2)
(defconstant BottomIf                3)
(defconstant Opposite                4)

;; Circulation direction 

(defconstant RaiseLowest             0)
(defconstant LowerHighest            1)

;; Property modes 

(defconstant PropModeReplace         0)
(defconstant PropModePrepend         1)
(defconstant PropModeAppend          2)

;;****************************************************************
;; * GRAPHICS DEFINITIONS
;; ****************************************************************

;; graphics functions, as in GC.alu 

(defconstant	GXclear			0		) ;; 0 
(defconstant GXand			1		) ;; src AND dst 
(defconstant GXandReverse		2		) ;; src AND NOT dst 
(defconstant GXcopy			3		) ;; src 
(defconstant GXandInverted		4		) ;; NOT src AND dst 
(defconstant	GXnoop			5		) ;; dst 
(defconstant GXxor			6		) ;; src XOR dst 
(defconstant GXor			7		) ;; src OR dst 
(defconstant GXnor			8		) ;; NOT src AND NOT dst 
(defconstant GXequiv			9		) ;; NOT src XOR dst 
(defconstant GXinvert		10		) ;; NOT dst 
(defconstant GXorReverse		11		) ;; src OR NOT dst 
(defconstant GXcopyInverted		12		) ;; NOT src 
(defconstant GXorInverted		13		) ;; NOT src OR dst 
(defconstant GXnand			14		) ;; NOT src OR NOT dst 
(defconstant GXset			15		) ;; 1 

;; LineStyle 

(defconstant LineSolid		0)
(defconstant LineOnOffDash		1)
(defconstant LineDoubleDash		2)

;; capStyle 

(defconstant CapNotLast		0)
(defconstant CapButt			1)
(defconstant CapRound		2)
(defconstant CapProjecting		3)

;; joinStyle 

(defconstant JoinMiter		0)
(defconstant JoinRound		1)
(defconstant JoinBevel		2)

;; fillStyle 

(defconstant FillSolid		0)
(defconstant FillTiled		1)
(defconstant FillStippled		2)
(defconstant FillOpaqueStippled	3)

;; fillRule 

(defconstant EvenOddRule		0)
(defconstant WindingRule		1)

;; subwindow mode 

(defconstant ClipByChildren		0)
(defconstant IncludeInferiors	1)

;; SetClipRectangles ordering 

(defconstant Unsorted		0)
(defconstant YSorted			1)
(defconstant YXSorted		2)
(defconstant YXBanded		3)

;; CoordinateMode for drawing routines 

(defconstant CoordModeOrigin		0	) ;; relative to the origin 
(defconstant CoordModePrevious       1	) ;; relative to previous point 

;; Polygon shapes 

;(defconstant Complex			0	) ;; paths may intersect 
(defconstant Nonconvex		1	) ;; no paths intersect, but not convex 
(defconstant Convex			2	) ;; wholly convex 

;; Arc modes for PolyFillArc 

(defconstant ArcChord		0	) ;; join endpoints of arc 
(defconstant ArcPieSlice		1	) ;; join endpoints to center of arc 

;; GC components: masks used in CreateGC, CopyGC, ChangeGC, OR'ed into
;;   GC.stateChanges 

(defconstant GCFunction              (expt 2 0))
(defconstant GCPlaneMask             (expt 2 1))
(defconstant GCForeground            (expt 2 2))
(defconstant GCBackground            (expt 2 3))
(defconstant GCLineWidth             (expt 2 4))
(defconstant GCLineStyle             (expt 2 5))
(defconstant GCCapStyle              (expt 2 6))
(defconstant GCJoinStyle		(expt 2 7))
(defconstant GCFillStyle		(expt 2 8))
(defconstant GCFillRule		(expt 2 9) )
(defconstant GCTile			(expt 2 10))
(defconstant GCStipple		(expt 2 11))
(defconstant GCTileStipXOrigin	(expt 2 12))
(defconstant GCTileStipYOrigin	(expt 2 13))
(defconstant GCFont 			(expt 2 14))
(defconstant GCSubwindowMode		(expt 2 15))
(defconstant GCGraphicsExposures     (expt 2 16))
(defconstant GCClipXOrigin		(expt 2 17))
(defconstant GCClipYOrigin		(expt 2 18))
(defconstant GCClipMask		(expt 2 19))
(defconstant GCDashOffset		(expt 2 20))
(defconstant GCDashList		(expt 2 21))
(defconstant GCArcMode		(expt 2 22))

(defconstant GCLastBit		22)
;;****************************************************************
;; * FONTS 
;; ****************************************************************

;; used in QueryFont -- draw direction 

(defconstant FontLeftToRight		0)
(defconstant FontRightToLeft		1)

(defconstant FontChange		255)

;;****************************************************************
;; *  IMAGING 
;; ****************************************************************

;; ImageFormat -- PutImage, GetImage 

(defconstant XYBitmap		0	) ;; depth 1, XYFormat 
(defconstant XYPixmap		1	) ;; depth == drawable depth 
(defconstant ZPixmap			2	) ;; depth == drawable depth 

;;****************************************************************
;; *  COLOR MAP STUFF 
;; ****************************************************************

;; For CreateColormap 

(defconstant AllocNone		0	) ;; create map with no entries 
(defconstant AllocAll		1	) ;; allocate entire map writeable 


;; Flags used in StoreNamedColor, StoreColors 

(defconstant DoRed			(expt 2 0))
(defconstant DoGreen			(expt 2 1))
(defconstant DoBlue			(expt 2 2))

;;****************************************************************
;; * CURSOR STUFF
;; ****************************************************************

;; QueryBestSize Class 

(defconstant CursorShape		0	) ;; largest size that can be displayed 
(defconstant TileShape		1	) ;; size tiled fastest 
(defconstant StippleShape		2	) ;; size stippled fastest 

;;**************************************************************** 
;; * KEYBOARD/POINTER STUFF
;; ****************************************************************

(defconstant AutoRepeatModeOff	0)
(defconstant AutoRepeatModeOn	1)
(defconstant AutoRepeatModeDefault	2)

(defconstant LedModeOff		0)
(defconstant LedModeOn		1)

;; masks for ChangeKeyboardControl 

(defconstant KBKeyClickPercent	(expt 2 0))
(defconstant KBBellPercent		(expt 2 1))
(defconstant KBBellPitch		(expt 2 2))
(defconstant KBBellDuration		(expt 2 3))
(defconstant KBLed			(expt 2 4))
(defconstant KBLedMode		(expt 2 5))
(defconstant KBKey			(expt 2 6))
(defconstant KBAutoRepeatMode	(expt 2 7))

(defconstant MappingSuccess     	0)
(defconstant MappingBusy        	1)
(defconstant MappingFailed		2)

(defconstant MappingModifier		0)
(defconstant MappingKeyboard		1)
(defconstant MappingPointer		2)

;;****************************************************************
;; * SCREEN SAVER STUFF 
;; ****************************************************************

(defconstant DontPreferBlanking	0)
(defconstant PreferBlanking		1)
(defconstant DefaultBlanking		2)

(defconstant DisableScreenSaver	0)
(defconstant DisableScreenInterval	0)

(defconstant DontAllowExposures	0)
(defconstant AllowExposures		1)
(defconstant DefaultExposures	2)

;; for ForceScreenSaver 

(defconstant ScreenSaverReset 0)
(defconstant ScreenSaverActive 1)

;;****************************************************************
;; * HOSTS AND CONNECTIONS
;; ****************************************************************

;; for ChangeHosts 

(defconstant HostInsert		0)
(defconstant HostDelete		1)

;; for ChangeAccessControl 

(defconstant EnableAccess		1      )
(defconstant DisableAccess		0)

;; Display classes  used in opening the connection 
;; * Note that the statically allocated ones are even numbered and the
;; * dynamically changeable ones are odd numbered 

(defconstant StaticGray		0)
(defconstant GrayScale		1)
(defconstant StaticColor		2)
(defconstant PseudoColor		3)
(defconstant TrueColor		4)
(defconstant DirectColor		5)


;; Byte order  used in imageByteOrder and bitmapBitOrder 

(defconstant LSBFirst		0)
(defconstant MSBFirst		1)


;(defconstant NULL 0)


