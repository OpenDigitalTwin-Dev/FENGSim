
# some extensions for gcl
# of course these could be in lisp, but keeping them on the
# tk side of the pipe can cut down overhead. for large things
# like getting a file

proc TextLoadFile {w file} {
    set f [open $file]
    $w delete 1.0 end
    while {![eof $f]} {
	$w insert end [read $f 10000]
    }
    close $f
}

proc insertWithTags {w text args} {
    set start [$w index insert]
    $w insert insert $text
    foreach tag [$w tag names $start] {
	$w tag remove $tag $start insert
    }
    foreach i $args {
	$w tag add $i $start insert
    }
}
# in WINDOW if TAG is set at INDEX then return the range
# of indices for which tag is set including index.

proc get_tag_range {w tag index} {
  set i 1
  set index [$w index $index]
  set range ""
  set ok  0
#  puts stdout $index
  foreach v [$w tag names $index] { if {$v == $tag} {set ok 1}}
  while $ok {
    set range [$w tag nextrange $tag "$index -$i chars" "$index +1 char"]
     if {[llength $range ] >= 2} { break;}
     if {[$w compare "$index - $i chars" <= "0.0 + 1 chars" ]} { break;}
     set i [expr $i + 1]
      }
  return $range
}

proc MultipleTagAdd {win tag start l} {
  set prev -1
  foreach v $l { puts stdout $v 
                   if { "$prev" == "-1" } {
                   set prev $v 
                   } else {
                   $win tag add $tag "$start + $prev chars" "$start + $v chars"
		   set prev -1 	
}}}


