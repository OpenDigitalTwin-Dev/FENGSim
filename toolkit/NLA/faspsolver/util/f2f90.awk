# Convert F77 comments to F90 style
# Writtne by Chensong Zhang for FASP (06/04/2010)

BEGIN {
}

/^c|^C/ {
  printf "!%s\n",$0;
  next;
}

!/^c|^C/ {
  printf "%s\n",$0;
  next;
}
