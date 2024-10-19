#!/usr/bin/env perl

# Extract tree structure and data from TreeLoadBalancer CommGraphWriter output.
# Write input file for dot to plot the tree.

my @saveargs = @ARGV;

require Getopt::Long;
Getopt::Long::GetOptions( 'help' => \$help
	, 'verbose' => \$verbose
	, 'debug' => \$debug
        , 'record=s' => \@records
	, 'output=s' => \$output
        , 'edgeout=s' => \@edgeout
        , 'nodeout=s' => \@nodeout
        , 'twopi' => \$runtwopi
	);


my $script_name = `basename $0`; chop $script_name;

if ( $help ) {
  print <<_EOM_;

Usage: $script_name [options] <log file>

Extract the tree data written to SAMRAI log files by CommGraphWriter.
Write out the tree in format to be used by dot.

Log files must be of the form *-<nproc>.log.*
All log files must have the same nproc.

  --output=<filename>
    The dot filename.  If omitted, generate a name ending in '.dot'
    from the log file.  To output to stdout, use '-'.

  --record=<#[,#...]>
    Comma-separated list of record numbers to extract.
    If omitted, extract all records.

  --edgeout=<s1> --edgeout=<s2> ...
    Specify which edges to draw.  String values must be of the form
    "c:<value for color>,w:<value for width>", where "value for ..."
    is one of the values from the input data.  If either "c:" or "w:"
    is omitted, a uniform value is assumed.

  --nodeout=<str>
    Specify which node values to draw.  String values must be of the form
    "c:<value for color>", where "value for ..."
    is one of the values from the input data.

  --twopi
    Run twopi on the dot file if it is successfully generated.
    Error if output is stdout.
    Generate a postscript file of the tree.

_EOM_
  exit(0);
}


@records = split(/,/, join(',',@records));
print "records: @records\n";


my $logfile = shift;
die "No logfile specified" if $logfile eq '';
print "logfile = '$logfile'\n" if $debug;



# Parse log files for edges and nodes.


# Output;
my $outfh;
if (!defined($output)) {
  ($output = $logfile) =~ s!.*/!!;
  $output =~ s!\.log(\.\d+)?$!!;
  $output .= ".dot";
}
print "Writing to file '$output'\n" if $debug;


my $logfh;
open( $logfh, "< $logfile" ) || die "Cannot open log file $logfile";
print "Opened file $logfile\n" if $debug;


while (!eof($logfh)) {

  # Look for start of output record.
  my $currecordnum;
  while ( !eof($logfh) && ($_ = <$logfh>) ) {
    if ( /^CommGraphWriter begin record number (\d+)\n$/ ) {
      $currecordnum = $1;
      last;
    }
  }
  last if !defined($currecordnum);
  next if ( @records > 0 && !number_in_list($currecordnum,@records) );


  print "\nRecord number $currecordnum\n";


  my @nvalues;		# 2D array of node values: [nvalueindex][nodeindex]
  my @nvaluenames;	# value index --> value name
  my %nvalueids;	# value name --> value index

  my @evalues;		# 2D array with all edge values: [evalueindex][edgeindex]
  my @evaluenames;	# value index --> value name
  my %evalueids;	# value name --> value index
  my @edgenames;	# edge index --> edge name
  my %edgeids;		# edge name --> edge index

  # Look for graph data;
  get_graph_from_log( $logfh, \@nvalues, \@nvaluenames, \%nvalueids, \@evalues, \@evaluenames, \%evalueids, \@edgenames, \%edgeids );



  if ($debug) {
    for (0..$#edgenames) {
      print "edgename[$_] = '$edgenames[$_]' $edgeids{$edgenames[$_]}\n";
    }
    for (0..$#evaluenames) {
      print "evaluenames[$_] = '$evaluenames[$_]' $evalueids{$evaluenames[$_]}\n";
    }
    for $edgename ( @edgenames ) {
      my $edgeid = $edgeids{$edgename};
      print "Edge $edgeid '$edgename':";
      for $evaluename (@evaluenames) {
        my $evalueid = $evalueids{$evaluename};
        print "   '$evaluename'=$evalues[$evalueid][$edgeid]";
      }
      print "\n";
    }
  }



  # Choose default edges to plot, if needed.
  if (!defined @edgeout) {
    print STDERR "No edgeout specified.\n";
    print STDERR "Available edgeouts are:";
    for ( @evaluenames ) { print STDERR " '$_'"; }
    print STDERR "\n";
    print STDERR "Available nodeouts are:";
    for ( @nvaluenames ) { print STDERR " '$_'"; }
    print STDERR "\n";
    next;
    #$edgeout[0] = "w:$evaluenames[0]";
    #print STDERR "Choosing default edge group '$edgeout[0]' to plot.\n";
  }



  # Validity check for edgeout switches.
  if ($debug) {
    print "Edge specifiers:\n";
  }
  for (@edgeout) {
    (my $name1, $attr1, $name2, $attr2) = parse_edgeout($_);
    print "edgeout='$_' gave:   '$name1' for '$attr1', '$name2' for '$attr2'\n" if $debug;
    die "Repeated attribute '$attr1' edgeout '$_'" if ( $attr1 ne '' && $attr1 eq $attr2 );
    die "Attribute '$attr1' in edgeout '$_' unrecognized" if ( $attr1 ne '' && $attr1 ne 'c' && $attr1 ne 'w' );
    die "Attribute '$attr2' in edgeout '$_' unrecognized" if ( $attr2 ne '' && $attr2 ne 'c' && $attr2 ne 'w' );
    die "Value '$name1' in edgeout '$_' not in file $logfile" if ( $name1 && !string_in_list($name1, @evaluenames) );
    die "Value '$name2' in edgeout '$_' not in file $logfile" if ( $name2 && !string_in_list($name2, @evaluenames) );
  }
  die "Only one nodeout switch is allowed" if @nodeout > 1;
  for (@nodeout) {
    (my $name1, $attr1) = parse_nodeout($_);
    print "nodeout='$_' gave:   '$name1' for '$attr1'\n" if $debug;
    die "Attribute '$attr1' in nodeout '$_' unrecognized" if ( $attr1 ne '' && $attr1 ne 'c' );
  }



  # Compute range of each edge value field.
  my @iemin, @iemax;
  my @vemin, @vemax;
  for ( 0..$#evalues ) {
    ($iemin[$_], $iemax[$_]) = iextrema(@{$evalues[$_]});
    $vemin[$_] = $evalues[$_][$iemin[$_]];
    $vemax[$_] = $evalues[$_][$iemax[$_]];
    print "Edge values ($_) '$evaluenames[$_]' in [$vemin[$_] @ $iemin[$_] , $vemax[$_] @ $iemax[$_] ]\n" if $debug;
  }



  # Compute range of each node value field.
  my @inmin, @inmax;
  my @vnmin, @vnmax;
  for ( 0..$#nvalues ) {
    ($inmin[$_], $inmax[$_]) = iextrema(@{$nvalues[$_]});
    $vnmin[$_] = $nvalues[$_][$inmin[$_]];
    $vnmax[$_] = $nvalues[$_][$inmax[$_]];
    print "Node values ($_) '$nvaluenames[$_]' in [$vnmin[$_] @ $inmin[$_] , $vnmax[$_] @ $inmax[$_] ]\n" if $debug;
  }



  # Write out tree in dot format.

  if ( !defined($outfh) ) {
    open $outfh, ">$output" || die "Cannot open output file '$output'";
    print $outfh "// Written by $0 with command:\n// '$0'";
    for (@saveargs) { print $outfh " '$_'"; }
    print $outfh "\n\n";

    print $outfh "// Available edgeouts are:";
    for ( @evaluenames ) { print $outfh " '$_'"; }
    print $outfh "\n";
    print $outfh "// Available nodeouts are:";
    for ( @nvaluenames ) { print $outfh " '$_'"; }
    print $outfh "\n";
  }

  my $edgewidth, $saturation;
  my $brightness = 1;
  my $minsat = 0.05; # Useful saturation is from $minsat to 1.
  my $maxsat = 1;
  my $minwidth = 1;
  my $maxwidth = 15;


  print $outfh "\ndigraph \"$logfile:$currecordnum\" {\n";



  for ( 0..$#nodeout ) {

    my $hue = 0.5;

    print STDERR "\n\tNode group: $_ defined by 'nodeout=$nodeout[$_]'\n";
    print $outfh "\n\t// Node group: $_ defined by 'nodeout=$nodeout[$_]'\n";
    (my $name1, $attr1) = parse_nodeout($nodeout[$_]);
    my $colorvid = $nvalueids{$name1};

    my (@colorfield, $colorfieldmin, $colorfieldmax);
    if ( defined($colorvid) ) {
      @colorfield = @{$nvalues[$colorvid]};
      $colorfieldmin = $vnmin[$colorvid];
      $colorfieldmax = $vnmax[$colorvid];
      my $colorfieldminstr = sprintf("%.4g", $colorfieldmin);
      my $colorfieldmaxstr = sprintf("%.4g", $colorfieldmax);
      print STDERR "\tcolor='$nvaluenames[$colorvid]' in [$colorfieldminstr @ $inmin[$colorvid], $colorfieldmaxstr @ $inmax[$colorvid]]\n";
      print $outfh "\t\"nodeout[$_]\" [shape=box,style=filled,color=\"$hue,$maxsat,$brightness\",label=\"$nvaluenames[$colorvid] in [$colorfieldminstr @ $inmin[$colorvid], $colorfieldmaxstr @ $inmax[$colorvid]]\"]\n";
    }

    for ( 0..$#colorfield ) {
      my $cval = $colorfield[$_];
      my $sat = $colorfieldmin == $colorfieldmax ? 1 : $minsat + ($maxsat-$minsat)*($cval-$colorfieldmin)/($colorfieldmax-$colorfieldmin);
      print $outfh "\t$_ [style=filled,shape=box,width=0.4,height=0.2, color=\"$hue,$sat,$brightness\"] // c:$cval\n";
    }

  }


  my @edgehues = ( 2./3, 0, 1.0/3, 1.0 );

  for ( 0..$#edgeout ) {

    (my $name1, $attr1, $name2, $attr2) = parse_edgeout($edgeout[$_]);
    #print "Parsed edgeout '$_' into: $name1, $attr1, $name2, $attr2\n";

    my $colorvid, $widthvid;
    if ( $attr1 eq 'c' ) { $colorvid = $evalueids{$name1}; }
    if ( $attr2 eq 'c' ) { $colorvid = $evalueids{$name2}; }
    if ( $attr1 eq 'w' ) { $widthvid = $evalueids{$name1}; }
    if ( $attr2 eq 'w' ) { $widthvid = $evalueids{$name2}; }

    print STDERR "\n\tEdge group: $_ defined by 'edgeout=$edgeout[$_]'\n";
    print $outfh "\n\t// Edge group: $_ defined by 'edgeout=$edgeout[$_]'\n";

    my $labelstr;

    my (@colorfield, $colorfieldmin, $colorfieldmax);
    if ( defined($colorvid) ) {
      @colorfield = @{$evalues[$colorvid]};
      $colorfieldmin = $vemin[$colorvid];
      $colorfieldmax = $vemax[$colorvid];
      my $colorfieldminstr = sprintf("%.4g", $colorfieldmin);
      my $colorfieldmaxstr = sprintf("%.4g", $colorfieldmax);
      print STDERR "\tcolor='$evaluenames[$colorvid]' in [$colorfieldminstr @ $edgenames[$iemin[$colorvid]], $colorfieldmaxstr @ $edgenames[$iemax[$colorvid]]]\n";
      $labelstr = "color='$evaluenames[$colorvid]' in [$colorfieldminstr @ $edgenames[$iemin[$colorvid]], $colorfieldmaxstr @ $edgenames[$iemax[$colorvid]]]";
    }

    my (@widthfield, $widthfieldmin, $widthfieldmax);
    if ( defined($widthvid) ) {
      @widthfield = @{$evalues[$widthvid]};
      $widthfieldmin = $vemin[$widthvid];
      $widthfieldmax = $vemax[$widthvid];
      my $widthfieldminstr = sprintf("%.4g", $widthfieldmin);
      my $widthfieldmaxstr = sprintf("%.4g", $widthfieldmax);
      print STDERR "\twidth='$evaluenames[$widthvid]' in [$widthfieldminstr @ $edgenames[$iemin[$widthvid]], $widthfieldmaxstr @ $edgenames[$iemax[$widthvid]]]\n";
      $labelstr .= "\\nwidth='$evaluenames[$widthvid]' in [$widthfieldminstr @ $edgenames[$iemin[$widthvid]], $widthfieldmaxstr @ $edgenames[$iemax[$widthvid]]]";
    }

    my $primaryvid = $evalueids{$name1};
    my @primaryfield = @{$evalues[$primaryvid]};

    my $edgehue = $edgehues[ ($_ % @edgehues) ];

    print $outfh "\ta$_ [style=invis]\n\tb$_ [style=invis]\n\ta$_ -> b$_ [label=\"$labelstr\", color=\"$edgehue,$maxsat,$brightness\", penwidth=3]\n";

    for ( 0..$#primaryfield ) {

      if ( defined($primaryfield[$_]) ) {
        # print "primary = $primaryfield[$_], color = $colorfield[$_]\n";
        my $edgename = $edgenames[$_];
        my $cval = $colorfield[$_];
        my $wval = $widthfield[$_];

        $edgename =~ m/(\d+)(.)(\d+)/ || die "Cannot parse edge $edgename";
        (my $na, $arrow, $nb) = ($1,$2,$3);

        #my $edgewidth = 1;
        my $edgewidth = $widthfieldmin == $widthfieldmax ? 1 : $minwidth + ($maxwidth-$minwidth)*($wval-$widthfieldmin)/($widthfieldmax-$widthfieldmin);
        #my $style = ($wval > $widthfieldmin && $widthfieldmin != $widthfieldmax) ? "solid" : "dotted";

        my $saturation = $colorfieldmin == $colorfieldmax ? 1 : $minsat + (1-$minsat)*($cval-$colorfieldmin)/($colorfieldmax-$colorfieldmin);

        my $edgestr = $arrow eq '>' ? "$na -> $nb" : "$nb -> $na";
        print $outfh "\t$edgestr [color=\"$edgehue,$saturation,$brightness\", penwidth=$edgewidth] // c:$cval w:$wval\n";
      }

    }

  }

  print $outfh "}\n";


}


close $logfh || die "Cannot close log file $logfile";

if ( !defined($outfh) ) {
  print STDERR "\nOutput file '$output' was NOT written!\n";
}
else {
  close $outfh || die "Cannot close output file '$output'";
  if ( defined($runtwopi) ) {
    die "Cannot use --runtwopi when writing output to stdout" if $output eq '-';
    my $psfile = $output;
    $psfile =~ s/\.dot$//;
    $psfile .= ".ps";
    print STDERR "\nRunning 'twopi -Tps -o $psfile $output'\n";
    my $errf = system('twopi', '-Tps', '-o', $psfile, $output);
    die "Error running twopi to generate '$psfile'" if $errf;
  }
}


exit;


sub get_graph_from_log {
  # Search for next CommGraphWriter section and extract its data.

  my $fh = shift;
  my @nvalues;		# 2D array of node values: [nvalueindex][nodeindex]
  my @nvaluenames;	# value index --> value name
  my %nvalueids;	# value name --> value index

  my @evalues;		# 2D array with all edge values: [evalueindex][edgeindex]
  my @evaluenames;	# value index --> value name
  my %evalueids;	# value name --> value index
  my @edgenames;	# edge index --> edge name
  my %edgeids;		# edge name --> edge index

  while ( $_ = <$fh> ) {

    print "line:$_" if $debug;
    last if eof($fh);
    last if ( $_ =~ m/^CommGraphWriter end record number/ );
    next if ( $_ =~ m'^#' );
    chop;

    my @tmpfields = split( /\t/, $_, 5 );

    if ( $tmpfields[1] eq '->' || $tmpfields[1] eq '<-' ) {
      # Edge
      my ($na, $arrow, $nb, $edgeval, $evaluename) = split( /\t/, $_, 5 );
      $arrow =~ s/-//;
      die "Column count not the expected 5" if !defined($evaluename);

      # Skip edges with invalid nodes.
      next if ( $na < 0 || $nb < 0 );

      my $edgename = $arrow eq '>' ? "$na>$nb" : "$nb>$na";

      if ( !defined $edgeids{$edgename} ) {
        push @edgenames, $edgename;
        $edgeids{$edgename} = $#edgenames;
      }
      my $edgeid = $edgeids{$edgename};

      if ( !defined $evalueids{$evaluename} ) {
        push @evaluenames, $evaluename;
        $evalueids{$evaluename} = $#evaluenames;
      }
      my $evalueid = $evalueids{$evaluename};

      $evalues[$evalueid][$edgeid] = $edgeval;
      #print "found edge '$edgename' with '$evaluename' = $edgeval \n";
    }
    else {
      # Node value
      my ($nodeid, $nodevalue, $nvaluename) = split( /\t/, $_, 3 );
      #print "nodeid,nodevaule,nvaluename = $nodeid,$nodevalue,$nvaluename\n";

      if ( !defined($nvalueids{$nvaluename}) ) {
        push @nvaluenames, $nvaluename;
        $nvalueids{$nvaluename} = $#nvaluenames;
      }
      my $nvalueid = $nvalueids{$nvaluename};
      $nvalues[$nvalueid][$nodeid] = $nodevalue;

    }

  }

  # Set the data, which was sent in by reference.
  @{scalar shift} = @nvalues;
  @{scalar shift} = @nvaluenames;
  %{scalar shift} = %nvalueids;
  @{scalar shift} = @evalues;
  @{scalar shift} = @evaluenames;
  %{scalar shift} = %evalueids;
  @{scalar shift} = @edgenames;
  %{scalar shift} = %edgeids;

}



# Parse nodeout to get values to use for node attributes.
sub parse_nodeout {
  my $str = shift;
  my @s1 = split(/,/, $str);
  die "nodeout '$str' has more than one attribute specifiers." if @s1 > 1;
  my @value, my @attr;
  for (@s1) {
    my @s2 = split /:/, $_, 2;
    push( @value, $s2[1] );
    push( @attr, $s2[0] );
  }
  return ( $value[0], $attr[0] );
}



# Parse edgeout to get values to use for edge attributes.
sub parse_edgeout {
  my $str = shift;
  my @s1 = split(/,/, $str);
  #print "s1: ", array_to_str(@s1), "\n";
  die "edgeout '$str' has more than two attribute specifiers." if @s1 > 2;
  my @value, my @attr;
  #print "ini attr: ", array_to_str(@attr), "\n";
  for (@s1) {
    my @s2 = split /:/, $_, 2;
    #print "s2: ", array_to_str(@s2), "\n";
    push( @value, $s2[1] );
    push( @attr, $s2[0] );
  }
  #print "attr: ", array_to_str(@attr), "\n";
  #print "value: ", array_to_str(@value), "\n";
  return ( $value[0], $attr[0], $value[1], $attr[1] );
}



sub array_to_str {
  my @a = @_;
  my $str = sprintf("%d items:", scalar(@a));
  for (0..$#a) {
    $str .= "  $_='$a[$_]'";
  }
  return $str;
}



# Whether a string is in a list.
sub string_in_list {
  my $s = shift;
  my @l = @_;
  for (@_) {
    if ( $_ eq $s ) { return 1; }
  }
  return 0;
}



# Whether a number is in a list.
sub number_in_list {
  my $s = shift;
  my @l = @_;
  for (@_) {
    if ( $_ == $s ) { return 1; }
  }
  return 0;
}


# Compute extrema points in an edge group.
sub iextrema {
  my @edges = @_;
  my $imin = 0;
  my $imax = 0;
  for (0..$#edges) {
    if ( defined($edges[$_]) ) {
      #print "Defined $edges[$_] for $_\n";
      $imin = $_ if $edges[$imin] > $edges[$_] || !defined($edges[$imin]);
      $imax = $_ if $edges[$imax] < $edges[$_] || !defined($edges[$imax]);
    }
  }
  return ($imin,$imax);
}
