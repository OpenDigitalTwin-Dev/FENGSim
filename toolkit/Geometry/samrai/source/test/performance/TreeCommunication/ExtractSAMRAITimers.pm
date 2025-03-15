package ExtractSAMRAITimers;

# Debugging parameters can be set externally.
my $verbose;
my $debug;

sub extract_timers {

# Aruments:

# 1 [i] Reference to array of number-of-processes.  These are the
# processes corresponding to the log files.

# 2 [i] Reference to array of log file names.  Log files not
# corresponding to a process-count value in argument 1 are ignored.

# 3 [i] Reference to array of test names.  Test names are as defined
# by in the TreeCommunication test code.  They must be in the same
# order as they appear in the log files.  The test number is the array
# index corresponding to the test names.

# 4 [o] Reference to a hash (%times) to be populated.
# $times{test_number:timer_name} will be an array of timers for that
# test number and time name.  The array contains timer values
# corresponding to the input nproc array.


my @nprocs = @{shift @_};
my @logs = @{shift @_};
my @test_names = @{shift @_};

# Temporary variable to hold output until the end.
my %times;

for $inproc (0 .. $#nprocs) {

  my $nproc = $nprocs[$inproc];

  # Get the log files corresponding to $nproc
  my $nprocstr = sprintf("%05d", $nproc);
  my @logs_for_nproc = grep /-$nprocstr\.log/, @logs;
  print STDERR ("logs for nproc=$nproc: ", @logs_for_nproc, "\n") if $verbose;

  for $fn (@logs_for_nproc) {

    if ( open (LOG, "< $fn") ) { warn "Opened $fn\n"; }
    else { warn "Cannot open $fn\n"; next; }

    for $test_index ( 0..$#test_names ) {

      while ( !eof(LOG) && ($_ = <LOG>) ) { last if /^Starting test/; }

      die "File $fn ended prematurely while searching for test $test_index ($test_names[$test_index])\n"
          if eof(LOG);

      /^Starting test Test([0-9]+) \((.*)\)$/;
      my $test_number = int($1);
      my $test_name = $2;
      print STDERR "Found line ${_}for test #$test_number ($test_name)\n"
          if $debug;

      die "Test $test_number is out of order.  Corrupted log file $fn?"
          if $test_number != $test_index;

      while ( !eof(LOG) && ($_=<LOG>) ) { last if /\brepetition\s*=\s*\d+/o; }
      die "File $fn ended prematurely while searching for repetitions\n" if eof(LOG);
      /\brepetition\s*=\s*(\d+)/o;
      my $repetition = $1;

      while ( !eof(LOG) && ($_=<LOG>) ) { last if /^WALLCLOCK TIME/o; }
      <LOG>; <LOG>;

      while ( !eof(LOG) && ($_ = <LOG>) ) {
        last if /TOTAL RUN TIME:/;
        s/^\s*//; # Strip leading white spaces.
        my @c = split /\s+/;
        $t = $c[0];
        $v = $c[5];
        $times{"$test_number:$t"}[$inproc] = $v/$repetition;
        print STDERR "$test_number:$t for $inproc is ", $times{"$test_number:$t"}[$inproc], "\n" if $debug;
      }
      die "File $fn ended prematurely while searching for timers." if eof(LOG);

    }

    close LOG;

  }

}

# Set output to the temporary variable.
%{@_[0]} = %times;

}




sub get_nprocs_from_log_names {
# Get list of nproc from log file names.
  my %nprocset;
  for (@_) {
    if ( /-([0-9]+).log/ ) {
      print STDERR "Log file $_ -> $1 procs\n" if $verbose;
      $nprocset{$1} = '';
    }
    else {
      die "Cannot determine nproc corresponding to log file $_\n";
    }
  }
  @nprocs = sort keys %nprocset;
  for (@nprocs) { $_ = sprintf("%d",$_); }
  @nprocs;
}



###################################################################
# Scan a timer paragraph for all the timers we need.
# Return the paragraph section name, the timer array
# and the timer label mapping.
###################################################################
sub parse_timer_selection {
  $_ = shift;
  my @lines = split /\n/;
  my $section_name = shift @lines;
  my @timers;
  my %timer_label;
  if ( $section_name !~ /^#/ ) {
    @lines = grep !/^#/, @lines; # Remove comment lines.
    for $line (@lines) {
      my ($timer,$label) = split /\s+/, $line, 2;
      $label = $timer unless "$label";
      push @timers, $timer;
      $timer_label{$timer} = $label;
    }
  }

  ($section_name, \@timers, %timer_label);
}


1;
