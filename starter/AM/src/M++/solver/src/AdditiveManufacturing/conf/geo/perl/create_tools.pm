# $Header: /public/M++/tools/create_tools.pm,v 1.1 2007-07-20 08:28:34 sydow Exp $
# ----------------------------------------------------------------------
# file:    M++/tools/create_tools.pm
# author:  Antje Sydow
# date:    begin: July 17, 2007; last modified: July 17, 2007
# purpose: contains the subroutines to be used in create_geo_* files
# usage:   include 'use create_tools;' in script

package create_tools;

use warnings;
use strict;

our (@ISA, @EXPORT, @EXPORT_OK, %EXPORT_TAGS, $VERSION);
use Exporter;
@ISA     = qw(Exporter);
@EXPORT  = qw(initialize_geofile
	      write_title
	      write_point 
	      write_hex8 
	      write_hex20 
	      write_hex27 
	      write_face4);

# ----------------------------------------------------------------------
# Definitions and Settings
# ----------------------------------------------------------------------
my $fform = '%2.6f';    # C-style format for output of floats
my $iform = '%3i';      # C-style format for output of integers
my $sform = '%s';       # C-style format for output of strings

# ---------------------------------

my $real   = '[+-]*\d+\.*\d*e[+-]\d+|[+-]*\d*\.\d*|[+-]*\d+';
my $int    = '[+-]*\d+';
my $string = '\w+';

# ---------------------------------
# exported functions
# ---------------------------------
sub initialize_geofile {
    # purpose:        create an empty .geo-file
    # call arguments: [0] name .geo-file
    my (@F) = @_;
    open(OUT,">","$F[0]") || die "Cannot open $F[0]!\n";
    close(OUT);
}

sub write_title {
    # purpose:        write section title to .geo-file
    # call arguments: [0] name .geo-file
    #                 [1] title
    my (@F) = @_;
    open(OUT,">>","$F[0]") || die "Cannot open $F[0]!\n";
    printf OUT ("$F[1] \n");
    close(OUT);
}

sub write_point {
    # purpose:        write point to .geo-file
    # call arguments: [0] name .geo-file
    #                 [1] x-coordinate
    #                 [2] y-coordinate
    #                 [3] z-coordinate
    my (@F) = @_;
    open(OUT,">>","$F[0]") || die "Cannot open $F[0]!\n";
    printf OUT ("$fform\ $fform\ $fform \n",
		 $F[1],  $F[2],  $F[3]);
    close(OUT);
}

sub write_hex8 {
    # purpose: write hexahedron with 8 geometry nodes to .geo-file
    # call arguments: [0] name .geo-file
    #                 [1] number of nodes per cell
    #                 [2] subdomain id
    #                 [3] comment, i.e. cell position
    #                 [4]-[11] node numbers
    my (@F) = @_;
    open(OUT,">>","$F[0]") || die "Cannot open $F[0]!\n";

    printf OUT "$F[1]\ $F[2]\ ";
    printf OUT ("$iform $iform $iform $iform $iform $iform $iform $iform ",
		 $F[4], $F[5], $F[6], $F[7], $F[8], $F[9], $F[10], $F[11]);
   
    printf OUT "\t // $F[3] \n";
    close(OUT);
}

sub write_hex20 {
    # purpose: write hexahedron with 20 geometry nodes to .geo-file
    # call arguments: [0] name .geo-file
    #                 [1] number of nodes per cell
    #                 [2] subdomain id
    #                 [3] comment, i.e. cell position
    #                 [4]-[23] node numbers
    my (@F) = @_;
    open(OUT,">>","$F[0]") || die "Cannot open $F[0]!\n";

    printf OUT "$F[1]\ $F[2]\ ";
    # corners
    printf OUT ("$iform $iform $iform $iform $iform $iform $iform $iform ",
		 $F[4], $F[5], $F[6], $F[7], $F[8], $F[9], $F[10], $F[11]);
    # edge centres
    printf OUT ("$iform $iform $iform $iform ",  
		$F[12], $F[13], $F[14], $F[15]);
    printf OUT ("$iform $iform $iform $iform ", 
		$F[16], $F[17], $F[18], $F[19]);
    printf OUT ("$iform $iform $iform $iform ", 
		$F[20], $F[21], $F[22], $F[23]);
    
    printf OUT "\t // $F[3] \n";
}

sub write_hex27 {
    # purpose: write hexahedron with 27 geometry nodes to .geo-file
    # call arguments: [0] name .geo-file
    #                 [1] number of nodes per cell
    #                 [2] subdomain id
    #                 [3] comment, i.e. cell position
    #                 [4]-[30] node numbers
    my (@F) = @_;
    open(OUT,">>","$F[0]") || die "Cannot open $F[0]!\n";

    printf OUT "$F[1]\ $F[2]\ ";
    # corners
    printf OUT ("$iform $iform $iform $iform $iform $iform $iform $iform ",
		 $F[4], $F[5], $F[6], $F[7], $F[8], $F[9], $F[10], $F[11]);
    # edge centres
    printf OUT ("$iform $iform $iform $iform ",  
		$F[12], $F[13], $F[14], $F[15]);
    printf OUT ("$iform $iform $iform $iform ", 
		$F[16], $F[17], $F[18], $F[19]);
    printf OUT ("$iform $iform $iform $iform ", 
		$F[20], $F[21], $F[22], $F[23]);
    # face centres
    printf OUT ("$iform $iform $iform $iform $iform $iform ", 
		$F[24], 
		$F[25], $F[26], $F[27], $F[28], 
		$F[29]);
    # cell centre
    printf OUT ("$iform ", $F[30]);
    
    printf OUT "\t // $F[3] \n";
}

sub write_face4 {
    # purpose: write face to .geo-file
    # call arguments: [0] name .geo-file
    #                 [1] number of nodes per face
    #                 [2] boundary condition id
    #                 [3] comment, i.e. face position
    #                 [4]-[7] node numbers
    my (@F) = @_;
    open(OUT,">>","$F[0]") || die "Cannot open $F[0]!\n";
    
    printf OUT "$F[1] $F[2] ";
    printf OUT ("$iform $iform $iform $iform ",
		$F[4], $F[5], $F[6], $F[7]);

    printf OUT "\t // $F[3] \n";
}
