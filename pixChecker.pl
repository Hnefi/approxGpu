use strict;
use warnings;

my $fileName = shift; #fes2_results_XXXX.stats
my $fileTag="NULL";
if($fileName =~ m/([a-z]+.dump.*)/)
{
    print "Setting fileTag properly to $1\n";
	$fileTag=$1;
}

open FILE, $fileName;
my $oldNum = -1;
my $elemNum = 0;
my $gpuVal;
my $cpuVal;

print "Starting file match....\n";
while(<FILE>)
{
    if(m/Element # (\d+) /) { #match "Element # <elemNum>
        $oldNum = $elemNum;
        $elemNum = $1;
        if(m/.*CPU.*: ([0-9]*\.[0-9+])/) { # match CPU number
            $cpuVal = $1;
        } elsif(m/.*GPU.*: ([0-9]*\.[0-9+])/) { # match GPU number
            $gpuVal = $1;
        } 
    }

    if( ($oldNum == $elemNum) && $elemNum > 0 ) {
        # check number equality
        if ( $gpuVal != $cpuVal ) {
            print "Element # $elemNum of GPU and CPU do not match!. GPU: $gpuVal , CPU: $cpuVal\n";
        } 
    } 
}
