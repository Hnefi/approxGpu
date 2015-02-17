# approxGpu

#### Larger SD-VBS readme file is also uploaded to this repo. (SD-VBS.pdf)

#### DIRECTORIES ####
/benchmarks/tracking - contains the following for image tracking benchmark (I excluded the rest of the sd-vbs from this commit to save space with big image files)

  /src/c - has all of the *.cu files that are specific to feature tracking

  /data/* - has image files and results files that test different input sizes, from "sim_fast" all the way up to "fullhd"
  
/common - has all of the common .cu files that are general image processes.

  /c - basic things like image blurring, sobel filters (whatever these are), etc that feature tracking uses.

  /kernels - my own folder that has the kernel-ized versions of these functions, which are called from the wrapper functions in /common/c
  (I did it this way to not break their code compatibility because the directory/makefile structure is complex.)

  /makefiles - contains their top level makefiles that are included from all benchmarks (if we were using the full sd-vbs this would be more useful than it seems here, but I didn't want to re-write it so still left it this way)
  
/cycles - output times for different runs of all sd-vbs benchmarks. i haven't used this in a while so it's probably full of old debugging runs.

#### COMPILING/RUNNING ####
cd /benchmarks/tracking/data/sim_fast (or other image size)
make compile # outputs the binary IN this specific data folder
make c-run # runs the binary

####NOTE if you just type "make" it will not work as it tries to also compile a "matlab version" of the same code and run that for comparison/error tracking. Could change the makefile dependencies to fix this.

#TODO's
PARALLELIZE THE BIG FUNCTIONS. (blur, resize, things that touch all of the pixels and are well suited to GPU-izing them).
One thing to remember throughout is that putting somehting on the GPU means it will eventually be "approximated", so a direct
matching algorithm might not make sense or will need to be replaced with "approximate matching" as we continue into the guts
of the feature tracking algorithm.

#CURRENT STATUS
- Blur & resize work on the following images: sim_fast, sim, sqcif, qcif, cif...
- But larger images like vga and fullhd for some reason do not... I can't tell why this is, and I'm just going to blindly
continue on without worrying about it until later on since these smaller image tracks can be worked with in the meantime.

#/readme
