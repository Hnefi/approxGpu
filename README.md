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

#NOTE if you just type "make" it will not work as it tries to also compile a "matlab version" of the same code and run that for comparison/error tracking. Could change the makefile dependencies to fix this.


#### FIRST ISSUE WE NEED TO FIGURE OUT ####
- for some reason, when you compile using nvcc, it re-compiles ALL of the *.cu files in the /common/c and /common/kernels directories, regardless if none of them or only some of them have been changed.
- this takes 5+ minutes per compile which is stupid long. (i don't think nvcc is very good at working with double precision math but this is just from reading forum posts on long compilation times)
- TODO: Need to somehow change nvcc flags or output or something so that it can link unchanged files without having to redo the whole thing every time you add a print statement.

#/readme
