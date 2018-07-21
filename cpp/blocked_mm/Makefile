
ifeq ($(DEBUG), 1)
  CXX_FLAGS := -g -O0 -D DEBUG
else
  CXX_FLAGS := -O3
endif

INCLUDE_DIRS := $(HOME)/packages/include/boost
LIBRARY_DIRS := $(HOME)/packages/lib
LIBRARIES := boost_system boost_program_options

ifeq ($(ICC), 1)
  CXX := icc
  CXX_FLAGS += -std=c++11 -D MKL_ILP64 -mkl -fp-model precise
else
  # Use g++ instead of icc
  CXX := g++-4.8
  CXX_FLAGS += -std=c++11 -march=native
  ifeq ($(MKL), 1)
    # if MKL=1 is passed in, still use MKL
    CXX_FLAGS += -DMKL_ILP64 -m64
    INCLUDE_DIRS += ${MKLROOT}/include
    LIBRARY_DIRS += ${MKLROOT}/lib/intel64
    LIBRARIES += mkl_intel_ilp64 mkl_sequential mkl_core m dl
  else
    # otherwise, use OpenBlas
    LIBRARY_DIRS += /usr/lib/openblas-base
    LIBRARIES += openblas
  endif
endif


INCLUDES := $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
L_FLAGS := $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
                $(foreach library,$(LIBRARIES),-l$(library))

SRCS = blocked_mm.cpp ../parser.cpp
OBJS = $(SRCS:.cpp=.o)

# define the executable file
MAIN = blocked_mm

#
# The following part of the makefile is generic; it can be used to
# build any executable just by changing the definitions above and by
# deleting dependencies appended to the file from 'make depend'
#

.PHONY: clean

all: $(MAIN)

$(MAIN): $(OBJS)
	$(CXX) $(CXX_FLAGS) $(OBJS) $(L_FLAGS) -o $(MAIN)

# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .c file) and $@: the name of the target of the rule (a .o file)
# (see the gnu make manual section about automatic variables)
.cpp.o:
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $<  -o $@

clean:
	$(RM) *.o ../parser.o *~ $(MAIN)
