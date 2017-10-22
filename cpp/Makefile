
ifeq ($(DEBUG), 1)
  CXX_FLAGS := -g -O0 -D DEBUG
else
  CXX_FLAGS := -O3
endif

ifeq ($(STATS), 1)
  CXX_FLAGS += -D STATS
endif

CXX := g++-4.8
CXX_FLAGS += -std=c++11 -DMKL_ILP64 -m64
INCLUDE_DIRS := /usr/include/boost $(HOME)/packages/include ${MKLROOT}/include ${IPPROOT}/include
LIBRARY_DIRS := /usr/lib/x86_64-linux-gnu $(HOME)/packages/lib ${MKLROOT}/lib/intel64 ${IPPROOT}/lib/intel64
LIBRARIES := boost_system boost_program_options armadillo mkl_intel_ilp64 mkl_sequential mkl_core pthread m dl ipps


INCLUDES := $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
L_FLAGS := $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
                -Wl,--no-as-needed $(foreach library,$(LIBRARIES),-l$(library))

SRCS = main.cpp algo.cpp arith.cpp parser.cpp clustering/cluster.cpp
OBJS = $(SRCS:.cpp=.o)

MAIN_DEBUG := simdex_debug
MAIN_STATS := simdex_stats
MAIN_RELEASE := simdex

# define the executable file 
ifeq ($(DEBUG), 1)
  MAIN := $(MAIN_DEBUG)
else ifeq ($(STATS), 1)
  MAIN := $(MAIN_STATS)
else
  MAIN := $(MAIN_RELEASE)
endif

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
	$(RM) *.o clustering/*.o *~ $(MAIN_DEBUG) $(MAIN_STATS) $(MAIN_RELEASE)
