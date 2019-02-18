
ifeq ($(DEBUG), 1)
  CXX_FLAGS := -g -O0 -D DEBUG
else
  CXX_FLAGS := -O3
endif

ifeq ($(NORMALIZE), 1)
  CXX_FLAGS += -D NORMALIZE
endif

ifeq ($(STATS), 1)
  CXX_FLAGS += -D STATS
endif

ifeq ($(NAIVE), 1)
  CXX_FLAGS += -D NAIVE
endif

ifeq ($(RULE), 1)
  CXX_FLAGS += -D ONLINE_DECISION_RULE
endif

ifeq ($(TEST_ONLY), 1)
  CXX_FLAGS += -D TEST_ONLY
endif


ifeq ($(ICC), 1)
  CXX := icc
  CXX_FLAGS += -std=c++11 -D MKL_ILP64 -mkl -ipp -fp-model precise
  INCLUDE_DIRS := $(HOME)/packages/include/boost $(HOME)/packages/include
  LIBRARY_DIRS := $(HOME)/packages/lib
  LIBRARIES := boost_system boost_program_options armadillo
else
  # Use g++ instead of icc
  CXX := g++-4.8
  CXX_FLAGS += -std=c++11 -march=native
  INCLUDE_DIRS := $(HOME)/packages/include/boost $(HOME)/packages/include ${IPPROOT}/include
  LIBRARY_DIRS := $(HOME)/packages/lib ${IPPROOT}/lib/intel64
  LIBRARIES := boost_system boost_program_options armadillo ipps
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
	$(CXX) $(CXX_FLAGS) $(OBJS) $(L_FLAGS) -fopenmp -o $(MAIN)

# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .c file) and $@: the name of the target of the rule (a .o file)
# (see the gnu make manual section about automatic variables)
.cpp.o:
	$(CXX) $(CXX_FLAGS) -fopenmp $(INCLUDES) -c $<  -o $@

clean:
	$(RM) *.o clustering/*.o *~ $(MAIN_DEBUG) $(MAIN_STATS) $(MAIN_RELEASE)
