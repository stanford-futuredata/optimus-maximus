
ifeq ($(DEBUG), 1)
  CXX_FLAGS := -g -O0
else
  CXX_FLAGS := -O3
endif

CXX := icc
CXX_FLAGS += -std=c++11 -D ICC -mkl -ipp
INCLUDE_DIRS := /usr/include/boost
LIBRARY_DIRS := /usr/lib/x86_64-linux-gnu
LIBRARIES := boost_system boost_program_options

INCLUDES := $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
L_FLAGS := $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
                $(foreach library,$(LIBRARIES),-l$(library))

SRCS = main.cpp algo.cpp arith.cpp parser.cpp
OBJS = $(SRCS:.cpp=.o)

# define the executable file 
MAIN = simdex

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
	$(RM) *.o *~ $(MAIN)
