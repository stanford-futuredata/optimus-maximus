
CC = icc
CFLAGS = -O3 -std=c++11 -mkl -ipp
INCLUDES =
LIBS =
#LFLAGS = linker flags (if necessary)

SRCS = main.cpp mz_special.cpp arith.cpp parser.cpp
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
	$(CC) $(CFLAGS) $(OBJS) $(LIBS) -o $(MAIN) 

# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .c file) and $@: the name of the target of the rule (a .o file) 
# (see the gnu make manual section about automatic variables)
.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	$(RM) *.o *~ $(MAIN)
