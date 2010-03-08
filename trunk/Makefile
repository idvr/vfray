#
# Makefile CPU
#

CXX = g++

INCLUDES =

LIBDIR =

OBJS = main.o vfRayPreComp.o vfRayKernelCPU.o

SRCS = $(OBJS:.o=.cc)

APP = vfRay

DEBUGFLAGS = -g
OPTFLAGS = -O3 -ffast-math

FLAGS = $(DEBUGFLAGS) \
	$(OPTFLAGS) \
	-Wall -Wno-deprecated \
	$(INCLUDES)

LIBS =

#-----------------------------------------------------------------------------

$(APP): $(OBJS)
	@echo "Linking ..."
	$(CXX) $(FLAGS) -o $(APP) $(OBJS) $(LIBDIR) $(LIBS)

depend:
	rm -f .depend
	$(CXX) -M $(FLAGS) $(SRCS) > .depend

.cc.o: $*.h
	@echo "Compiling ..."
	$(CXX) $(FLAGS) -c $*.cc

clean:
	rm -f *.o *~ \#* $(APP) .depend

ifeq (.depend,$(wildcard .depend))
include .depend
endif
