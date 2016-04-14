CHECK_ICC = $(shell which icc)
CHECK_NVCC = $(shell which nvcc)
NVCC     = nvcc
CXXFLAGS = -std=c++11 -O3
NVCCFLAGS= -std=c++11 -O3
LDFLAGS  = -lpthread
NVLDFLAGS= -L${CUDA_HOME}/lib64 -lcuda -lcudart
SRC      = $(wildcard *.cpp)
SRC     := $(filter-out scalingtests_memtest.cpp, $(SRC))
CUSRC    = $(wildcard *.cu)
OBJS     = $(SRC:.cpp=.o)
MICOBJS  = $(SRC:.cpp=.mic)
CUOBJS   = $(CUSRC:.cu=.cuo)
OTHROBJS = ../sorting/sorting.o ../selection/selection.o ../aggregation/group.o \
           ../preprocessing/loaddata.o ../timer/timer.o
OTHRMICOBJS = ../sorting/sorting.mic ../selection/selection.mic \
			  ../aggregation/group.mic ../preprocessing/loaddata.mic \
			  ../timer/timer.mic
OTHERCUOBJS = ../sorting/sorting.cuo ../selection/selection.cuo \
			  ../aggregation/group.cuo ../preprocessing/loaddata.o \
			  ../timer/timer.o	
EXP      = scalingtests.out
MICEXP   = scalingtests.mic.out
CUDAEXP  = scalingtests.cu.out

ifneq (,$(CHECK_ICC))

CC = icc
all : $(EXP) $(MICEXP)

else

CC = g++

ifneq (,$(CHECK_NVCC))

all : $(EXP) $(CUDAEXP)

else

all : $(EXP)

endif
endif

$(EXP) : $(OBJS) $(OTHROBJS)
	$(CC) $^ -o $@ $(LDFLAGS)

$(MICEXP) : $(MICOBJS) $(OTHRMICOBJS)
	$(CC) -mmic $^ -o $@ $(LDFLAGS)

$(CUDAEXP) : $(CUOBJS) $(OTHERCUOBJS)
	$(CC) $^ -o $@ $(NVLDFLAGS)

%.o : %.cpp
	$(CC) $(CXXFLAGS) -c $^ -o $@

%.mic : %.cpp
	$(CC) -mmic $(CXXFLAGS) -c $^ -o $@

%.cuo : %.cu
	$(NVCC) $(NVCCFLAGS) -c $^ -o $@

.PHONY : clean cleanall

clean :
	rm -vf $(EXP)
	rm -vf $(OBJS)
ifneq (,$(CHECK_NVCC))
	rm -vf $(CUDAEXP)
	rm -vf $(CUOBJS)
endif
ifneq (,$(CHECK_ICC))
	rm -vf $(MICEXP)
	rm -vf $(MICOBJS)
endif

cleanall: clean
	rm -vf $(OTHROBJS)
ifneq (,$(CHECK_NVCC))
	rm -vf $(OTHERCUOBJS)
endif
ifneq (,$(CHECK_ICC))
	rm -vf $(OTHRMICOBJS)
endif
