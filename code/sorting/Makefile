CC       = g++
NVCC     = nvcc
CXXFLAGS = -std=c++11 -O3
NVCCFLAGS= -std=c++11 -O3
LDFLAGS  = -lpthread
NVLDFLAGS= -L${CUDA_HOME}/lib64 -lcuda -lcudart
SRC      = $(wildcard *.cpp)
CUSRC    = $(wildcard *.cu)
OBJS     = $(SRC:.cpp=.o)
CUOBJS   = $(CUSRC:.cu=.cuo)
TEST     = sortingtest.out
EXP      = sortingexp.out
BITONIC  = bitonic.out
CUDATEST = sortingtest.cu.out
CUDAEXP  = sortingexp.cu.out

all : $(TEST) $(EXP) $(CUDATEST) $(CUDAEXP)

$(CUDATEST) : sortingtest.cuo sorting.cuo ../preprocessing/loaddata.o ../timer/timer.o
	$(CC) $^ -o $@ $(NVLDFLAGS)

$(CUDAEXP) : sortingexp.cuo sorting.cuo ../preprocessing/loaddata.o ../timer/timer.o
	$(CC) $^ -o $@ $(NVLDFLAGS)

$(TEST) : sortingtest.o sorting.o ../preprocessing/loaddata.o ../timer/timer.o
	$(CC) $^ -o $@ $(LDFLAGS)

$(EXP) : sortingexp.o sorting.o ../preprocessing/loaddata.o ../timer/timer.o
	$(CC) $^ -o $@ $(LDFLAGS)

%.o : %.cpp
	$(CC) $(CXXFLAGS) -c $^ -o $@

%.cuo : %.cu
	$(NVCC) $(NVCCFLAGS) -c $^ -o $@

.PHONY : clean

clean :
	rm -vf $(TEST)
	rm -vf $(EXP)
	rm -vf $(CUDATEST)
	rm -vf $(CUDAEXP)
	rm -vf $(OBJS)
	rm -vf $(CUOBJS)
