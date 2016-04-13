CC       = g++
NVCC     = nvcc
CXXFLAGS = -std=c++11 -O3
NVCCFLAGS= -std=c++11 -O3
LDFLAGS  = -lpthread
NVLDFLAGS= -L${CUDA_HOME}/lib64 -lcudart
SRC      = $(wildcard *.cpp)
CUSRC    = $(wildcard *.cu)
OBJS     = $(SRC:.cpp=.o)
CUOBJS   = $(CUSRC:.cu=.cuo)
TEST     = grouptest.out
EXP      = groupexp.out
CUDATEST = grouptest.cu.out
CUDAEXP  = groupexp.cu.out

all : $(TEST) $(EXP) $(CUDATEST) $(CUDAEXP)

$(CUDATEST) : grouptest.cuo group.cuo ../preprocessing/loaddata.o ../timer/timer.o
	$(CC) $^ -o $@ $(NVLDFLAGS)

$(CUDAEXP) : groupexp.cuo group.cuo ../preprocessing/loaddata.o ../timer/timer.o
	$(CC) $^ -o $@ $(NVLDFLAGS)

$(TEST) : grouptest.o group.o ../preprocessing/loaddata.o ../timer/timer.o
	$(CC) $^ -o $@ $(LDFLAGS)

$(EXP) : groupexp.o group.o ../preprocessing/loaddata.o ../timer/timer.o
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