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
TEST     = selectiontest.out
EXP      = selectionexp.out
CUDATEST = selectiontest.cu.out
CUDAEXP  = selectionexp.cu.out

all : $(TEST) $(EXP) $(CUDATEST) $(CUDAEXP)

$(CUDATEST) : selectiontest.cuo selection.cuo ../preprocessing/loaddata.o ../timer/timer.o
	$(CC) $^ -o $@ $(NVLDFLAGS)

$(CUDAEXP) : selectionexp.cuo selection.cuo ../preprocessing/loaddata.o ../timer/timer.o
	$(CC) $^ -o $@ $(NVLDFLAGS)

$(TEST) : selectiontest.o selection.o ../preprocessing/loaddata.o ../timer/timer.o
	$(CC) $^ -o $@ $(LDFLAGS)

$(EXP) : selectionexp.o selection.o ../preprocessing/loaddata.o ../timer/timer.o
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
