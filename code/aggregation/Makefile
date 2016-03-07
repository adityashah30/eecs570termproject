CC       = g++
CXXFLAGS = -std=c++11
LDFLAGS  = -lpthread
INCFLAGS = -I/home/wangyut/eecs570/boost_1_60_0/jb/boost/include
SRC      = $(wildcard *.cpp)
OBJS     = $(patsubst %.cpp,%.o,$(SRC))
EXEC     = grouptest.out

$(EXEC) : $(OBJS) ../preprocessing/loaddata.o ../timer/timer.o
	$(CC) $^ -g -o $@ $(LDFLAGS)

%.o : %.cpp
	$(CC) $(INCFLAGS) $(CXXFLAGS) -g -c $^ -o $@

.PHONY : clean

clean :
	rm -vf $(EXEC)
	rm -vf $(OBJS)