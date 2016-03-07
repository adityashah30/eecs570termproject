CC       = g++
CXXFLAGS = -std=c++11 -O3
LDFLAGS  = -lpthread
INCFLAGS = -I$(BOOST_HOME)/include
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
