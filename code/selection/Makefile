
CC       = icc
CXXFLAGS = -std=c++11 -mmic -O3 -qopt-report=5 -qopt-report-phase=vec
LDFLAGS  = -lpthread 
INCFLAGS = -I$(BOOST_HOME)/include
SRC      = $(wildcard *.cpp)
OBJS     = $(patsubst %.cpp,%.o,$(SRC))
EXEC     = selectiontest.out

$(EXEC) : $(OBJS) ../preprocessing/loaddata.o ../timer/timer.o
	$(CC) $^ -o $@ $(LDFLAGS)

%.o : %.cpp
	$(CC) $(INCFLAGS) $(CXXFLAGS) -c $^ -o $@

.PHONY : clean

clean :
	rm -vf $(EXEC)
	rm -vf $(OBJS)
	rm -vf ../preprocessing/loaddata.o ../timer/timer.o
