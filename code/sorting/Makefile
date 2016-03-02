CC       = g++
CXXFLAGS = -std=c++11
LDFLAGS  = -lpthread
INCFLAGS = -I$(BOOST_HOME)/include
SRC      = $(wildcard *.cpp)
OBJS     = $(patsubst %.cpp,%.o,$(SRC))
EXEC     = sortingtest.out

$(EXEC) : $(OBJS) ../preprocessing/loaddata.o
	$(CC) $^ -o $@ $(LDFLAGS)

%.o : %.cpp
	$(CC) $(INCFLAGS) $(CXXFLAGS) -c $^ -o $@

.PHONY : clean

clean :
	rm -vf $(EXEC)
	rm -vf $(OBJS)
