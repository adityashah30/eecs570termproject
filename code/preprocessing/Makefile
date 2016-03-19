CC       = g++
CXXFLAGS = -std=c++11 -O3
INCFLAGS = -I$(BOOST_HOME)/include
SRC      = $(wildcard *.cpp)
OBJS     = $(patsubst %.cpp,%.o,$(SRC))
EXEC     = loaddatatest.out

$(EXEC) : $(OBJS)
	$(CC) $^ -o $@

%.o : %.cpp
	$(CC) $(INCFLAGS) $(CXXFLAGS) -c $^ -o $@

.PHONY : clean

clean :
	rm -vf $(EXEC)
	rm -vf $(OBJS)
