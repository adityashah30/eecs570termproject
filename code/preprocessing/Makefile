CC       = g++
CXXFLAGS = -std=c++11
SRC      = $(wildcard *.cpp)
OBJS     = $(patsubst %.cpp,%.o,$(SRC))
EXEC     = loaddatatest.out

$(EXEC) : $(OBJS)
	$(CC) $^ -o $@

%.o : %.cpp
	$(CC) $(CXXFLAGS) -c $^ -o $@

.PHONY : clean

clean :
	rm -vf $(EXEC)
	rm -vf $(OBJS)
