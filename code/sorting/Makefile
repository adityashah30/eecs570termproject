CC       = g++
CXXFLAGS = -std=c++11 -O3
LDFLAGS  = -lpthread
INCFLAGS = -I$(BOOST_HOME)/include
SRC      = $(wildcard *.cpp)
OBJS     = $(patsubst %.cpp,%.o,$(SRC))
TEST     = sortingtest.out
EXP      = sortingexp.out

all : $(TEST) $(EXP)

$(TEST) : sortingtest.o sorting.o ../preprocessing/loaddata.o ../timer/timer.o
	$(CC) $^ -o $@ $(LDFLAGS)

$(EXP) : sortingexp.o sorting.o ../preprocessing/loaddata.o ../timer/timer.o
	$(CC) $^ -o $@ $(LDFLAGS)

%.o : %.cpp
	$(CC) $(INCFLAGS) $(CXXFLAGS) -c $^ -o $@

.PHONY : clean

clean :
	rm -vf $(TEST)
	rm -vf $(EXP)
	rm -vf $(OBJS)
