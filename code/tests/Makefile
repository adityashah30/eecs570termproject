CC       = g++
CXXFLAGS = -std=c++11 -O3
LDFLAGS  = -lpthread
INCFLAGS = -I$(BOOST_HOME)/include
SRC      = $(wildcard *.cpp)
OBJS     = $(patsubst %.cpp,%.o,$(SRC))
OTHROBJS = ../sorting/sorting.o ../selection/selection.o ../aggregation/group.o\
           ../preprocessing/loaddata.o ../timer/timer.o
EXP      = scalingtests.out

all : $(EXP)

$(EXP) : $(OBJS) $(OTHROBJS)
	$(CC) $^ -o $@ $(LDFLAGS)

%.o : %.cpp
	$(CC) $(INCFLAGS) $(CXXFLAGS) -c $^ -o $@

.PHONY : clean cleanall

clean :
	rm -vf $(EXP)
	rm -vf $(OBJS)

cleanall: clean
	rm -vf $(OTHROBJS)
