CHECK_ICC= $(shell which icc)

CXXFLAGS = -std=c++11 -O3
LDFLAGS  = -lpthread
SRC      = $(wildcard *.cpp)
OBJS     = $(SRC:.cpp=.o)
MICOBJS  = $(SRC:.cpp=.mic)
OTHROBJS = ../sorting/sorting.o ../selection/selection.o ../aggregation/group.o \
           ../preprocessing/loaddata.o ../timer/timer.o
OTHRMICOBJS = ../sorting/sorting.mic ../selection/selection.mic \
			  ../aggregation/group.mic ../preprocessing/loaddata.mic \
			  ../timer/timer.mic
EXP      = scalingtests.out
MICEXP   = scalingtests.mic.out

ifneq (,$(CHECK_ICC))

CC = icc
all : $(EXP) $(MICEXP)

else

CC = g++
all : $(EXP)

endif

$(EXP) : $(OBJS) $(OTHROBJS)
	$(CC) $^ -o $@ $(LDFLAGS)

$(MICEXP) : $(MICOBJS) $(OTHRMICOBJS)
	$(CC) -mmic $^ -o $@ $(LDFLAGS)

%.o : %.cpp
	$(CC) $(CXXFLAGS) -c $^ -o $@

%.mic : %.cpp
	$(CC) -mmic $(CXXFLAGS) -c $^ -o $@

.PHONY : clean cleanall

clean :
	rm -vf $(EXP)
	rm -vf $(OBJS)
ifneq (,$(CHECK_ICC))
	rm -vf $(MICOBJS)
endif

cleanall: clean
	rm -vf $(OTHROBJS)
ifneq (,$(CHECK_ICC))
	rm -vf $(OTHRMICOBJS)
endif
