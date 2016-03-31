CHECK_ICC= $(shell which icc)
CC       = g++
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
MICEXP   = scalingtests.out.mic

all : compiler_check

compiler_check :
ifneq (,$(CHECK_ICC))
	@echo "Found ICC. Using ICC"
	$(eval CC = icc)
	make $(EXP)
	make $(MICEXP)
else
	@echo "Did not find ICC. Using G++"
	$(eval CC = g++)
	make $(EXP)
endif

$(EXP) : $(OBJS) $(OTHROBJS)
	$(CC) $^ -o $@ $(LDFLAGS)

$(MICEXP) : $(MICEXP) $(OTHRMICOBJS)
	$(CC) -mmic $^ -o $@ $(LDFLAGS)

%.o : %.cpp
	$(CC) $(CXXFLAGS) -c $^ -o $@

%.mic : %.cpp
	$(CC) -mmic $(CXXFLAGS) -c $^ -o $@

.PHONY : clean cleanall

clean :
	rm -vf $(EXP)
	rm -vf $(OBJS)
	rm -vf $(MICOBJS)

cleanall: clean
	rm -vf $(OTHROBJS)
	rm -vf $(OTHRMICOBJS)
