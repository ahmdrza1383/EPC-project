SYSTEMC_HOME ?= /usr/local/systemc-2.3.1
TARGET_ARCH  = linux64


CXX          = g++
CXXFLAGS     = -std=c++14 -Wall -Wno-deprecated -g
INCLUDES     = -I. -I./include -I$(SYSTEMC_HOME)/include
LDFLAGS      = -L$(SYSTEMC_HOME)/lib-$(TARGET_ARCH) -lsystemc -lm
RPATH        = -Wl,-rpath=$(SYSTEMC_HOME)/lib-$(TARGET_ARCH)

SRCS         = $(wildcard src/*.cpp) \
               $(wildcard src/hw/*.cpp) \
               $(wildcard src/sw/*.cpp) \
               $(wildcard testbench/*.cpp)

OBJS         = $(patsubst %.cpp, build/%.o, $(SRCS))

TARGET       = epc_sim

all: dir $(TARGET)


$(TARGET): $(OBJS)
	@echo "Linking target: $@"
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(RPATH)


build/%.o: %.cpp
	@mkdir -p $(dir $@)
	@echo "Compiling: $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@


dir:
	@mkdir -p build

clean:
	@echo "Cleaning up..."
	rm -rf build $(TARGET) *.vcd

.PHONY: all clean dir