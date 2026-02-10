SYSTEMC_HOME = /usr/local/systemc-2.3.1

CXX = g++

CXXFLAGS = -I$(SYSTEMC_HOME)/include -O2 -std=c++14

LDFLAGS = -L$(SYSTEMC_HOME)/lib-linux64 -lsystemc -lm -Wl,-rpath=$(SYSTEMC_HOME)/lib-linux64

TARGET = epc_sim
SRCS = main.cpp
HEADERS = config.h Spiral_ALU.h Dim_Unit.h Cost_Unit.h Penguin_Core.h

all: $(TARGET)

$(TARGET): $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.vcd