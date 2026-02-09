# مسیر نصب SystemC خود را اینجا تنظیم کنید
SYSTEMC_HOME ?= /usr/local/systemc-2.3.1

CXX = g++
CXXFLAGS = -I$(SYSTEMC_HOME)/include -O2 -std=c++11
LDFLAGS = -L$(SYSTEMC_HOME)/lib-linux64 -lsystemc -lm

TARGET = epc_sim
SRCS = main.cpp

all: $(TARGET)

$(TARGET): $(SRCS) epc_modules.h
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

clean:
	rm -f $(TARGET)