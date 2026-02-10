# 1. تنظیم دقیق مسیر SystemC (بر اساس ارور قبلی شما)
SYSTEMC_HOME = /usr/local/systemc-2.3.1

# 2. تنظیم کامپایلر
CXX = g++

# 3. فلگ‌های کامپایل (شامل مسیر Include)
# نکته مهم: -I به کامپایلر می‌گوید systemc.h کجاست
CXXFLAGS = -I$(SYSTEMC_HOME)/include -O2 -std=c++14

# 4. فلگ‌های لینکر (شامل مسیر Library و Rpath)
# نکته: Wl,-rpath باعث می‌شود موقع اجرا دیگر نیازی به export نباشد
LDFLAGS = -L$(SYSTEMC_HOME)/lib-linux64 -lsystemc -lm -Wl,-rpath=$(SYSTEMC_HOME)/lib-linux64

# 5. نام فایل خروجی و ورودی‌ها
TARGET = epc_sim
SRCS = main.cpp
HEADERS = config.h Spiral_ALU.h Dim_Unit.h Cost_Unit.h Penguin_Core.h

# 6. دستور پیش‌فرض
all: $(TARGET)

# 7. قانون ساخت
$(TARGET): $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

# 8. تمیزکاری
clean:
	rm -f $(TARGET) *.vcd