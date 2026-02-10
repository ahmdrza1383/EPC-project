#ifndef DIM_UNIT_H
#define DIM_UNIT_H

#include <systemc.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "config.h"
#include "Spiral_ALU.h"

SC_MODULE(Dim_Unit) {
    sc_in<bool> clk;
    sc_in<bool> start;
    sc_out<bool> done;

    // --- هویت این واحد ---
    int my_d1; // من مسئول کدام بُعد هستم؟

    // --- ورودی‌های داده (Shared Memory) ---
    double* current_pos_ptr; // کل آرایه موقعیت‌ها
    double* iter_best_pos_ptr;
    double* Q_val_ptr;       // مقدار Q محاسبه شده در لایه بالا
    double* M_ptr;

    // --- خروجی ---
    // هر واحد فقط خانه مربوط به خودش را در آرایه جدید می‌نویسد
    double* next_pos_ptr;    

    Spiral_ALU* alu;

    void process_logic() {
        while (true) {
            wait();
            if (start.read()) {
                done.write(false);

                double accumulated_move = 0;
                int move_count = 0;

                // 1. منطق انتخاب همسایه (فقط برای بُعد من: my_d1)
                std::vector<int> partners;

                #if STRATEGY_ID == 1 // RANDOM
                    int attempts = 0;
                    // انتخاب چند شریک تصادفی برای این بُعد
                    while(partners.size() < N_NEIGHBORS && attempts < DIM*2) {
                        int rand_d2 = rand() % DIM;
                        if (rand_d2 != my_d1) {
                            bool exists = false;
                            for(int p : partners) if(p == rand_d2) exists = true;
                            if(!exists) partners.push_back(rand_d2);
                        }
                        attempts++;
                    }
                #else // ALL_PAIRS
                    // در حالت All-Pairs سخت‌افزاری، هر بُعد با همه ابعاد دیگر تعامل می‌کند
                    for(int d2 = 0; d2 < DIM; d2++) {
                        if (d2 != my_d1) partners.push_back(d2);
                    }
                #endif

                // 2. انجام محاسبات برای شرکا
                for (int d2 : partners) {
                    double nx, ny; // ny استفاده نمی‌شود چون ما مسئول d1 هستیم
                    
                    // فراخوانی واحد ریاضی
                    alu->compute(current_pos_ptr[my_d1], current_pos_ptr[d2], 
                                 iter_best_pos_ptr[my_d1], iter_best_pos_ptr[d2], 
                                 *Q_val_ptr, nx, ny);
                    
                    accumulated_move += nx;
                    move_count++;
                }

                // 3. مدل‌سازی زمان اجرا
                // چون این ماژول‌ها موازی هستند، این زمان فقط یکبار محاسبه می‌شود
                wait(5, SC_NS); 

                // 4. میانگین‌گیری و نویز
                double new_val = current_pos_ptr[my_d1];
                if (move_count > 0) {
                    new_val = accumulated_move / move_count;
                }

                // نویز
                double noise = (*M_ptr) * ((rand() / (double)RAND_MAX) * 2 - 1);
                new_val += noise;

                // بررسی مرزها
                if (new_val > UB) new_val = UB;
                if (new_val < LB) new_val = LB;

                // نوشتن در آرایه خروجی (فقط ایندکس خودم)
                next_pos_ptr[my_d1] = new_val;

                done.write(true);
            }
        }
    }

    SC_CTOR(Dim_Unit) {
        alu = new Spiral_ALU("ALU");
        SC_THREAD(process_logic);
        sensitive << clk.pos();
        done.initialize(true);
    }
};

#endif // DIM_UNIT_H