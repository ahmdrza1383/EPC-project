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

    // --- هویت ---
    int my_d1;

    // --- حافظه مشترک ---
    // نکته: این‌ها پوینتر هستند و باید قبل از استفاده مقداردهی شوند
    double* current_pos_ptr;
    double* iter_best_pos_ptr;
    double* Q_val_ptr;
    double* M_ptr;

    // --- خروجی ---
    double* next_pos_ptr;    

    Spiral_ALU* alu;

    void process_logic() {
        while (true) {
            wait();
            if (start.read()) {
                done.write(false);

                // محافظت در برابر پوینترهای نال (برای جلوگیری از Segfault)
                if (!current_pos_ptr || !iter_best_pos_ptr || !Q_val_ptr || !M_ptr || !next_pos_ptr) {
                    std::cout << "Error: Null pointer in Dim_Unit_" << my_d1 << std::endl;
                    done.write(true);
                    continue;
                }

                double accumulated_move = 0;
                int move_count = 0;

                std::vector<int> partners;

                // انتخاب همسایگان
                #if STRATEGY_ID == 1 // RANDOM
                    int attempts = 0;
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
                    for(int d2 = 0; d2 < DIM; d2++) {
                        if (d2 != my_d1) partners.push_back(d2);
                    }
                #endif

                // محاسبات موازی
                for (int d2 : partners) {
                    double nx, ny;
                    alu->compute(current_pos_ptr[my_d1], current_pos_ptr[d2], 
                                 iter_best_pos_ptr[my_d1], iter_best_pos_ptr[d2], 
                                 *Q_val_ptr, nx, ny);
                    
                    accumulated_move += nx;
                    move_count++;
                }

                // شبیه‌سازی زمان
                wait(5, SC_NS); 

                // میانگین‌گیری
                double new_val = current_pos_ptr[my_d1];
                if (move_count > 0) {
                    new_val = accumulated_move / move_count;
                }

                // نویز
                double noise = (*M_ptr) * ((rand() / (double)RAND_MAX) * 2 - 1);
                new_val += noise;

                // مرزها
                if (new_val > UB) new_val = UB;
                if (new_val < LB) new_val = LB;

                // نوشتن خروجی
                next_pos_ptr[my_d1] = new_val;

                done.write(true);
            }
        }
    }

    SC_CTOR(Dim_Unit) {
        alu = new Spiral_ALU("ALU");
        alu->clk(clk); 
        
        // مقداردهی اولیه پوینترها به nullptr برای امنیت
        current_pos_ptr = nullptr;
        iter_best_pos_ptr = nullptr;
        Q_val_ptr = nullptr;
        M_ptr = nullptr;
        next_pos_ptr = nullptr;

        SC_THREAD(process_logic);
        sensitive << clk.pos();
        set_stack_size(0x10000);
        done.initialize(true);
    }
};

#endif // DIM_UNIT_H