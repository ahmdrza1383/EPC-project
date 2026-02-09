#ifndef PENGUIN_CORE_H
#define PENGUIN_CORE_H

#include <systemc.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include "config.h"
#include "Spiral_ALU.h"

SC_MODULE(Penguin_Core) {
    sc_in<bool> clk;
    sc_in<bool> start;
    sc_out<bool> done;

    // --- شناسه هویت (Identity) ---
    int core_id;                 // شماره من
    int* iter_best_idx_ptr;      // اشاره‌گر به شماره بهترین پنگوئن

    // حافظه داخلی
    double position[DIM];
    double accumulated_moves[DIM];
    double move_counts[DIM];

    // ورودی‌های مشترک
    double* iter_best_pos_ptr;
    double* M_ptr;
    double* mu_ptr;

    Spiral_ALU* alu;

    void process_logic() {
        while (true) {
            wait();
            if (start.read()) {
                done.write(false);

                // =========================================================
                // منطق دقیق پایتون: تفکیک رفتار بهترین پنگوئن
                // =========================================================
                
                // بررسی: آیا من بهترین پنگوئن این دور هستم؟
                bool is_best = (core_id == *iter_best_idx_ptr);

                if (is_best) {
                    // *** رفتار پنگوئن بهترین: حرکت اسپیرال ندارد ***
                    // فقط ۱۰ نانوثانیه صبر می‌کند (مدل‌سازی بیکاری یا تاخیر عبور سیگنال)
                    wait(10, SC_NS);
                    
                    // موقعیت تغییر نمی‌کند (فقط کپی می‌شود تا بعداً نویز بخورد)
                    // در واقع position[d] همان می‌ماند.
                } 
                else {
                    // *** رفتار پنگوئن معمولی: حرکت اسپیرال دارد ***

                    // 1. Reset Buffers
                    for(int i=0; i<DIM; i++) {
                        accumulated_moves[i] = 0;
                        move_counts[i] = 0;
                    }

                    // 2. Calc Q
                    double dist = 0;
                    for(int k=0; k<DIM; k++) {
                        double diff = position[k] - iter_best_pos_ptr[k];
                        dist += diff * diff;
                    }
                    dist = std::sqrt(dist);
                    double Q = std::exp(-(*mu_ptr) * dist);
                    if (Q > 1) Q=1; if (Q < 1e-4) Q=1e-4;

                    // 3. Movement Logic
                    for (int d1 = 0; d1 < DIM; d1++) {
                        std::vector<int> partners;

                        #if STRATEGY_ID == 1 // RANDOM
                            int attempts = 0;
                            while(partners.size() < N_NEIGHBORS && attempts < DIM*2) {
                                int rand_d2 = rand() % DIM;
                                if (rand_d2 != d1) { // نباید خودش باشد
                                    bool exists = false;
                                    for(int p : partners) if(p == rand_d2) exists = true;
                                    if(!exists) partners.push_back(rand_d2);
                                }
                                attempts++;
                            }
                        #else // ALL_PAIRS
                            for(int d2 = d1 + 1; d2 < DIM; d2++) partners.push_back(d2);
                        #endif

                        for (int d2 : partners) {
                            double nx, ny;
                            alu->compute(position[d1], position[d2], 
                                         iter_best_pos_ptr[d1], iter_best_pos_ptr[d2], 
                                         Q, nx, ny);
                            
                            accumulated_moves[d1] += nx;
                            move_counts[d1]++;
                            
                            #if STRATEGY_ID == 0
                                accumulated_moves[d2] += ny;
                                move_counts[d2]++;
                            #endif
                        }
                    }

                    wait(20, SC_NS); // Time Simulation

                    // اعمال میانگین حرکت‌ها (قبل از نویز)
                    for (int d = 0; d < DIM; d++) {
                        if (move_counts[d] > 0) {
                            position[d] = accumulated_moves[d] / move_counts[d];
                        }
                    }
                } // پایان شرط if(is_best)

                // =========================================================
                // اعمال نویز (مشترک برای همه - حتی بهترین)
                // =========================================================
                for (int d = 0; d < DIM; d++) {
                    double noise = (*M_ptr) * ((rand() / (double)RAND_MAX) * 2 - 1);
                    double new_pos = position[d] + noise;

                    // Bounds Check
                    if (new_pos > UB) new_pos = UB;
                    if (new_pos < LB) new_pos = LB;
                    
                    position[d] = new_pos;
                }
                
                done.write(true);
            }
        }
    }

    SC_CTOR(Penguin_Core) {
        alu = new Spiral_ALU("Internal_ALU");
        alu->clk(clk);
        SC_THREAD(process_logic);
        sensitive << clk.pos();
        done.initialize(true);
    }
};

#endif // PENGUIN_CORE_H