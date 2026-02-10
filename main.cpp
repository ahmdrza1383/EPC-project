#include <iostream>
#include <vector>
#include <iomanip>
#include "config.h"
#include "Penguin_Core.h"

// تابع کمکی (Max/Min)
bool is_better(double new_val, double best_val) {
    #if OPT_MODE == 0 // Minimize
        return new_val < best_val;
    #else // Maximize
        return new_val > best_val;
    #endif
}

double calculate_cost(double p[DIM]) {
    #if FUNC_ID == 0 // Rosenbrock
        double sum = 0;
        for (int i = 0; i < DIM - 1; i++) {
            double t1 = 100 * std::pow(p[i+1] - std::pow(p[i], 2), 2);
            double t2 = std::pow(p[i] - 1, 2);
            sum += t1 + t2;
        }
        return sum;
    #elif FUNC_ID == 1 // Sphere (Inverted for Max test or Normal for Min)
        double sum = 0;
        for (int i = 0; i < DIM; i++) sum += p[i] * p[i];
        // اگر مود Max باشد، باید منفی کنیم تا معنی پیدا کند (یا تابع دیگری باشد)
        // اما طبق کد شما sphere معمولی است.
        return sum; 
    #endif
}

int sc_main(int argc, char* argv[]) {
    srand(30); // Seed طبق کد پایتون شما

    sc_clock clk("clk", 10, SC_NS);
    sc_signal<bool> start_all;
    std::vector<sc_signal<bool>*> core_dones;

    double iter_best_pos[DIM];
    
    // متغیر مشترک جدید: ایندکس بهترین پنگوئن
    int iter_best_idx_shared = -1; 

    double M = M_INIT;
    double mu = MU_INIT;
    
    double global_best_score;
    #if OPT_MODE == 0
        global_best_score = 1e9;
    #else
        global_best_score = -1e9;
    #endif

    std::vector<Penguin_Core*> cores;
    
    std::cout << "=== SYSTEMC CONFIGURATION ===" << std::endl;
    std::cout << "Mode: " << (OPT_MODE == 0 ? "MINIMIZE" : "MAXIMIZE") << std::endl;
    std::cout << "Strategy: " << (STRATEGY_ID == 1 ? "RANDOM" : "ALL_PAIRS") << std::endl;

    for(int i=0; i<POP_SIZE; i++) {
        char name[20];
        sprintf(name, "Core_%d", i);
        Penguin_Core* core = new Penguin_Core(name);
        sc_signal<bool>* done = new sc_signal<bool>();
        
        core->clk(clk);
        core->start(start_all);
        core->done(*done);
        
        // --- اتصالات جدید برای تطابق دقیق با پایتون ---
        core->core_id = i; // شناسه منحصر به فرد
        core->iter_best_idx_ptr = &iter_best_idx_shared; // پوینتر به متغیر مشترک
        
        core->iter_best_pos_ptr = iter_best_pos;
        core->M_ptr = &M;
        core->mu_ptr = &mu;
        core->iter_best_idx_ptr = &iter_best_idx_shared;
        for(int d=0; d<DIM; d++) {
            core->position[d] = LB + ((rand()/(double)RAND_MAX) * (UB - LB));
        }

        core->fix_pointers(); 

        cores.push_back(core);
        core_dones.push_back(done);
    }


    sc_trace_file *tf = sc_create_vcd_trace_file("epc_waves");
    sc_trace(tf, clk, "clk");
    sc_trace(tf, start_all, "Start_All");
    sc_trace(tf, global_best_score, "Global_Best");

    for (int t = 0; t < MAX_ITER; t++) {
        
        // 1. Software: Find Best
        double best_val_iter;
        #if OPT_MODE == 0
             best_val_iter = 1e9;
        #else
             best_val_iter = -1e9;
        #endif
        
        int best_idx = -1;

        for(int i=0; i<POP_SIZE; i++) {
            double cost = calculate_cost(cores[i]->position);
            
            if (is_better(cost, best_val_iter)) {
                best_val_iter = cost;
                best_idx = i;
            }
            if (is_better(cost, global_best_score)) {
                global_best_score = cost;
            }
        }

        // آپدیت حافظه مشترک
        for(int d=0; d<DIM; d++) {
            iter_best_pos[d] = cores[best_idx]->position[d];
        }
        
        // *** نکته حیاتی: آپدیت کردن ایندکس بهترین برای سخت‌افزار ***
        iter_best_idx_shared = best_idx;

        std::cout << "Iter " << std::setw(3) << t+1 
                  << " | Global Best: " << global_best_score 
                  << " | Best ID: " << best_idx << std::endl;

        // 2. Hardware Run
        start_all.write(true);
        sc_start(10, SC_NS);
        start_all.write(false);

        bool all_finished = false;
        while(!all_finished) {
            sc_start(10, SC_NS);
            all_finished = true;
            for(auto d : core_dones) {
                if (d->read() == false) {
                    all_finished = false;
                    break;
                }
            }
        }
        
        M *= COOLING_RATE;
        mu *= COOLING_RATE;
    }

    sc_close_vcd_trace_file(tf);
    std::cout << "=== FINISHED ===" << std::endl;

    for(auto c : cores) delete c;
    for(auto d : core_dones) delete d;

    return 0;
}