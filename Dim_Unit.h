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

    int my_d1;

    double* current_pos_ptr;
    double* iter_best_pos_ptr;
    double* Q_val_ptr;
    double* M_ptr;
    double* a_ptr;
    double* next_pos_ptr;    

    Spiral_ALU* alu;

    void process_logic() {
        while (true) {
            wait();
            if (start.read()) {
                done.write(false);

                if (!current_pos_ptr || !iter_best_pos_ptr || !Q_val_ptr || !M_ptr || !next_pos_ptr) {
                    done.write(true);
                    continue;
                }

                double accumulated_move = 0;
                int move_count = 0;

                std::vector<int> partners;

                #if STRATEGY_ID == 1 
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
                #else 
                    for(int d2 = 0; d2 < DIM; d2++) {
                        if (d2 != my_d1) partners.push_back(d2);
                    }
                #endif

                for (int d2 : partners) {
                    double nx, ny;
                    alu->compute(current_pos_ptr[my_d1], current_pos_ptr[d2], 
                                 iter_best_pos_ptr[my_d1], iter_best_pos_ptr[d2], 
                                 *Q_val_ptr, *a_ptr,
                                 nx, ny);
                    
                    accumulated_move += nx;
                    move_count++;
                }

                wait(5, SC_NS);

                double new_val = current_pos_ptr[my_d1];
                if (move_count > 0) {
                    new_val = accumulated_move / move_count;
                }

                double noise = (*M_ptr) * ((rand() / (double)RAND_MAX) * 2 - 1);
                new_val += noise;

                if (new_val > UB) new_val = UB;
                if (new_val < LB) new_val = LB;

                next_pos_ptr[my_d1] = new_val;

                done.write(true);
            }
        }
    }

    SC_CTOR(Dim_Unit) {
        alu = new Spiral_ALU("ALU");
        alu->clk(clk); 
        
        current_pos_ptr = nullptr;
        iter_best_pos_ptr = nullptr;
        Q_val_ptr = nullptr;
        M_ptr = nullptr;
        a_ptr = nullptr;
        next_pos_ptr = nullptr;

        SC_THREAD(process_logic);
        sensitive << clk.pos();
        set_stack_size(0x10000);
        done.initialize(true);
    }
};

#endif