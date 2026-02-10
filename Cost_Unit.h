#ifndef COST_UNIT_H
#define COST_UNIT_H

#include <systemc.h>
#include <cmath>
#include <iostream>
#include "config.h"

SC_MODULE(Cost_Unit) {
    sc_in<bool> clk;
    sc_in<bool> start;
    sc_out<bool> done;

    double* position_ptr;
    double* cost_output_ptr;

    void compute_logic() {
        while(true) {
            wait();
            if (start.read()) {
                done.write(false);
                
                if (!position_ptr || !cost_output_ptr) {
                    done.write(true);
                    continue; 
                }

                double sum = 0;

                #if FUNC_ID == 0
                    for (int i = 0; i < DIM - 1; i++) {
                        double x_next = position_ptr[i+1];
                        double x_curr = position_ptr[i];
                        double t1 = 100 * std::pow(x_next - std::pow(x_curr, 2), 2);
                        double t2 = std::pow(x_curr - 1, 2);
                        sum += t1 + t2;
                    }
                #elif FUNC_ID == 1
                    for (int i = 0; i < DIM; i++) {
                        sum += position_ptr[i] * position_ptr[i];
                    }
                #endif
                
                wait(50, SC_NS);

                *cost_output_ptr = sum;
                done.write(true);
            }
        }
    }

    SC_CTOR(Cost_Unit) {
        SC_THREAD(compute_logic);
        sensitive << clk.pos();
        set_stack_size(0x10000); 
        done.initialize(true);
        
        position_ptr = nullptr;
        cost_output_ptr = nullptr;
    }
};

#endif