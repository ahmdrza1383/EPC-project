#ifndef PENGUIN_CORE_H
#define PENGUIN_CORE_H

#include <systemc.h>
#include <vector>
#include "config.h"
#include "Dim_Unit.h"

SC_MODULE(Penguin_Core) {
    sc_in<bool> clk;
    sc_in<bool> start;
    sc_out<bool> done;

    // هویت
    int core_id;
    int* iter_best_idx_ptr;

    // حافظه داخلی
    double position[DIM];
    double next_position[DIM]; 

    // پوینترهای ورودی (از main می‌آیند)
    double* iter_best_pos_ptr;
    double* M_ptr;
    double* mu_ptr;
    
    // متغیر داخلی Q
    double calculated_Q; 

    // فرزندان
    std::vector<Dim_Unit*> dim_units;
    std::vector<sc_signal<bool>*> dim_dones;
    sc_signal<bool> start_dims;

    void process_logic() {
        while (true) {
            wait();
            if (start.read()) {
                done.write(false);

                // اگر پوینترها ست نشده باشند، خطا ندهیم ولی کاری هم نکنیم
                if (!iter_best_idx_ptr || !M_ptr || !mu_ptr || !iter_best_pos_ptr) {
                     done.write(true);
                     continue;
                }

                if (core_id == *iter_best_idx_ptr) {
                    // رفتار بهترین پنگوئن: فقط نویز
                    for(int d=0; d<DIM; d++) {
                         double noise = (*M_ptr) * ((rand() / (double)RAND_MAX) * 2 - 1);
                         double val = position[d] + noise;
                         if (val > UB) val = UB;
                         if (val < LB) val = LB;
                         position[d] = val;
                    }
                    wait(5, SC_NS);
                } 
                else {
                    // رفتار عادی: محاسبه Q و استارت زدن فرزندان
                    double dist = 0;
                    for(int k=0; k<DIM; k++) {
                        double diff = position[k] - iter_best_pos_ptr[k];
                        dist += diff * diff;
                    }
                    dist = std::sqrt(dist);
                    calculated_Q = std::exp(-(*mu_ptr) * dist);
                    if (calculated_Q > 1) calculated_Q=1; 
                    if (calculated_Q < 1e-4) calculated_Q=1e-4;

                    // شروع پردازش موازی
                    start_dims.write(true);
                    wait(); 
                    start_dims.write(false);

                    // انتظار برای پایان فرزندان
                    bool all_dims_finished = false;
                    while (!all_dims_finished) {
                        wait(); 
                        all_dims_finished = true;
                        for(auto d : dim_dones) {
                            if (d->read() == false) {
                                all_dims_finished = false;
                                break;
                            }
                        }
                    }

                    // کپی نتایج
                    for(int d=0; d<DIM; d++) {
                        position[d] = next_position[d];
                    }
                }
                
                done.write(true);
            }
        }
    }

    // تابع مهم برای اتصال پوینترهای فرزندان
    void fix_pointers() {
        for(auto unit : dim_units) {
            // این مقادیر الان معتبر هستند (چون در main ست شده‌اند)
            unit->iter_best_pos_ptr = this->iter_best_pos_ptr;
            unit->M_ptr = this->M_ptr;
        }
    }

    SC_CTOR(Penguin_Core) {
        SC_THREAD(process_logic);
        sensitive << clk.pos();
        done.initialize(true);
        
        // مقداردهی اولیه پوینترها به نال
        iter_best_idx_ptr = nullptr;
        iter_best_pos_ptr = nullptr;
        M_ptr = nullptr;
        mu_ptr = nullptr;

        for(int d=0; d<DIM; d++) {
            char name[20];
            sprintf(name, "DimUnit_%d", d);
            
            Dim_Unit* unit = new Dim_Unit(name);
            sc_signal<bool>* d_done = new sc_signal<bool>();

            unit->clk(clk);
            unit->start(start_dims);
            unit->done(*d_done);

            unit->my_d1 = d;
            unit->current_pos_ptr = position;
            unit->next_pos_ptr = next_position;
            unit->Q_val_ptr = &calculated_Q;
            
            // نکته: M_ptr و iter_best_pos_ptr اینجا ست نمی‌شوند
            // چون هنوز مقدار ندارند. تابع fix_pointers بعداً این کار را می‌کند.

            dim_units.push_back(unit);
            dim_dones.push_back(d_done);
        }
    }
};

#endif // PENGUIN_CORE_H