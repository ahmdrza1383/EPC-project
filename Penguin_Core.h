#ifndef PENGUIN_CORE_H
#define PENGUIN_CORE_H

#include <systemc.h>
#include <vector>
#include "config.h"
#include "Dim_Unit.h" // ماژول فرزند جدید

SC_MODULE(Penguin_Core) {
    sc_in<bool> clk;
    sc_in<bool> start;
    sc_out<bool> done;

    // --- هویت ---
    int core_id;
    int* iter_best_idx_ptr;

    // --- حافظه ---
    double position[DIM];      // وضعیت فعلی
    double next_position[DIM]; // بافر وضعیت بعدی (برای جلوگیری از تداخل موازی)

    // --- ورودی‌های مشترک ---
    double* iter_best_pos_ptr;
    double* M_ptr;
    double* mu_ptr;
    
    // متغیر داخلی برای Q که باید به فرزندان بدهیم
    double calculated_Q; 

    // --- فرزندان موازی (Parallel Execution Units) ---
    std::vector<Dim_Unit*> dim_units;
    std::vector<sc_signal<bool>*> dim_dones; // سیگنال پایان هر فرزند
    sc_signal<bool> start_dims; // سیگنال شروع همگانی برای فرزندان

    void process_logic() {
        while (true) {
            wait();
            if (start.read()) {
                done.write(false);

                // 1. بررسی آیا بهترین هستم؟
                if (core_id == *iter_best_idx_ptr) {
                    // رفتار بهترین پنگوئن: فقط نویز (بدون حرکت اسپیرال)
                    // اینجا سریال انجام می‌دهیم چون سبک است
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
                    // 2. محاسبه Q (یکبار انجام می‌شود)
                    double dist = 0;
                    for(int k=0; k<DIM; k++) {
                        double diff = position[k] - iter_best_pos_ptr[k];
                        dist += diff * diff;
                    }
                    dist = std::sqrt(dist);
                    calculated_Q = std::exp(-(*mu_ptr) * dist);
                    if (calculated_Q > 1) calculated_Q=1; 
                    if (calculated_Q < 1e-4) calculated_Q=1e-4;

                    // 3. === فاز موازی‌سازی داخلی ===
                    // شلیک دستور شروع به تمام ۲۰ واحد ابعاد
                    start_dims.write(true);
                    wait(); // یک سیکل صبر برای اعمال سیگنال
                    start_dims.write(false);

                    // انتظار برای پایان همه ۲۰ واحد
                    // (اینجا جایی است که قدرت سخت‌افزار مشخص می‌شود)
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

                    // 4. آپدیت نهایی (Commit)
                    // مقادیر محاسبه شده توسط واحدها را از بافر به متغیر اصلی منتقل می‌کنیم
                    for(int d=0; d<DIM; d++) {
                        position[d] = next_position[d];
                    }
                }
                
                done.write(true);
            }
        }
    }

    SC_CTOR(Penguin_Core) {
        SC_THREAD(process_logic);
        sensitive << clk.pos();
        done.initialize(true);

        // --- ساختن ۲۰ واحد پردازش موازی ---
        for(int d=0; d<DIM; d++) {
            char name[20];
            sprintf(name, "DimUnit_%d", d);
            
            Dim_Unit* unit = new Dim_Unit(name);
            sc_signal<bool>* d_done = new sc_signal<bool>();

            // اتصالات پورت‌ها
            unit->clk(clk);
            unit->start(start_dims); // همه به یک استارت وصل هستند
            unit->done(*d_done);

            // تنظیم پوینترها
            unit->my_d1 = d;
            unit->current_pos_ptr = position;
            unit->next_pos_ptr = next_position; // نوشتن در بافر خروجی
            unit->iter_best_pos_ptr = iter_best_pos_ptr; // این بعدا در main وصل می‌شود، اما اینجا فقط پاس می‌دهیم؟ 
            // **نکته مهم:** پوینترهای iter_best_pos_ptr در کانستراکتور هنوز مقداردهی نشده‌اند.
            // ما باید این‌ها را به صورت عمومی تعریف کنیم یا بعدا ست کنیم.
            // راه حل ساده: پوینترهای کلاس Dim_Unit را مستقیم به متغیرهای کلاس Penguin_Core وصل نکنیم،
            // بلکه به متغیرهای پوینتری Penguin_Core وصل کنیم. (در ادامه توضیح میدهم)
            
            unit->Q_val_ptr = &calculated_Q;
            unit->M_ptr = M_ptr; // این هم نال است فعلا!

            dim_units.push_back(unit);
            dim_dones.push_back(d_done);
        }
    }
    
    // تابع کمکی برای اصلاح پوینترهای فرزندان بعد از ساختن پنگوئن در Main
    void fix_pointers() {
        for(auto unit : dim_units) {
            unit->iter_best_pos_ptr = iter_best_pos_ptr;
            unit->M_ptr = M_ptr;
        }
    }
};

#endif // PENGUIN_CORE_H