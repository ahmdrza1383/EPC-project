#include <iostream>
#include "epc_modules.h" // فراخوانی ماژول‌های سخت‌افزاری

// تابع محاسبه هزینه (این بخش نرم‌افزاری است و روی CPU اجرا می‌شود)
double calculate_cost(double p[DIM]) {
    // تابع Rosenbrock
    double sum = 0;
    for (int i = 0; i < DIM - 1; i++) {
        double t1 = 100 * std::pow(p[i+1] - std::pow(p[i], 2), 2);
        double t2 = std::pow(p[i] - 1, 2);
        sum += t1 + t2;
    }
    return sum;
}

int sc_main(int argc, char* argv[]) {
    // تنظیم کلاک سیستم (10 نانوثانیه = 100 مگاهرتز)
    sc_clock clk("clk", 10, SC_NS);
    
    // سیگنال‌های کنترلی سراسری
    sc_signal<bool> start_all;
    
    // سیگنال‌های وضعیت برای هر هسته
    std::vector<sc_signal<bool>*> core_dones;

    // متغیرهای مشترک (Shared Memory Simulation)
    double iter_best_pos[DIM];
    double M = 0.5;
    double mu = 0.05;
    double global_best_score = 1e9;

    // --- ساخت و پیکربندی هسته‌های پردازشی ---
    std::vector<Penguin_Core*> cores;
    
    std::cout << "Initializing " << POP_SIZE << " Hardware Cores..." << std::endl;

    for(int i=0; i<POP_SIZE; i++) {
        char name[20];
        sprintf(name, "Core_%d", i);
        
        Penguin_Core* core = new Penguin_Core(name);
        sc_signal<bool>* done = new sc_signal<bool>();
        
        // اتصال پورت‌ها
        core->clk(clk);
        core->start(start_all);
        core->done(*done);
        
        // اتصال پوینترهای حافظه
        core->iter_best_pos_ptr = iter_best_pos;
        core->M_ptr = &M;
        core->mu_ptr = &mu;
        
        // مقداردهی اولیه تصادفی موقعیت‌ها
        for(int d=0; d<DIM; d++) {
            core->position[d] = ((rand()/(double)RAND_MAX)*20.0) - 10.0;
        }

        cores.push_back(core);
        core_dones.push_back(done);
    }

    std::cout << "=== SYSTEMC SIMULATION STARTED (Co-design Mode) ===" << std::endl;
    std::cout << "Strategy: Hardware Parallel Spirals | Function: Rosenbrock" << std::endl;

    // حلقه اصلی زمان (Iteration Loop)
    for (int t = 0; t < MAX_ITER; t++) {
        
        // -------------------------------------------------
        // بخش نرم‌افزار (Software): ارزیابی و انتخاب بهترین
        // -------------------------------------------------
        double best_val = 1e9;
        int best_idx = -1;

        for(int i=0; i<POP_SIZE; i++) {
            double cost = calculate_cost(cores[i]->position);
            if (cost < best_val) {
                best_val = cost;
                best_idx = i;
            }
            if (cost < global_best_score) global_best_score = cost;
        }

        // انتقال بهترین موقعیت به رجیسترهای مشترک (برای خواندن سخت‌افزار)
        for(int d=0; d<DIM; d++) {
            iter_best_pos[d] = cores[best_idx]->position[d];
        }

        std::cout << "Iter " << std::setw(3) << t+1 
                  << " | Global Best: " << global_best_score 
                  << " | M: " << M << std::endl;

        // -------------------------------------------------
        // بخش سخت‌افزار (Hardware): اجرای موازی هسته‌ها
        // -------------------------------------------------
        
        // 1. ارسال پالس شروع به تمام ۳۰ هسته همزمان
        start_all.write(true);
        sc_start(10, SC_NS); // یک کلاک
        start_all.write(false);

        // 2. انتظار برای پایان کار تمام هسته‌ها (Barrier Synchronization)
        bool all_finished = false;
        while(!all_finished) {
            sc_start(10, SC_NS); // جلو بردن زمان
            
            all_finished = true;
            for(auto d : core_dones) {
                if (d->read() == false) {
                    all_finished = false;
                    break;
                }
            }
        }
        
        // 3. کاهش پارامترها (Cooling)
        M *= 0.99;
        mu *= 0.99;
    }

    std::cout << "=== SIMULATION FINISHED ===" << std::endl;
    std::cout << "Final Best Cost: " << global_best_score << std::endl;

    // پاکسازی حافظه (اختیاری ولی خوب)
    for(auto c : cores) delete c;
    for(auto d : core_dones) delete d;

    return 0;
}