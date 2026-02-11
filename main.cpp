#include <iostream>
#include <vector>
#include <iomanip>
#include <ctime>
#include "config.h"
#include "Penguin_Core.h"
#include "Cost_Unit.h" 

bool is_better(double new_val, double best_val) {
    #if OPT_MODE == 0 
        return new_val < best_val;
    #else 
        return new_val > best_val;
    #endif
}

int sc_main(int argc, char* argv[]) {
    // srand(50); 
    srand(time(0));

    sc_clock clk("clk", 10, SC_NS);
    
    sc_signal<bool> start_move_all; 
    sc_signal<bool> start_calc_all; 
    
    std::vector<Penguin_Core*> penguins;
    std::vector<sc_signal<bool>*> penguin_dones;

    std::vector<Cost_Unit*> cost_units;
    std::vector<sc_signal<bool>*> cost_dones;

    double current_costs[POP_SIZE];

    double iter_best_pos[DIM];
    int iter_best_idx_shared = -1; 
    double M = M_INIT;
    double mu = MU_INIT;
    double current_a = A_INIT;
    
    double global_best_score;
    #if OPT_MODE == 0
        global_best_score = 1e9;
    #else
        global_best_score = -1e9;
    #endif
    
    std::cout << "=== SYSTEMC CONFIGURATION ===" << std::endl;
    std::cout << "Mode: " << (OPT_MODE == 0 ? "MINIMIZE" : "MAXIMIZE") << std::endl;
    std::cout << "Strategy: " << (STRATEGY_ID == 1 ? "RANDOM" : "ALL_PAIRS") << std::endl;

    for(int i=0; i<POP_SIZE; i++) {
        char p_name[20];
        sprintf(p_name, "Penguin_%d", i);
        Penguin_Core* p = new Penguin_Core(p_name);
        sc_signal<bool>* p_done = new sc_signal<bool>();

        p->clk(clk);
        p->start(start_move_all); 
        p->done(*p_done);

        p->core_id = i;
        p->iter_best_idx_ptr = &iter_best_idx_shared;
        p->iter_best_pos_ptr = iter_best_pos;
        p->M_ptr = &M;
        p->mu_ptr = &mu;
        p->a_ptr = &current_a;
        
        for(int d=0; d<DIM; d++) {
            p->position[d] = LB + ((rand()/(double)RAND_MAX) * (UB - LB));
        }
        p->fix_pointers();
        
        penguins.push_back(p);
        penguin_dones.push_back(p_done);

        char c_name[20];
        sprintf(c_name, "CostUnit_%d", i);
        Cost_Unit* c = new Cost_Unit(c_name);
        sc_signal<bool>* c_done = new sc_signal<bool>();

        c->clk(clk);
        c->start(start_calc_all); 
        c->done(*c_done);

        c->position_ptr = p->position;      
        c->cost_output_ptr = &current_costs[i]; 

        cost_units.push_back(c);
        cost_dones.push_back(c_done);
    }

    for (int t = 0; t < MAX_ITER; t++) {

        start_calc_all.write(true);
        sc_start(10, SC_NS);
        start_calc_all.write(false);

        bool costs_finished = false;
        while(!costs_finished) {
            sc_start(10, SC_NS);
            costs_finished = true;
            for(auto d : cost_dones) {
                if (d->read() == false) {
                    costs_finished = false;
                    break;
                }
            }
        }

        double best_val_iter;
        #if OPT_MODE == 0
            best_val_iter = 1e9;
        #else
            best_val_iter = -1e9;
        #endif
        int best_idx = -1;

        for(int i=0; i<POP_SIZE; i++) {
            double cost = current_costs[i];
            
            if (is_better(cost, best_val_iter)) {
                best_val_iter = cost;
                best_idx = i;
            }
            if (is_better(cost, global_best_score)) {
                global_best_score = cost;
            }
        }

        for(int d=0; d<DIM; d++) {
            iter_best_pos[d] = penguins[best_idx]->position[d];
        }
        iter_best_idx_shared = best_idx;

        std::cout << "Iter " << std::setw(3) << t+1 
                  << " | Global Best: " << global_best_score 
                  << " | Best ID: " << best_idx << std::endl;

        start_move_all.write(true);
        sc_start(10, SC_NS);
        start_move_all.write(false);

        bool moves_finished = false;
        while(!moves_finished) {
            sc_start(10, SC_NS);
            moves_finished = true;
            for(auto d : penguin_dones) {
                if (d->read() == false) {
                    moves_finished = false;
                    break;
                }
            }
        }

        M *= COOLING_RATE;
        mu *= COOLING_RATE;
        current_a *= COOLING_RATE;
    }

    std::cout << "=== FINISHED ===" << std::endl;

    for(auto p : penguins) delete p;
    for(auto pd : penguin_dones) delete pd;
    for(auto c : cost_units) delete c;
    for(auto cd : cost_dones) delete cd;

    return 0;
}