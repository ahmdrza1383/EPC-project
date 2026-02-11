#ifndef SPIRAL_ALU_H
#define SPIRAL_ALU_H

#include <systemc.h>
#include <cmath>
#include "config.h"

SC_MODULE(Spiral_ALU) {
    sc_in<bool> clk;
    
    const double a = 1.0;
    const double b = 0.5;

    void compute(double curr_x, double curr_y, double best_x, double best_y, double Q, double dynamic_a,
                 double &out_nx, double &out_ny) {
        
        double theta_i = std::atan2(curr_y, curr_x);
        double theta_j = std::atan2(best_y, best_x);

        double term1 = (1.0 - Q) * std::exp(b * theta_j);
        double term2 = Q * std::exp(b * theta_i);
        double S = term1 + term2;
        if (S <= 0) S = 1e-6;

        double theta_k = (1.0 / b) * std::log(S);
        double r_k = dynamic_a * S;

        out_nx = r_k * std::cos(theta_k);
        out_ny = r_k * std::sin(theta_k);
    }

    SC_CTOR(Spiral_ALU) {}
};

#endif 