#ifndef CONFIG_H
#define CONFIG_H

#include <cmath>

#define DIM 100
#define POP_SIZE 20
#define MAX_ITER 100

const double LB = -10.0;
const double UB =  10.0;

#define STRATEGY_ID 1 

const int N_NEIGHBORS = (int)std::sqrt(DIM); 

#define OPT_MODE 0 

#define FUNC_ID 1

const double M_INIT = 0.5;
const double MU_INIT = 0.05;
const double COOLING_RATE = 0.99;

#endif