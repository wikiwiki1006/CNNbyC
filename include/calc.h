#ifndef CALC_H
#define CALC_H

#include <stddef.h>

double *ReLU(const double *input, int n_in);

double *BCE(const double *y_true, const double *y_pred, int nnodes, int batch);

double *SoftMax(const double *input, int n_in);

void ConvForward(CONV *conv, double *input, KERNEL *kernel);

void PoolForward(POOL *pool, CONV *conv);

void FlattenForward(POOL *pool, FLLAYER *fllayer);

void FL2FCForward(FLLAYER *fllayer, FCLAYER *fclayer);

void FCForward(FCLAYER *prev_node, FCLAYER *curr_node);

#endif
