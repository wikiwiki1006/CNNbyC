#ifndef CALC_H
#define CALC_H

#include <stddef.h>
#include "layer.h"

void ReLU(const double* input, double* output, int n_in);

double *BCE(const double *y_true, const double *y_pred, int nnodes, int batch);

void FinalDelta(const double* y_true, const double* y_pred, double* delta, int nnodes, int batch);

void SoftMax(const double* input, double* output, int n_in);

void ConvForward(CONV *conv, double *input, KERNEL *kernel);

void PoolForward(POOL *pool, CONV *conv);

void FlattenForward(POOL *pool, FLLAYER *fllayer);

void FL2FCForward(FLLAYER *fllayer, FCLAYER *fclayer);

void FCForward(FCLAYER *prev_node, FCLAYER *curr_node);

void FCBackward(FCLAYER *curr_layer, FCLAYER *prev_layer, double lr);

void FL2PLBackward(FLLAYER *curr_layer, POOL *prev_layer);

void FC2FLBackward(FCLAYER *curr_layer, FLLAYER *prev_layer, double lr);

void PL2CVBackward(POOL* curr_layer, CONV* prev_layer);

void ConvBackward(CONV* conv, KERNEL* kernel);

void UpdateKernelWeightsBiases(KERNEL* kernel_layer, double lr, unsigned int batch_size);

void UpdateFCWeightsBiases(FCLAYER* curr_layer, FCLAYER* prev_layer, double lr, unsigned int batch_size);

void InitDWeightBias(double* dweights, double* biases, int nweights, int nbiases);

void InitDelta(double* delta, int nnodes);



#endif
