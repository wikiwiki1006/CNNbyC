#ifndef CALC_H
#define CALC_H

#include <stddef.h>

double *ReLU(const double *input, int n_in);

double BCE(const int *y_true, const double *y_pred, int n);

double *SoftMax(const double *input, int n_in);

void ConvForward(CONV *conv, double *input, KERNEL *kernel);

#endif
