#include "VX_MLP.h"
#include <vector>
#include <math.h>
#include <stdlib.h>

CVX_MLP::CVX_MLP(const int numInputs, const int numOutputs)
{
  std::vector<double> layer;
  layer.push_back(2.0);
  layer.push_back(1.0);
  layer.push_back(1.0);
  layer.push_back(3.0);
  this->weights.push_back(layer);
  this->numInputs = numInputs;
  this->numOutputs = numOutputs;
}

CVX_MLP::~CVX_MLP(void)
{
}

double* CVX_MLP::Apply(double* inputs)
{
  //apply input activation
  for (int i = 0; i < numInputs; ++i)
  {
    inputs[i] = tanh(inputs[i]);
  }
  double* outputs = (double*) malloc(sizeof(double) * numOutputs);
  for (int j = 0; j < numOutputs; ++j)
  {
    double sum = weights[j][0]; //the bias
    for (int k = 1; k < numInputs + 1; ++k)
    {
      sum += inputs[k - 1] * weights[j][k]; //weight inputs
    }
    outputs[j] = tanh(sum); //apply output activation
  }
  return outputs;
}
