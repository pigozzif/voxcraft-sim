#include "VX_MLP.h"
#include <string>
#include <math.h>
#include <stdlib.h>

CVX_MLP::CVX_MLP(const int numInputs, const int numOutputs, const std::string weights)
{
  this->numInputs = numInputs;
  this->numOutputs = numOutputs;
  SetWeights(weights);
}

CVX_MLP::~CVX_MLP(void)
{
  for (int i = 0; i < numOutputs; ++i) {
    free(weights[i]);
  }
  free(weights);
}

void CVX_MLP::SetWeights(const std::string weights)
{
  this->weights = (double**) malloc(sizeof(double*) * numOutputs);
  for (int i = 0; i < numOutputs; ++i) {
    this->weights[i] = (double*) malloc(sizeof(double) * (numInputs + 1));
  }
  std::string delim = ",";
  std::size_t start = 0U;
  std::size_t end = weights.find(delim);
  int i = 0;
  int j = 0;
  while (end != std::string::npos) {
    this->weights[i][j++] = atof(weights.substr(start, end - start).c_str());
    if (j >= numInputs + 1) {
      j = 0;
      ++i;
    }
    start = end + delim.length();
    end = weights.find(delim, start);
  }
}

double* CVX_MLP::Apply(double* inputs) const
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
