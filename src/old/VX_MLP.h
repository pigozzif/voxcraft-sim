#ifndef CVX_MLP_H
#define CVX_MLP_H

#include <string>

class CVX_MLP
{
public:
  CVX_MLP(const int numInputs, const int numOutputs, const std::string weights);
  ~CVX_MLP(void);

  double* Apply(double* inputs) const;
  inline int getNumInputs() const { return numInputs; }
  inline int getNumOutputs() const { return numOutputs; }

  double** GetWeights() const { return weights; };
  void SetWeights(const std::string weights);

private:
  int numInputs;
  int numOutputs;
  double** weights;
};

#endif //CVX_MLP_H
