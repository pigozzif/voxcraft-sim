#ifndef CVX_MLP_H
#define CVX_MLP_H

#include <vector>


class CVX_MLP
{
public:
  CVX_MLP(const int numInputs, const int numOutputs);
  ~CVX_MLP(void);

  double* Apply(double* inputs);
  inline int getNumInputs() { return numInputs; }
  inline int getNumOutputs() { return numOutputs; }

private:
  int numInputs;
  int numOutputs;
  std::vector<std::vector<double> > weights;
};

#endif //CVX_MLP_H
