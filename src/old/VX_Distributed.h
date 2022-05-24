#ifndef CVX_DISTRIBUTED_H
#define CVX_DISTRIBUTED_H

#include "VX_MLP.h"

class CVX_Distributed
{
public:
  CVX_Distributed(const int numInputs, const int numOutputs, const int numMaterials);
  ~CVX_Distributed(void);

  void UpdateMatTemp(CVX_Object* pObjUpdate);

private:
  int numMaterials;
  CVX_MLP* mlp;
  double** lastSignals;
  double** currSignals;
};

#endif //CVX_DISTRIBUTED_H
