#ifndef CVX_DISTRIBUTED_H
#define CVX_DISTRIBUTED_H

#include "VX_MLP.h"
#include "VX_Object.h"
#include <string>

class CVX_Distributed
{
public:
  CVX_Distributed(const int numInputs, const int numMaterials, const std::string weights);
  ~CVX_Distributed(void);

  void UpdateMatTemp(CVX_Object* pObjUpdate) const;
  double* GetLastSignals(int i, CVX_Object* pObjUpdate) const;

private:
  int numMaterials;
  CVX_MLP* mlp;
  double** lastSignals;
  double** currSignals;
};

#endif //CVX_DISTRIBUTED_H
