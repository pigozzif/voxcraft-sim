#ifndef CVX_DISTRIBUTED_H
#define CVX_DISTRIBUTED_H

#include "VX_MLP.h"
#include "VX_Object.h"
#include "VX_Voxel.h"
#include <string>

class CVX_Distributed
{
public:
  CVX_Distributed(const int numInputs, const int numVoxels, const std::string weights);
  ~CVX_Distributed(void);

  void UpdateMatTemp(CVX_Object* pObjUpdate, CVX_Voxel* voxel);
  void UpdateLastSignals(void);
  double* GetLastSignals(int i, CVX_Object* pObjUpdate) const;

private:
  int numVoxels;
  CVX_MLP* mlp;
  double** lastSignals;
  double** currSignals;
};

#endif //CVX_DISTRIBUTED_H
