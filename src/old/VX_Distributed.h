#ifndef CVX_DISTRIBUTED_H
#define CVX_DISTRIBUTED_H

#include "VX_MLP.h"
#include "VX_Object.h"
#include "VX_Voxel.h"
#include "Voxelyze.h"
#include <string>

#define NUM_SENSORS 6
#define NUM_SIGNALS 6

class CVX_Distributed
{
public:
  CVX_Distributed(const int numVoxels, const std::string weights, CVoxelyze* sim);
  ~CVX_Distributed(void);

  double UpdateVoxelTemp(CVX_Object* pObj, CVX_Voxel* voxel);
  void UpdateLastSignals(void);
  double* GetLastSignals(CVX_Voxel* voxel, CVX_Object* pObj) const;
  void sense(CVX_Voxel* voxel, double* sensors) const;
  
  static Vec3D<float>* getOffset(CVX_Voxel::linkDirection dir) const;

private:
  int numVoxels;
  CVX_MLP* mlp;
  double** lastSignals;
  double** currSignals;
  CVoxelyze* sim;
};

#endif //CVX_DISTRIBUTED_H
