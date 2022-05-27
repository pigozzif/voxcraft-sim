#ifndef CVX_DISTRIBUTED_H
#define CVX_DISTRIBUTED_H

#include "VX_MLP.h"
#include "VX_Object.h"
#include "VX_Voxel.h"
#include "VX_Sim.h"
#include <string>

class CVX_TouchSensor
{
public:
  CVX_TouchSensor(void);
  ~CVX_TouchSensor(void);
  
  double sense(CVX_Voxel* source, CVX_Voxel* target, CVX_Voxel::linkDirection dir) const;
  Vec3D<double>* getOffset(CVX_Voxel::linkDirection dir) const;
};

class CVX_Distributed
{
public:
  CVX_Distributed(const int numInputs, const int numVoxels, const std::string weights);
  ~CVX_Distributed(void);

  double UpdateVoxelTemp(CVX_Sim* sim, CVX_Object* pObj, CVX_Voxel* voxel);
  void UpdateLastSignals(void);
  double* GetLastSignals(CVX_Voxel* voxel, CVX_Object* pObj) const;

private:
  int numVoxels;
  CVX_MLP* mlp;
  double** lastSignals;
  double** currSignals;
  CVX_TouchSensor* touchSensor;
};

#endif //CVX_DISTRIBUTED_H
