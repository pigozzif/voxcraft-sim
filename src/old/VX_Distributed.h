#ifndef CVX_DISTRIBUTED_H
#define CVX_DISTRIBUTED_H

#include "VX_MLP.h"
#include "VX_Object.h"
#include "VX_Voxel.h"
#include "Voxelyze.h"
#include <string>

#define NUM_SENSORS 6
#define NUM_SIGNALS 6

class CVX_MLP
{
public:
  CVX_MLP(const int numInputs, const int numOutputs, const std::string weights);
  ~CVX_MLP(void);

  double* apply(double* inputs) const;
  inline int getNumInputs() const { return numInputs; }
  inline int getNumOutputs() const { return numOutputs; }

  double** getWeights() const { return weights; };
  void setWeights(const std::string weights);

private:
  int numInputs;
  int numOutputs;
  double** weights;
};

class CVX_Distributed
{
public:
  CVX_Distributed(const int numVoxels, const std::string weights, CVoxelyze* sim);
  ~CVX_Distributed(void);

  double updateVoxelTemp(CVX_Object* pObj, CVX_Voxel* voxel);
  void updateLastSignals(void);
  double* getLastSignals(CVX_Voxel* voxel, CVX_Object* pObj) const;
  void sense(CVX_Voxel* voxel, double* sensors) const;
  
  Vec3D<float>* getOffset(CVX_Voxel::linkDirection dir) const;

private:
  int numVoxels;
  CVX_MLP* mlp;
  double** lastSignals;
  double** currSignals;
  CVoxelyze* sim;
};

#endif //CVX_DISTRIBUTED_H
