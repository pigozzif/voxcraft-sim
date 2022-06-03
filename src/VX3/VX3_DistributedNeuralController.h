#if !defined(CVX_DISTRIBUTED_H)
#define VX3_DISTRIBUTED_H

#include "VX3_Voxel.h"
#include "VX3.cuh"
#include "VX3_VoxelyzeKernel.cuh"
#include <string>
#include <map>

#define NUM_SENSORS 6
#define NUM_SIGNALS 6

class VX3_MLP
{
public:
  VX3_MLP(const int numInputs, const int numOutputs, const std::string weights);
  ~VX3_MLP(void);

 // __device__ double* apply(double* inputs) const;
 // __device__ inline int getNumInputs(void) const { return numInputs; }
 // __device__ inline int getNumOutputs(void) const { return numOutputs; }

//  __device__ double** getWeights(void) const { return weights; };
  __device__ void setWeights(const std::string weights);

private:
  int numInputs;
  int numOutputs;
  double** weights;
};

/*class VX3_NeuralDistributedController
{
public:
  __device__ VX3_NeuralDistributedController(const std::string weights, VX3_VoxelyzeKernel* kernel);
  __device__ ~VX3_NeuralDistributedController(void);

  __device__ double updateVoxelTemp(VX3_Voxel* voxel, VX3_VoxelyzeKernel* kernel);
  __device__ void updateLastSignals(VX3_VoxelyzeKernel* kernel);
  __device__ double* getLastSignals(CVX_Voxel* voxel) const;
  __device__ void sense(CVX_Voxel* voxel, double* sensors, VX3_VoxelyzeKernel* kernel) const;
  
  __device__ VX3_Vec3D<float>* getOffset(const linkDirection dir) const;

private:
  int numVoxels;
  VX3_MLP* mlp;
  std::map<VX3_Voxel*, double[]> lastSignals;
  std::map<VX3_Voxel*, double[]> currSignals;
};*/

#endif //VX3 _DISTRIBUTED_H
