#if !defined(VX3_DISTRIBUTED_H)
#define VX3_DISTRIBUTED_H

#include "VX3.cuh"
#include "VX3_Voxel.h"
#include "VX3_Collision.h"

#define NUM_SENSORS 6
#define NUM_SIGNALS 6

class VX3_VoxelyzeKernel;

class VX3_MLP
{
public:
  __device__ VX3_MLP(const int numInputs, const int numOutputs, double** weights);
  __device__ ~VX3_MLP(void);
  
  __device__ void apply(void) const;
  __device__ inline int getNumInputs(void) const { return numInputs; }
  __device__ inline int getNumOutputs(void) const { return numOutputs; }

  __device__ double** getWeights(void) const { return weights; };
  __device__ void setWeights(double** weights);
  
  double* inputs;
  double* outputs;

private:
  int numInputs;
  int numOutputs;
  double** weights;
};

class VX3_DistributedNeuralController
{
public:
  __device__ VX3_DistributedNeuralController(double** weights, VX3_VoxelyzeKernel* kernel);
  __device__ ~VX3_DistributedNeuralController(void);
  
  __device__ double updateVoxelTemp(VX3_Voxel* voxel, VX3_VoxelyzeKernel* kernel);
  __device__ void updateLastSignals(VX3_VoxelyzeKernel* kernel);
  __device__ void getLastSignals(VX3_Voxel* voxel) const;
  __device__ void sense(VX3_Voxel* voxel, VX3_VoxelyzeKernel* kernel) const;
  
  __device__ VX3_Vec3D<float>* getOffset(const linkDirection dir) const;

  VX3_MLP* mlp;
  VX3_dVector<VX3_Collision*> collisions;
};

#endif //VX3_DISTRIBUTED_H
