#if !defined(VX3_DISTRIBUTED_H)
#define VX3_DISTRIBUTED_H

#include "VX3.cuh"
#include "VX3_Voxel.h"

#define NUM_SENSORS 5
#define NUM_SIGNALS 12

class VX3_VoxelyzeKernel;

struct Vote {
  double v;
  int x;
  int y;
  int z;
  int is_touching;
};

class VX3_MLP
{
public:
  __device__ VX3_MLP(const int numInputs, const int numOutputs, double* weights);
  __device__ ~VX3_MLP(void);
  
  __device__ void apply(VX3_Voxel* voxel) const;
  
  int numInputs;
  int numOutputs;
  double* weights;
};

class VX3_DistributedNeuralController
{
public:
  __device__ VX3_DistributedNeuralController(VX3_VoxelyzeKernel* kernel, double* weights, int random_seed=0);  
  __device__ double updateVoxelTemp(VX3_Voxel* voxel, VX3_VoxelyzeKernel* kernel);
  __device__ void updateLastSignals(VX3_VoxelyzeKernel* kernel);
  __device__ void getLastSignals(VX3_Voxel* voxel) const;
  __device__ void sense(VX3_Voxel* voxel, VX3_VoxelyzeKernel* kernel);
  __device__ bool has(int* values, int value, int n);
  __device__ void vote(void);
  __device__ void printVotes(VX3_VoxelyzeKernel* kernel);
  
  __device__ VX3_Vec3D<float>* getOffset(const linkDirection dir) const;

  VX3_MLP* mlp;
  VX3_dVector<int>* votes;
  VX3_dVector<Vote>* tempVotes;
  bool firstRightContact;
  bool firstLeftContact;
};

#endif //VX3_DISTRIBUTED_H
