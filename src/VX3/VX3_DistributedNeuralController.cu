#include "VX3_DistributedNeuralController.h"
#include "VX3.cuh"
#include "VX3_Voxel.h"
#include "VX3_VoxelyzeKernel.cuh"
#include "VX3_Collision.h"
#include "VX3_MemoryCleaner.h"
#include <algorithm>
#include <stdlib.h>
#include <cstdlib>
#include <math.h>
#include <string>
#include <map>

__device__ VX3_MLP::~VX3_MLP(void)
{
  for (int i = 0; i < numOutputs; ++i) {
    VcudaFree(weights[i]);
  }
  VcudaFree(weights);
}

__device__ VX3_MLP::VX3_MLP(const int numInputs, const int numOutputs, double** weights)
{
  this->numInputs = numInputs;
  this->numOutputs = numOutputs;
  setWeights(weights);
}

__device__ void VX3_MLP::setWeights(double** weights)
{
  VcudaMalloc((void **) &this->weights, sizeof(double*) * numOutputs);
  for (int i = 0; i < numOutputs; ++i) {
    VcudaMalloc((void **) &this->weights[i], sizeof(double) * (numInputs + 1));
  }
  for (int i = 0; i < numOutputs; ++i) {
    for (int j = 0; j < numInputs + 1; ++j) {
      this->weights[i][j] = weights[i][j];
    }
  }
}

__device__ double* VX3_MLP::apply(double* inputs) const
{
  //apply input activation
  for (int i = 0; i < numInputs; ++i)
  {
    inputs[i] = tanh(inputs[i]);
  }
  double* outputs;
  VcudaMalloc((void **) &outputs, sizeof(double) * numOutputs);
  for (int j = 0; j < numOutputs; ++j)
  {
    double sum = weights[j][0]; //the bias
    for (int k = 1; k < numInputs + 1; ++k)
    {
      sum += inputs[k - 1] * weights[j][k]; //weight inputs
    }
    outputs[j] = tanh(sum); //apply output activation
  }
  return outputs;
}

__device__ VX3_DistributedNeuralController::VX3_DistributedNeuralController(double** weights, VX3_VoxelyzeKernel* kernel)
{
  this->numVoxels = kernel->num_d_voxels;
  mlp = new VX3_MLP(NUM_SENSORS + NUM_SIGNALS, NUM_SIGNALS + 2, weights);
  for (int i = 0; i < kernel->num_d_voxels; ++i) {
    VX3_Voxel* voxel = kernel->d_voxels + i;
    voxel->initLastSignals(NUM_SIGNALS);
    voxel->initCurrSignals(NUM_SIGNALS);
    for (int i = 0; i < NUM_SIGNALS; ++i) {
      voxel->lastSignals[i] = 0.0;
      voxel->currSignals[i] = 0.0;
    }
  }
}

__device__ double VX3_DistributedNeuralController::updateVoxelTemp(VX3_Voxel* voxel, VX3_VoxelyzeKernel* kernel)
{
  double* sensors = new double[NUM_SENSORS];
  for (int i = 0 ; i < NUM_SENSORS; ++i) {
    sensors[i] = -1.0;
  }
  sense(voxel, sensors, kernel);
  
  double* signals = getLastSignals(voxel);
  double* inputs = new double[mlp->getNumInputs()];
  for (int i = 0; i < NUM_SIGNALS + NUM_SENSORS; ++i) {
    inputs[i] = (i < NUM_SENSORS) ? sensors[i] : signals[i - NUM_SENSORS];
  }
  double* outputs = mlp->apply(inputs);
  
  double actuation = outputs[0];
  delete[] sensors;
  VcudaFree(signals);
  delete[] inputs;
  VcudaFree(outputs);
  return actuation;
}

__device__ void VX3_DistributedNeuralController::updateLastSignals(VX3_VoxelyzeKernel* kernel)
{
  for (int i = 0; i < kernel->num_d_voxels; ++i) {
    VX3_Voxel* voxel = kernel->d_voxels + i;
    for (int j = 0; j < NUM_SIGNALS; ++j) {
      voxel->lastSignals[j] = voxel->currSignals[j]; 
    }
  }
}

__device__ double* VX3_DistributedNeuralController::getLastSignals(VX3_Voxel* voxel) const
{
  double* signals;
  VcudaMalloc((void **) &signals, sizeof(double) * NUM_SIGNALS);
  for (int dir = 0; dir < NUM_SIGNALS; ++dir) {
    VX3_Voxel* adjVoxel = voxel->adjacentVoxel((linkDirection)dir); 
    signals[dir] = (adjVoxel) ? adjVoxel->lastSignals[dir] : 0.0;
  }
  return signals;
}

__device__ void VX3_DistributedNeuralController::sense(VX3_Voxel* voxel, double* sensors, VX3_VoxelyzeKernel* kernel) const
{
  VX3_dVector<VX3_Collision*> collisions = VX3_dVector<VX3_Collision*>();
  for (int j = 0; j < kernel->d_v_collisions.size(); ++j) {
    VX3_Collision* collision = kernel->d_v_collisions.get(j);
    if (collision->pV1 == voxel || collision->pV2 == voxel) {
      collisions.push_back(collision);
    }
  }
  
  for (int j = 0; j < collisions.size(); ++j) {
    VX3_Collision* collision = collisions.get(j);
    if (collision->force == VX3_Vec3D<float>(0,0,0)) {
      continue;
    }
    for (int i = 0; i < NUM_SENSORS; ++i) {
      VX3_Vec3D<float>* offset = getOffset((linkDirection)i);
      double s = voxel->material()->nominalSize();
      VX3_Voxel* other = (collision->pV1 == voxel) ? collision->pV2 : collision->pV1;
      if (VX3_Vec3D<float>(other->pos.x / s + offset->x, other->pos.y / s + offset->y, other->pos.z / s + offset->z) == 
          VX3_Vec3D<float>(voxel->pos.x / s + offset->x, voxel->pos.y / s + offset->y, voxel->pos.z / s + offset->z)) {
        sensors[i] = 1.0;
      }
    }
  }
  
  if (voxel->iz == 0) {
    sensors[5] = (voxel->floorPenetration() >= 0) ? 1.0 : -1.0;
  }
}

__device__ VX3_Vec3D<float>* VX3_DistributedNeuralController::getOffset(const linkDirection dir) const
{
  switch (dir) {
    case 0:
      return new VX3_Vec3D<float>(1,0,0);
    case 1:
      return new VX3_Vec3D<float>(-1,0,0);
    case 2:
      return new VX3_Vec3D<float>(0,1,0);
    case 3:
      return new VX3_Vec3D<float>(0,-1,0);
    case 4:
      return new VX3_Vec3D<float>(0,0,1);
    default:
      return new VX3_Vec3D<float>(0,0,-1);
  }
}

__device__ VX3_DistributedNeuralController::~VX3_DistributedNeuralController(void) {
  VcudaFree(mlp);
}
