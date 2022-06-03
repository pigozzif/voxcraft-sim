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

VX3_MLP::~VX3_MLP(void)
{
  for (int i = 0; i < numOutputs; ++i) {
    MycudaFree(weights[i]);
  }
  MycudaFree(weights);
}

__device__ void VX3_MLP::init(const int numInputs, const int numOutputs, double** weights)
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
  /*std::string delim = ",";
  std::size_t start = 0U;
  std::size_t end = weights->find(delim);
  int i = 0;
  int j = 0;
  while (end != std::string::npos) {
    this->weights[i][j++] = atof(weights->substr(start, end - start).c_str());
    if (j >= numInputs + 1) {
      j = 0;
      ++i;
    }
    start = end + delim.length();
    end = weights->find(delim, start);
  }*/
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

__device__ void VX3_NeuralDistributedController::init(double** weights, VX3_VoxelyzeKernel* kernel)
{
  this->numVoxels = kernel->num_d_voxels;
  mlp = new VX3_MLP();
  mlp->init(NUM_SENSORS + NUM_SIGNALS, NUM_SIGNALS + 2, weights);
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

__device__ double VX3_NeuralDistributedController::updateVoxelTemp(VX3_Voxel* voxel, VX3_VoxelyzeKernel* kernel)
{
  double* sensors = new double[NUM_SENSORS];
  //VcudaMalloc((void **) &sensors, sizeof(double) * NUM_SENSORS);
  for (int i = 0 ; i < NUM_SENSORS; ++i) {
    sensors[i] = -1.0;
  }
  sense(voxel, sensors, kernel);
  
  /*double* signals = getLastSignals(voxel, pObj);
  double* inputs = new double[mlp->getNumInputs()];
  std::copy(sensors, sensors + NUM_SENSORS, inputs);
  std::copy(signals, signals + NUM_SIGNALS, inputs + NUM_SENSORS);
  double* outputs = mlp->apply(inputs);
  
  double actuation = outputs[0];
  MycudaFree(sensors);
  MycudaFree(signals);
  delete[] inputs;
  MycudaFree(outputs);*/
  return 0.0;//actuation;
}/*

__device__ void VX3_NeuralDistributedController::updateLastSignals(VX3_VoxelyzeKernel* kernel)
{
  for (int i = 0; i < kernel->num_d_voxels; ++i) {
    std::copy(currSignals[kernel->d_voxels + i], currSignals[kernel->d_voxels + i] + NUM_SIGNALS, lastSignals[kernel->d_voxels + i]);
  }
}

__device__ double* VX3_NeuralDistributedController::getLastSignals(VX3_Voxel* voxel) const
{
  double* signals
  VcudaMalloc((void **) &signals, sizeof(double) * NUM_SIGNALS);
  for (int dir = 0; dir < NUM_SIGNALS; ++dir) {
    VX3_Voxel* adjVoxel = voxel->adjacentVoxel((linkDirection)dir); 
    signals[dir] = (adjVoxel) ? lastSignals[adjVoxel][dir] : 0.0;
  }
  return signals;
}*/

__device__ void VX3_NeuralDistributedController::sense(VX3_Voxel* voxel, double* sensors, VX3_VoxelyzeKernel* kernel) const
{
  VX3_dVector<VX3_Collision*> collisions = VX3_dVector<VX3_Collision*>();
  for (int j = 0; j < kernel->d_v_collisions.size(); ++j) {
    VX3_Collision* collision = kernel->d_v_collision.get(i);
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

__device__ VX3_Vec3D<float>* VX3_NeuralDistributedController::getOffset(const linkDirection dir) const
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
