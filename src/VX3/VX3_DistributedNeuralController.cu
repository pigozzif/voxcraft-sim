#include "VX3_DistributedNeuralController.h"
#include "VX3.cuh"
#include "VX3_Voxel.h"
#include "VX3_VoxelyzeKernel.cuh"
#include "VX3_Collision.h"
#include <algorithm>
#include <stdlib.h>
#include <cstdlib>
#include <math.h>
#include <string>
#include <map>

__device__ CVX_MLP::CVX_MLP(const int numInputs, const int numOutputs, const std::string weights)
{
  this->numInputs = numInputs;
  this->numOutputs = numOutputs;
  setWeights(weights);
}

__device__ CVX_MLP::~CVX_MLP(void)
{
  for (int i = 0; i < numOutputs; ++i) {
    free(weights[i]);
  }
  free(weights);
}

__device__ void CVX_MLP::setWeights(const std::string weights)
{
  this->weights = (double**) malloc(sizeof(double*) * numOutputs);
  for (int i = 0; i < numOutputs; ++i) {
    this->weights[i] = (double*) malloc(sizeof(double) * (numInputs + 1));
  }
  std::string delim = ",";
  std::size_t start = 0U;
  std::size_t end = weights.find(delim);
  int i = 0;
  int j = 0;
  while (end != std::string::npos) {
    this->weights[i][j++] = atof(weights.substr(start, end - start).c_str());
    if (j >= numInputs + 1) {
      j = 0;
      ++i;
    }
    start = end + delim.length();
    end = weights.find(delim, start);
  }
}

__device__ double* CVX_MLP::apply(double* inputs) const
{
  //apply input activation
  for (int i = 0; i < numInputs; ++i)
  {
    inputs[i] = tanh(inputs[i]);
  }
  double* outputs = (double*) malloc(sizeof(double) * numOutputs);
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

__device__ CVX_Distributed::CVX_Distributed(const std::string weights, VX3_VoxelyzeKernel* kernel)
{
  this->numVoxels = kernel->num_d_voxels;
  mlp = new CVX_MLP(NUM_SENSORS + NUM_SIGNALS, NUM_SIGNALS + 2, weights);
  lastSignals = new std::map<VX3_Voxel*, double[]>();
  currSignals = new std::map<VX3_Voxel*, double[]>();
  for (VX3_Voxel* voxel : kernel->d_voxels) {
    lastSignals.emplace(voxel, new double[4]);
    currSignals.emplace(voxel, new double[4]);
    std::fill(lastSignals[voxel], currSignals[voxel] + NUM_SIGNALS, 0.0);
    std::fill(currSignals[voxel], currSignals[voxel] + NUM_SIGNALS, 0.0);
  }
}

__device__ CVX_Distributed::~CVX_Distributed(void)
{
  delete mlp;
}

__device__ double CVX_Distributed::updateVoxelTemp(VX3_Voxel* voxel, VX3_VoxelyzeKernel* kernel)
{
  double* sensors = (double*) malloc(sizeof(double) * NUM_SENSORS);
  std::fill(sensors, sensors + NUM_SENSORS, -1.0);
  sense(voxel, sensors, kernel);
  
  double* signals = getLastSignals(voxel, pObj);
  double* inputs = new double[mlp->getNumInputs()];
  std::copy(sensors, sensors + NUM_SENSORS, inputs);
  std::copy(signals, signals + NUM_SIGNALS, inputs + NUM_SENSORS);
  double* outputs = mlp->apply(inputs);
  
  double actuation = outputs[0];
  free(sensors);
  free(signals);
  free(inputs);
  free(outputs);
  return actuation;
}

__device__ void CVX_Distributed::updateLastSignals(VX3_VoxelyzeKernel* kernel)
{
  for (int i = 0; i < kernel->num_d_voxels; ++i) {
    std::copy(currSignals[kernel->d_voxels + i], currSignals[kernel->d_voxels + i] + NUM_SIGNALS, lastSignals[kernel->d_voxels + i]);
  }
}

__device__ double* CVX_Distributed::getLastSignals(VX3_Voxel* voxel) const
{
  double* signals = (double*) malloc(sizeof(double) * NUM_SIGNALS);
  for (int dir = 0; dir < NUM_SIGNALS; ++dir) {
    VX3_Voxel* adjVoxel = voxel->adjacentVoxel((linkDirection)dir); 
    signals[dir] = (adjVoxel) ? lastSignals[adjVoxel][dir] : 0.0;
  }
  return signals;
}

__device__ void CVX_Distributed::sense(VX3_Voxel* voxel, double* sensors, VX3_VoxelyzeKernel* kernel) const
{
  VX3_dVector<VX3_Collision*> collisions = VX3_dVector<VX3_Collision*>();
  for (VX3_Collision* collision : kernel->d_v_collisions) {
    if (collision->pV1 == voxel || collision->pV2 == voxel) {
      collisions.push_back(collision);
    }
  }
  
  for (VX3_Collision* collision : collisions) {
    if (collision->force == VX3_Vec3D<float>(0,0,0)) {
      continue;
    }
    for (int i = 0; i < NUM_SENSORS; ++i) {
      VX3_Vec3D<float>* offset = getOffset((VX3_Voxel::linkDirection)i);
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

__device__ VX3_Vec3D<float>* CVX_Distributed::getOffset(const linkDirection dir) const
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
