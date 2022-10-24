#include "VX3_DistributedNeuralController.h"
#include "VX3.cuh"
#include "VX3_Voxel.h"
#include "VX3_VoxelyzeKernel.cuh"
#include "VX3_Collision.h"
#include "VX3_MemoryCleaner.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cstdlib>

__device__ VX3_MLP::~VX3_MLP(void) {
  VcudaFree(weights);
}

__device__ VX3_MLP::VX3_MLP(const int numInputs, const int numOutputs, double* weights) {
  this->numInputs = numInputs;
  this->numOutputs = numOutputs;
  this->weights = weights;
}

__device__ void VX3_MLP::apply(VX3_Voxel* voxel) const {
  //apply input activation
  for (int i = 0; i < numInputs; ++i) {
    voxel->inputs[i] = tanh(voxel->inputs[i]);
  }
  for (int j = 0; j < numOutputs; ++j) {
    double sum = weights[j * (numInputs + 1)]; //the bias
    for (int k = 1; k < numInputs + 1; ++k) {
      sum += voxel->inputs[k - 1] * weights[j * (numInputs + 1) + k]; //weight inputs
    }
    voxel->outputs[j] = tanh(sum); //apply output activation
  }
}

__device__ VX3_DistributedNeuralController::VX3_DistributedNeuralController(VX3_VoxelyzeKernel* kernel, double* weights, int random_seed) {
  mlp = new VX3_MLP(NUM_SENSORS + NUM_SIGNALS, NUM_SIGNALS + 2, weights);
  for (int i = 0; i < kernel->num_d_voxels; ++i) {
    VX3_Voxel* voxel = kernel->d_voxels + i;
    voxel->initArrays(mlp->numInputs, mlp->numOutputs, NUM_SIGNALS);
    for (int i = 0; i < NUM_SIGNALS; ++i) {
      voxel->inputs[i] = 0.0;
      voxel->outputs[i] = 0.0;
      voxel->lastSignals[i] = 0.0;
      voxel->currSignals[i] = 0.0;
    }
  }
  votes = new VX3_dVector<int>();
  tempVotes = new VX3_dVector<Vote>();
  firstRightContact = false;
  firstLeftContact = false;
  count = false;
}

__device__ double VX3_DistributedNeuralController::updateVoxelTemp(VX3_Voxel* voxel, VX3_VoxelyzeKernel* kernel) {
  for (int i = 0 ; i < NUM_SENSORS; ++i) {
    voxel->inputs[i] = -1.0;
  }
  sense(voxel, kernel);
  
  getLastSignals(voxel);
  mlp->apply(voxel);
  
  for (int dir = 0; dir < NUM_SIGNALS; ++dir) {
    voxel->currSignals[dir] = voxel->outputs[2 + ((dir % 2 == 0) ? dir + 1 : dir - 1)];
  }
  if (firstRightContact || firstLeftContact) {
    tempVotes->push_back({voxel->outputs[1], voxel->ix, voxel->iy, voxel->iz, (voxel->inputs[1] > 0.0) ? 1 : 0});
  }
  count = !count;
  return /*(count) ? 1 : 0;*/voxel->outputs[0];
}

__device__ void VX3_DistributedNeuralController::printVotes(VX3_VoxelyzeKernel* kernel) {
  printf("%ld:", kernel->CurStepCount);
  for (int i = 0; i < tempVotes->size(); ++i) {
    printf("%f,%d,%d,%d,%d/", tempVotes->get(i).v, tempVotes->get(i).x, tempVotes->get(i).y, tempVotes->get(i).z, tempVotes->get(i).is_touching);
  }
  printf("\n");
}

__device__ void VX3_DistributedNeuralController::vote(void) {
  if (!firstLeftContact && !firstRightContact) {
    return;
  }
  int numPos = 0;
  int numNeg = 0;
  for (int i = 0; i < tempVotes->size(); ++i) {
    if (tempVotes->get(i).v > 0.0) {
      numPos += 1;
    }
    else {
      numNeg += 1;
    }
  }
  votes->push_back(/*(count) ? 1 : 0);*/(numPos >= numNeg) ? 1 : 0);
  tempVotes->clear();
}

__device__ void VX3_DistributedNeuralController::updateLastSignals(VX3_VoxelyzeKernel* kernel) {
  for (int i = 0; i < kernel->num_d_voxels; ++i) {
    VX3_Voxel* voxel = kernel->d_voxels + i;
    for (int j = 0; j < NUM_SIGNALS; ++j) {
      voxel->lastSignals[j] = voxel->currSignals[j]; 
    }
  }
}

__device__ void VX3_DistributedNeuralController::getLastSignals(VX3_Voxel* voxel) const {
  for (int dir = 0; dir < NUM_SIGNALS; ++dir) {
    VX3_Voxel* adjVoxel = voxel->adjacentVoxel((linkDirection)dir); 
    voxel->inputs[dir + NUM_SENSORS] = (adjVoxel) ? adjVoxel->lastSignals[dir] : 0.0;
  }
}

__device__ void VX3_DistributedNeuralController::sense(VX3_Voxel* voxel, VX3_VoxelyzeKernel* kernel) {
  voxel->inputs[0] = sin(-2 * 3.14159 * kernel->CurStepCount);
  if (voxel->collisions.size() != 0) {
    voxel->inputs[1] = 1.0;
  }
  for (int j = 0; j < voxel->collisions.size(); ++j) {
    int collision = voxel->collisions.get(j);
    if (!firstRightContact && collision == 2) {
      firstRightContact = true;
    }
    if (!firstLeftContact && collision == 1) {
      firstLeftContact = true;
    }
  }
  
  if (voxel->iz == 0) {
    bool is_flying = voxel->floorPenetration() >= 0;
    voxel->inputs[2] = (is_flying) ? 1.0 : -1.0;
    kernel->flying_voxels += 1;
  }
}

__device__ VX3_Vec3D<float>* VX3_DistributedNeuralController::getOffset(const linkDirection dir) const {
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
