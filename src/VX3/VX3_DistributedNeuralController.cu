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
  VcudaFree(weights_x);
  VcudaFree(weights_h);
  VcudaFree(weights_y);
}

__device__ VX3_MLP::VX3_MLP(const int numInputs, const int numOutputs, int numHidden, double* weights_x, double* weights_h, double* weights_y) {
  this->numInputs = numInputs;
  this->numOutputs = numOutputs;
  this->numHidden = numHidden;
  this->weights_x = weights_x;
  this->weights_h = weights_h;
  this->weights_y = weights_y;
}

__device__ void VX3_MLP::apply(VX3_Voxel* voxel) const {
  //apply input activation
  for (int i = 0; i < numInputs; ++i) {
    voxel->inputs[i] = tanh(voxel->inputs[i]);
  }
  for (int j = 0; j < numHidden; ++j) {
    double sum = 0.0;
    for (int k = 0; k < numInputs; ++k) {
      sum += voxel->inputs[k] * weights_x[j * numInputs + k]; //weight inputs
    }
    voxel->temp_hidden[j] = tanh(sum); //apply output activation
  }
  for (int j = 0; j < numHidden; ++j) {
    double sum = weights_h[j * (numHidden + 1)] + voxel->temp_hidden[j]; //the bias
    for (int k = 1; k < numHidden + 1; ++k) {
      sum += voxel->prev_hidden[k - 1] * weights_h[j * (numHidden + 1) + k]; //weight inputs
    }
    voxel->hidden[j] = tanh(sum); //apply output activation
  }
  for (int j = 0; j < numOutputs; ++j) {
    double sum = weights_y[j * (numHidden + 1)]; //the bias
    for (int k = 1; k < numHidden + 1; ++k) {
      sum += voxel->hidden[k - 1] * weights_y[j * (numHidden + 1) + k]; //weight inputs
    }
    voxel->outputs[j] = tanh(sum); //apply output activation
  }
}

__device__ VX3_DistributedNeuralController::VX3_DistributedNeuralController(VX3_VoxelyzeKernel* kernel, double* weights_x, double* weights_h, double* weights_y, int random_seed) {
  mlp = new VX3_MLP(NUM_SENSORS + NUM_SIGNALS, 2, NUM_HIDDEN, weights_x, weights_h, weights_y);
  for (int i = 0; i < kernel->num_d_voxels; ++i) {
    VX3_Voxel* voxel = kernel->d_voxels + i;
    voxel->initArrays(mlp->numInputs, mlp->numOutputs, mlp->numHidden, NUM_SIGNALS);
    for (int i = 0; i < mlp->numOutputs; ++i) {
      voxel->outputs[i] = 0.0;
    }
    for (int i = 0; i < mlp->numHidden; ++i) {
      voxel->hidden[i] = 0.0;
      voxel->prev_hidden[i] = 0.0;
      voxel->temp_hidden[i] = 0.0;
    }
    for (int i = 0; i < NUM_SIGNALS; ++i) {
      voxel->inputs[NUM_SENSORS + i] = 0.0;
      voxel->lastSignals[i] = 0.0;
      voxel->currSignals[i] = 0.0;
    }
  }
  firstRightContact = false;
  firstLeftContact = false;
}

__device__ double VX3_DistributedNeuralController::updateVoxelTemp(VX3_Voxel* voxel, VX3_VoxelyzeKernel* kernel) {
  sense(voxel, kernel);
  
  getLastSignals(voxel);
  mlp->apply(voxel);
  
  //int id = voxel->iy * 9 + voxel->ix;
  //int vote = random(1000, clock() + id);
  //int vote2 = random(1000, clock() + id + 20);
  //voxel->outputs[0] = (vote - 500.0) / 500.0;
  //voxel->outputs[1] = (vote2 - 500.0) / 500.0;
  for (int dir = 0; dir < NUM_SIGNALS / 2; ++dir) {
    int new_dir = dir * 2;
    voxel->currSignals[new_dir] = voxel->outputs[0];
    voxel->currSignals[new_dir + 1] = voxel->outputs[1];
  }
  if (firstRightContact || firstLeftContact) {
    voxel->vote = {voxel->outputs[1], voxel->ix, voxel->iy, voxel->iz, (voxel->inputs[1] > 0.0) ? 1 : 0};
  }
  return voxel->outputs[0];
}

__device__ void VX3_DistributedNeuralController::printVotes(VX3_VoxelyzeKernel* kernel) {
  printf("%ld:", kernel->CurStepCount);
  for (int i = 0; i < kernel->num_d_voxels; ++i) {
    if (kernel->d_voxels[i].matid == 4) {
      printf("%f,%d,%d,%d,%d/", kernel->d_voxels[i].vote.v, kernel->d_voxels[i].vote.x, kernel->d_voxels[i].vote.y, kernel->d_voxels[i].vote.z, kernel->d_voxels[i].vote.is_touching);
    }
  }
  printf("\n");
}

__device__ void VX3_DistributedNeuralController::vote(VX3_VoxelyzeKernel* kernel) {
  if (!firstLeftContact && !firstRightContact) {
    return;
  }
  for (int i = 0; i < kernel->num_d_voxels; ++i) {
    if (kernel->d_voxels[i].matid == 4 && kernel->d_voxels[i].vote.v > 0.0) {
      numPos += 1.0;
    }
  }
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
  for (int dir = 0; dir < NUM_SIGNALS / 2; ++dir) {
    VX3_Voxel* adjVoxel = voxel->adjacentVoxel((linkDirection)dir);
    for (int i = 0; i < 2; ++i) {
      voxel->inputs[dir * 2 + i + NUM_SENSORS] = (adjVoxel) ? adjVoxel->lastSignals[dir * 2 + i] : 0.0;
    }
  }
}

__device__ void VX3_DistributedNeuralController::sense(VX3_Voxel* voxel, VX3_VoxelyzeKernel* kernel) {
  voxel->inputs[0] = sin(-2 * 3.14159 * kernel->CurStepCount);
  voxel->inputs[1] = (voxel->collisions.size() != 0) ? 1.0 : -1.0;
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
    bool is_flying = voxel->floorPenetration() < 0;
    voxel->inputs[2] = (is_flying) ? -1.0 : 1.0;
    kernel->flying_voxels += (is_flying) ? 1 : 0;
  }
  voxel->inputs[3] = voxel->velocity().y;
  voxel->inputs[4] = voxel->velocity().x;
  if (kernel->CurStepCount != 0 && ((kernel->min_x > voxel->pos.x) || (voxel->pos.x > kernel->max_x) || (voxel->pos.z > kernel->max_z))) {
    kernel->out_of_bounds = 1;
  }
}

__device__ bool VX3_DistributedNeuralController::has(int* values, int value, int n) {
  for (int i = 0; i < n; ++i) {
    if (values[i] == value) {
      return true;
    }
  }
  return false;
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
