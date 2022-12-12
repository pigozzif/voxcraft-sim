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
  mlp = new VX3_MLP(NUM_SENSORS + NUM_SIGNALS, 2, weights);
  for (int i = 0; i < kernel->num_d_voxels; ++i) {
    VX3_Voxel* voxel = kernel->d_voxels + i;
    voxel->initArrays(mlp->numInputs, mlp->numOutputs, NUM_SIGNALS);
    for (int i = 0; i < mlp->numOutputs; ++i) {
      voxel->outputs[i] = 0.0;
    }
    for (int i = 0; i < NUM_SIGNALS; ++i) {
      voxel->inputs[NUM_SENSORS + i] = 0.0;
      voxel->lastSignals[i] = 0.0;
      voxel->currSignals[i] = 0.0;
    }
  }
  votes = new VX3_dVector<int>();
  tempVotes = new VX3_dVector<Vote>();
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
    tempVotes->push_back({voxel->outputs[1], voxel->ix, voxel->iy, voxel->iz, (voxel->inputs[1] > 0.0) ? 1 : 0});
  }
  return voxel->outputs[0];
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
  /*if (numPos == 0) {
    votes->push_back(0);
  }
  else if (numNeg == 0) {
    votes->push_back(1);
  }
  else {
    votes->push_back(-1);
  }*/
  votes->push_back(numPos);
  //votes->push_back((numPos >= numNeg) ? 1 : 0);
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
