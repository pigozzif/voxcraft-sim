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
  VcudaFree(abcd);
}

__device__ VX3_MLP::VX3_MLP(const int numInputs, const int numOutputs, double* weights) {
  this->numInputs = numInputs;
  this->numOutputs = numOutputs;
  this->abcd = weights;
}

__device__ void VX3_MLP::apply(VX3_Voxel* voxel) {
  //apply input activation
  for (int i = 0; i < numInputs; ++i) {
    voxel->inputs[i] = tanh(voxel->inputs[i]);
  }
  for (int j = 0; j < numOutputs; ++j) {
    double sum = 0.0;
    for (int k = 0; k < numInputs; ++k) {
      sum += voxel->inputs[k] * voxel->weights[j * numInputs + k]; //weight inputs
    }
    voxel->outputs[j] = tanh(sum); //apply output activation
  }
  for (int i = 0; i < numOutputs; ++i) {
    for (int j = 0; j < numInputs; ++j) {
      int w = ((i * numInputs) + j);
      printf("%f;", voxel->weights[w]);
    }
  }
  printf("\n");
  hebbianUpdate(voxel);
  normalizeWeights(voxel);
}

__device__ void VX3_MLP::hebbianUpdate(VX3_Voxel* voxel) {
  for (int i = 0; i < numOutputs; ++i) {
    for (int j = 0; j < numInputs; ++j) {
      int w = ((i * numInputs) + j) * 4;
      double x_j = voxel->inputs[j];
      double y_i = voxel->outputs[i];
      double dw = eta * (abcd[w] * x_j * y_i + abcd[w + 1] * x_j + abcd[w + 2] * y_i + abcd[w + 3]);
      voxel->weights[w / 4] += dw;
    }
  }
}

__device__ void VX3_MLP::normalizeWeights(VX3_Voxel* voxel) {
  for (int i = 0; i < numOutputs; ++i) {
    double norm = 0.0;
    for (int j = 0; j < numInputs; ++j) {
      double w = voxel->weights[((i * numInputs) + j)];
      norm += w * w;
    }
    norm = sqrt(norm);
    if (norm == 0.0) {
      continue;
    }
    for (int j = 0; j < numInputs; ++j) {
      voxel->weights[((i * numInputs) + j)] /= norm;
    }
  }
}

__device__ VX3_DistributedNeuralController::VX3_DistributedNeuralController(VX3_VoxelyzeKernel* kernel, double* weights, int random_seed) {
  mlp = new VX3_MLP(NUM_SENSORS + NUM_SIGNALS, NUM_SIGNALS + 2, weights);
  for (int i = 0; i < kernel->num_d_voxels; ++i) {
    VX3_Voxel* voxel = kernel->d_voxels + i;
    voxel->initArrays(mlp->numInputs, mlp->numOutputs, NUM_SIGNALS);
    voxel->outputs[0] = 0.0;
    voxel->outputs[1] = 0.0;
    for (int i = 0; i < NUM_SIGNALS; ++i) {
      voxel->inputs[NUM_SENSORS + i] = 0.0;
      voxel->outputs[2 + i] = 0.0;
      voxel->lastSignals[i] = 0.0;
      voxel->currSignals[i] = 0.0;
    }
    for (int i = 0; i < mlp->numOutputs; ++i) {
      for (int j = 0; j < mlp->numInputs; ++j) {
        voxel->weights[(i * mlp->numInputs) + j] = 0.0;
      }
    }
  }
  votes = new VX3_dVector<int>();
  tempVotes = new VX3_dVector<Vote>();
  firstRightContact = false;
  firstLeftContact = false;
}

__device__ double VX3_DistributedNeuralController::updateVoxelTemp(VX3_Voxel* voxel, VX3_VoxelyzeKernel* kernel) {
  for (int i = 0 ; i < NUM_SENSORS; ++i) {
    voxel->inputs[i] = -1.0;
  }
  sense(voxel, kernel);
  
  getLastSignals(voxel);
  mlp->apply(voxel);
  
  //int id = voxel->iy * 9 + voxel->ix;
  //int vote = random(1000, clock() + id);
  //int vote2 = random(1000, clock() + id + 20);
  //voxel->outputs[0] = (vote - 500.0) / 500.0;
  //voxel->outputs[1] = (vote2 - 500.0) / 500.0;
  for (int dir = 0; dir < NUM_SIGNALS / 2; ++dir) {
    dir = dir * 2;
    voxel->currSignals[dir] = voxel->outputs[2 + ((dir % 2 == 0) ? dir + 1 : dir - 1)];
    voxel->currSignals[dir + 1] = voxel->outputs[1];
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
  //votes->push_back(numPos);
  votes->push_back((numPos >= numNeg) ? 1 : 0);
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
  if (voxel->collisions.size() != 0) {
    voxel->inputs[1] = 1.0;
  }
  voxel->idx += 1;
  if (voxel->idx >= TOUCH_HISTORY) {
    voxel->idx = 0;
  }
  //voxel->inputs[1] = (has(voxel->touches, 1, TOUCH_HISTORY)) ? 1.0 : -1.0;
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
