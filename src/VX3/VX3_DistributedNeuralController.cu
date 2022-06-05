#include "VX3_DistributedNeuralController.h"
#include "VX3.cuh"
#include "VX3_Voxel.h"
#include "VX3_VoxelyzeKernel.cuh"
#include "VX3_Collision.h"
#include "VX3_MemoryCleaner.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

__device__ VX3_MLP::~VX3_MLP(void) {
  /*for (int i = 0; i < numOutputs; ++i) {
    VcudaFree(weights[i]);
  }
  VcudaFree(weights);*/
  VcudaFree(outputs);
  VcudaFree(inputs);
}

__device__ VX3_MLP::VX3_MLP(const int numInputs, const int numOutputs, double* weights) {
  this->numInputs = numInputs;
  this->numOutputs = numOutputs;
  VcudaMalloc((void **) &outputs, sizeof(double) * numOutputs);
  VcudaMalloc((void **) &inputs, sizeof(double) * numInputs);
  printf("we are here");
  //this->weights = weights;
  //setWeights("");
}

__device__ void VX3_MLP::setWeights(char* weights) {
  VcudaMalloc((void**) &this->weights, sizeof(double*) * numOutputs);
  for (int i = 0; i < numOutputs; ++i) {
    VcudaMalloc((void**) &this->weights[i], sizeof(double) * (numInputs + 1));
  }
  /*int i = 0;
  int j = 0;
  char *p = strtok(weights, ",");
  while (p != NULL) {
    this->weights[i][j++] = atof(p);
    p = strtok(NULL, ",");
    if (j >= numInputs - 1) {
      j = 0;
      ++i;
    }
  }*/
}

__device__ void VX3_MLP::apply(void) const {
  //apply input activation
  for (int i = 0; i < numInputs; ++i) {
    inputs[i] = tanh(inputs[i]);
  }
  for (int j = 0; j < numOutputs; ++j) {
    double sum = weights[j * (numInputs + 1)]; //the bias
    for (int k = 1; k < numInputs + 1; ++k) {
      sum += inputs[k - 1] * weights[j * (numInputs + 1) + k]; //weight inputs
    }
    outputs[j] = tanh(sum); //apply output activation
  }
}

__device__ VX3_DistributedNeuralController::VX3_DistributedNeuralController(double** weights, VX3_VoxelyzeKernel* kernel) {
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

__device__ double VX3_DistributedNeuralController::updateVoxelTemp(VX3_Voxel* voxel, VX3_VoxelyzeKernel* kernel) {
  for (int i = 0 ; i < NUM_SENSORS; ++i) {
    mlp->inputs[i] = -1.0;
  }
  sense(voxel, kernel);
  
  getLastSignals(voxel);
  mlp->apply();
  
  return mlp->outputs[0];
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
    mlp->inputs[dir + NUM_SENSORS] = (adjVoxel) ? adjVoxel->lastSignals[dir] : 0.0;
  }
}

__device__ void VX3_DistributedNeuralController::sense(VX3_Voxel* voxel, VX3_VoxelyzeKernel* kernel) const {
  for (int j = 0; j < kernel->d_v_collisions.size(); ++j) {
    VX3_Collision* collision = kernel->d_v_collisions.get(j);
    if (collision->pV1 == voxel || collision->pV2 == voxel) {
      if (collision->force == VX3_Vec3D<float>(0,0,0)) {
        continue;
      }
      for (int i = 0; i < NUM_SENSORS; ++i) {
        VX3_Vec3D<float>* offset = getOffset((linkDirection)i);
        double s = voxel->material()->nominalSize();
        VX3_Voxel* other = (collision->pV1 == voxel) ? collision->pV2 : collision->pV1;
        if (VX3_Vec3D<float>(other->pos.x / s + offset->x, other->pos.y / s + offset->y, other->pos.z / s + offset->z) == 
            VX3_Vec3D<float>(voxel->pos.x / s + offset->x, voxel->pos.y / s + offset->y, voxel->pos.z / s + offset->z)) {
          mlp->inputs[i] = 1.0;
        }
      }
    }
  }
  
  if (voxel->iz == 0) {
    mlp->inputs[5] = (voxel->floorPenetration() >= 0) ? 1.0 : -1.0;
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

__device__ VX3_DistributedNeuralController::~VX3_DistributedNeuralController(void) {
  VcudaFree(mlp);
}
