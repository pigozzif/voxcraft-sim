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
  VcudaFree(weights);
}

__device__ VX3_MLP::VX3_MLP(const int numInputs, const int numOutputs, double* weights) {//double a, double b, double c, double d, double e, double f, double g, double h, double i, double j, double k, double l, double m, double n, double o, double p, double q, double r, double s, double t, double u, double v, double w, double x, double y, double z, double aa, double ab, double ac, double ad, double ae, double af, double ag, double ah, double ai, double aj, double ak, double al, double am, double an, double ao, double ap, double aq, double ar, double as, double at, double au, double av, double aw, double ax, double ay, double az, double ba, double bb, double bc, double bd, double be, double bf, double bg, double bh, double bi, double bj, double bk, double bl, double bm, double bn, double bo, double bp, double bq, double br, double bs, double bt, double bu, double bv, double bw, double bx, double by, double bz, double ca, double cb, double cc, double cd, double ce, double cf, double cg, double ch, double ci, double cj, double ck, double cl, double cm, double cn, double co, double cp, double cq, double cr, double cs, double ct, double cu, double cv, double cw, double cx, double cy, double cz) {
  this->numInputs = numInputs;
  this->numOutputs = numOutputs;
  this->weights = weights;
  //printf("Weights size: %d\n", weights->size());
  for (int i = 0; i < 12 * 8 + 8; ++i) {
    printf("%f,", weights[i]);
  }
  //setWeights(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, aa, ab, ac, ad, ae, af, ag, ah, ai, aj, ak, al, am, an, ao, ap, aq, ar, as, at, au, av, aw, ax, ay, az, ba, bb, bc, bd, be, bf, bg, bh, bi, bj, bk, bl, bm, bn, bo, bp, bq, br, bs, bt, bu, bv, bw, bx, by, bz, ca, cb, cc, cd, ce, cf, cg, ch, ci, cj, ck, cl, cm, cn, co, cp, cq, cr, cs, ct, cu, cv, cw, cx, cy, cz);
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

__device__ VX3_DistributedNeuralController::VX3_DistributedNeuralController(VX3_VoxelyzeKernel* kernel, double* weights) {
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
  tempVotes = new VX3_dVector<double>();
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
  
  for (int dir = 0; dir < NUM_SIGNALS; ++dir) {
    voxel->currSignals[dir] = voxel->outputs[2 + ((dir % 2 == 0) ? dir + 1 : dir - 1)];
  }
  if (firstRightContact && firstLeftContact) {
    tempVotes->push_back(voxel->outputs[1]);
  }
  return voxel->outputs[0];
}

__device__ void VX3_DistributedNeuralController::vote(void) const {
  if (!firstLeftContact && !firstRightContact) {
    return;
  }
  int numPos = 0;
  int numNeg = 0;
  for (int i = 0; i < tempVotes->size(); ++i) {
    if (tempVotes->get(i) > 0.0) {
      numPos += 1;
    }
    else {
      numNeg += 1;
    }
  }
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
  for (int dir = 0; dir < NUM_SIGNALS; ++dir) {
    VX3_Voxel* adjVoxel = voxel->adjacentVoxel((linkDirection)dir); 
    voxel->inputs[dir + NUM_SENSORS] = (adjVoxel) ? adjVoxel->lastSignals[dir] : 0.0;
  }
}

__device__ void VX3_DistributedNeuralController::sense(VX3_Voxel* voxel, VX3_VoxelyzeKernel* kernel) {
  for (int i = 0; i < NUM_SENSORS; ++i) {
    if (i == 0) {
      voxel->inputs[i] = sin(-2 * 3.14159 * kernel->CurStepCount);
    }
    else {
      voxel->inputs[i] = 0.0;
    }
    /*VX3_Vec3D<float>* corner_pos = voxel->cornerPosition((voxelCorner)i);
    if (kernel->check_left_wall_collision(corner_pos)) {
      voxel->inputs[i] = 1.0;
      if (!firstLeftContact) {
        firstLeftContact = true;
      }
    }
    else if (kernel->check_right_wall_collision(corner_pos)) {
      voxel->inputs[i] = 1.0;
      if (!firstRightContact) {
        firstRightContact = true;
      }
    }*/
  }
   /*for (int j = 0; j < voxel->collisions.size(); ++j) {
    VX3_Collision* collision = voxel->collisions.get(j);
    //if (!collision) {
    //  continue;
    //}
    if (!collision->pV1 || !collision->pV2) printf("One is NULL\n"); 
    else printf("COLLISION\n");
    if (collision->pV1 && collision->pV2) printf("COLLISION BETWEEN (%d,%d,%d) AND (%d,%d,%d)\n", collision->pV1->ix, collision->pV1->iy, collision->pV1->iz, collision->pV2->ix, collision->pV2->iy, collision->pV2->iz);
    if (collision->pV1 == voxel || collision->pV2 == voxel) {
      if (collision->force == VX3_Vec3D<float>(0,0,0)) {
        printf("ZERO FORCE\n");
        continue;
      }
      printf("We made it!\n");
      for (int i = 0; i < NUM_SENSORS; ++i) {
        VX3_Vec3D<float>* offset = getOffset((linkDirection)i);
        double s = voxel->material()->nominalSize();
        VX3_Voxel* other = (collision->pV1 == voxel) ? collision->pV2 : collision->pV1;
        if (VX3_Vec3D<float>(other->pos.x / s + offset->x, other->pos.y / s + offset->y, other->pos.z / s + offset->z) == 
            VX3_Vec3D<float>(voxel->pos.x / s + offset->x, voxel->pos.y / s + offset->y, voxel->pos.z / s + offset->z)) {
          voxel->inputs[i] = 1.0;
          if (!firstRightContact && other->matid == 1) {
            firstRightContact = true;
          }
          if (!firstLeftContact && other->matid == 2) {
            firstLeftContact = true;
          }
        }
      }
    }
  }*/
  
  /*if (voxel->iz == 0) {
    voxel->inputs[5] = (voxel->floorPenetration() >= 0) ? 1.0 : -1.0;
  }*/
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
