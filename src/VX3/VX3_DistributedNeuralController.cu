#include "VX3_DistributedNeuralController.h"
#include "VX3.cuh"
#include "VX3_Voxel.h"
#include "VX3_VoxelyzeKernel.cuh"
#include "VX3_Collision.h"
#include "VX3_MemoryCleaner.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

/*__device__ VX3_MLP::~VX3_MLP(void) {
  VcudaFree(weights);
  VcudaFree(outputs);
  VcudaFree(inputs);
}*/

__device__ VX3_MLP::VX3_MLP(const int numInputs, const int numOutputs, double a, double b, double c, double d, double e, double f, double g, double h, double i, double j, double k, double l, double m, double n, double o, double p, double q, double r, double s, double t, double u, double v, double w, double x, double y, double z, double aa, double ab, double ac, double ad, double ae, double af, double ag, double ah, double ai, double aj, double ak, double al, double am, double an, double ao, double ap, double aq, double ar, double as, double at, double au, double av, double aw, double ax, double ay, double az, double ba, double bb, double bc, double bd, double be, double bf, double bg, double bh, double bi, double bj, double bk, double bl, double bm, double bn, double bo, double bp, double bq, double br, double bs, double bt, double bu, double bv, double bw, double bx, double by, double bz, double ca, double cb, double cc, double cd, double ce, double cf, double cg, double ch, double ci, double cj, double ck, double cl, double cm, double cn, double co, double cp, double cq, double cr, double cs, double ct, double cu, double cv, double cw, double cx, double cy, double cz) {
  this->numInputs = numInputs;
  this->numOutputs = numOutputs;
  VcudaMalloc((void**) &outputs, sizeof(double) * numOutputs);
  VcudaMalloc((void**) &inputs, sizeof(double) * numInputs);
  setWeights(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, aa, ab, ac, ad, ae, af, ag, ah, ai, aj, ak, al, am, an, ao, ap, aq, ar, as, at, au, av, aw, ax, ay, az, ba, bb, bc, bd, be, bf, bg, bh, bi, bj, bk, bl, bm, bn, bo, bp, bq, br, bs, bt, bu, bv, bw, bx, by, bz, ca, cb, cc, cd, ce, cf, cg, ch, ci, cj, ck, cl, cm, cn, co, cp, cq, cr, cs, ct, cu, cv, cw, cx, cy, cz);
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

__device__ VX3_DistributedNeuralController::VX3_DistributedNeuralController(VX3_VoxelyzeKernel* kernel, double a, double b, double c, double d, double e, double f, double g, double h, double i, double j, double k, double l, double m, double n, double o, double p, double q, double r, double s, double t, double u, double v, double w, double x, double y, double z, double aa, double ab, double ac, double ad, double ae, double af, double ag, double ah, double ai, double aj, double ak, double al, double am, double an, double ao, double ap, double aq, double ar, double as, double at, double au, double av, double aw, double ax, double ay, double az, double ba, double bb, double bc, double bd, double be, double bf, double bg, double bh, double bi, double bj, double bk, double bl, double bm, double bn, double bo, double bp, double bq, double br, double bs, double bt, double bu, double bv, double bw, double bx, double by, double bz, double ca, double cb, double cc, double cd, double ce, double cf, double cg, double ch, double ci, double cj, double ck, double cl, double cm, double cn, double co, double cp, double cq, double cr, double cs, double ct, double cu, double cv, double cw, double cx, double cy, double cz) {
  mlp = new VX3_MLP(NUM_SENSORS + NUM_SIGNALS, NUM_SIGNALS + 2, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, aa, ab, ac, ad, ae, af, ag, ah, ai, aj, ak, al, am, an, ao, ap, aq, ar, as, at, au, av, aw, ax, ay, az, ba, bb, bc, bd, be, bf, bg, bh, bi, bj, bk, bl, bm, bn, bo, bp, bq, br, bs, bt, bu, bv, bw, bx, by, bz, ca, cb, cc, cd, ce, cf, cg, ch, ci, cj, ck, cl, cm, cn, co, cp, cq, cr, cs, ct, cu, cv, cw, cx, cy, cz);
  for (int i = 0; i < kernel->num_d_voxels; ++i) {
    VX3_Voxel* voxel = kernel->d_voxels + i;
    voxel->initLastSignals(NUM_SIGNALS);
    voxel->initCurrSignals(NUM_SIGNALS);
    for (int i = 0; i < NUM_SIGNALS; ++i) {
      voxel->lastSignals[i] = 0.0;
      voxel->currSignals[i] = 0.0;
    }
  }
  votes = new VX3_dVector<int>();
  tempVotes = new VX3_dVector<double>();
  firstRightContact = false;
  firstLeftContact = false;
  for (int i = 0; i < (mlp->numInputs + 1) * (mlp->numOutputs); ++i) printf("%f,", mlp->weights[i]);
  printf("\n");
}

__device__ double VX3_DistributedNeuralController::updateVoxelTemp(VX3_Voxel* voxel, VX3_VoxelyzeKernel* kernel) {
  for (int i = 0 ; i < NUM_SENSORS; ++i) {
    mlp->inputs[i] = -1.0;
  }
  sense(voxel, kernel);
  
  getLastSignals(voxel);
  mlp->apply();
  
  for (int dir = 0; dir < NUM_SIGNALS; ++dir) {
    voxel->currSignals[dir] = mlp->outputs[2 + ((dir % 2 == 0) ? dir + 1 : dir - 1)];
  }
  if (firstRightContact && firstLeftContact) {
    tempVotes->push_back(mlp->outputs[1]);
  }
  return mlp->outputs[0];
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
    mlp->inputs[dir + NUM_SENSORS] = (adjVoxel) ? adjVoxel->lastSignals[dir] : 0.0;
  }
}

__device__ void VX3_DistributedNeuralController::sense(VX3_Voxel* voxel, VX3_VoxelyzeKernel* kernel) {
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
          if (!firstRightContact && other->matid == 1) {
            firstRightContact = true;
          }
          if (!firstLeftContact && other->matid == 2) {
            firstLeftContact = true;
          }
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

__device__ void VX3_MLP::setWeights(double a, double b, double c, double d, double e, double f, double g, double h, double i, double j, double k, double l, double m, double n, double o, double p, double q, double r, double s, double t, double u, double v, double w, double x, double y, double z, double aa, double ab, double ac, double ad, double ae, double af, double ag, double ah, double ai, double aj, double ak, double al, double am, double an, double ao, double ap, double aq, double ar, double as, double at, double au, double av, double aw, double ax, double ay, double az, double ba, double bb, double bc, double bd, double be, double bf, double bg, double bh, double bi, double bj, double bk, double bl, double bm, double bn, double bo, double bp, double bq, double br, double bs, double bt, double bu, double bv, double bw, double bx, double by, double bz, double ca, double cb, double cc, double cd, double ce, double cf, double cg, double ch, double ci, double cj, double ck, double cl, double cm, double cn, double co, double cp, double cq, double cr, double cs, double ct, double cu, double cv, double cw, double cx, double cy, double cz) {
  VcudaMalloc((void**) &weights, sizeof(double*) * numOutputs * (numInputs + 1));
  weights[0] = a;
  weights[1] = b;
  weights[2] = c;
  weights[3] = d;
  weights[4] = e;
  weights[5] = f;
  weights[6] = g;
  weights[7] = h;
  weights[8] = i;
  weights[9] = j;
  weights[10] = k;
  weights[11] = l;
  weights[12] = m;
  weights[13] = n;
  weights[14] = o;
  weights[15] = p;
  weights[16] = q;
  weights[17] = r;
  weights[18] = s;
  weights[19] = t;
  weights[20] = u;
  weights[21] = v;
  weights[22] = w;
  weights[23] = x;
  weights[24] = y;
  weights[25] = z;
  weights[26] = aa;
  weights[27] = ab;
  weights[28] = ac;
  weights[29] = ad;
  weights[30] = ae;
  weights[31] = af;
  weights[32] = ag;
  weights[33] = ah;
  weights[34] = ai;
  weights[35] = aj;
  weights[36] = ak;
  weights[37] = al;
  weights[38] = am;
  weights[39] = an;
  weights[40] = ao;
  weights[41] = ap;
  weights[42] = aq;
  weights[43] = ar;
  weights[44] = as;
  weights[45] = at;
  weights[46] = au;
  weights[47] = av;
  weights[48] = aw;
  weights[49] = ax;
  weights[50] = ay;
  weights[51] = az;
  weights[52] = ba;
  weights[53] = bb;
  weights[54] = bc;
  weights[55] = bd;
  weights[56] = be;
  weights[57] = bf;
  weights[58] = bg;
  weights[59] = bh;
  weights[60] = bi;
  weights[61] = bj;
  weights[62] = bk;
  weights[63] = bl;
  weights[64] = bm;
  weights[65] = bn;
  weights[66] = bo;
  weights[67] = bp;
  weights[68] = bq;
  weights[69] = br;
  weights[70] = bs;
  weights[71] = bt;
  weights[72] = bu;
  weights[73] = bv;
  weights[74] = bw;
  weights[75] = bx;
  weights[76] = by;
  weights[77] = bz;
  weights[78] = ca;
  weights[79] = cb;
  weights[80] = cc;
  weights[81] = cd;
  weights[82] = ce;
  weights[83] = cf;
  weights[84] = cg;
  weights[85] = ch;
  weights[86] = ci;
  weights[87] = cj;
  weights[88] = ck;
  weights[89] = cl;
  weights[90] = cm;
  weights[91] = cn;
  weights[92] = co;
  weights[93] = cp;
  weights[94] = cq;
  weights[95] = cr;
  weights[96] = cs;
  weights[97] = ct;
  weights[98] = cu;
  weights[99] = cv;
  weights[100] = cw;
  weights[101] = cx;
  weights[102] = cy;
  weights[103] = cz;
}
