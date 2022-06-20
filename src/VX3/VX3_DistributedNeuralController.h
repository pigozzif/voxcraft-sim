#if !defined(VX3_DISTRIBUTED_H)
#define VX3_DISTRIBUTED_H

#include "VX3.cuh"
#include "VX3_Voxel.h"

#define NUM_SENSORS 6
#define NUM_SIGNALS 6

class VX3_VoxelyzeKernel;

class VX3_MLP
{
public:
  __device__ VX3_MLP(const int numInputs, const int numOutputs, double a, double b, double c, double d, double e, double f, double g, double h, double i, double j, double k, double l, double m, double n, double o, double p, double q, double r, double s, double t, double u, double v, double w, double x, double y, double z, double aa, double ab, double ac, double ad, double ae, double af, double ag, double ah, double ai, double aj, double ak, double al, double am, double an, double ao, double ap, double aq, double ar, double as, double at, double au, double av, double aw, double ax, double ay, double az, double ba, double bb, double bc, double bd, double be, double bf, double bg, double bh, double bi, double bj, double bk, double bl, double bm, double bn, double bo, double bp, double bq, double br, double bs, double bt, double bu, double bv, double bw, double bx, double by, double bz, double ca, double cb, double cc, double cd, double ce, double cf, double cg, double ch, double ci, double cj, double ck, double cl, double cm, double cn, double co, double cp, double cq, double cr, double cs, double ct, double cu, double cv, double cw, double cx, double cy, double cz);
  __device__ ~VX3_MLP(void);
  
  __device__ void apply(void) const;
  __device__ inline int getNumInputs(void) const { return numInputs; }
  __device__ inline int getNumOutputs(void) const { return numOutputs; }

  __device__ double* getWeights(void) const { return weights; };
  __device__ void setWeights(double a, double b, double c, double d, double e, double f, double g, double h, double i, double j, double k, double l, double m, double n, double o, double p, double q, double r, double s, double t, double u, double v, double w, double x, double y, double z, double aa, double ab, double ac, double ad, double ae, double af, double ag, double ah, double ai, double aj, double ak, double al, double am, double an, double ao, double ap, double aq, double ar, double as, double at, double au, double av, double aw, double ax, double ay, double az, double ba, double bb, double bc, double bd, double be, double bf, double bg, double bh, double bi, double bj, double bk, double bl, double bm, double bn, double bo, double bp, double bq, double br, double bs, double bt, double bu, double bv, double bw, double bx, double by, double bz, double ca, double cb, double cc, double cd, double ce, double cf, double cg, double ch, double ci, double cj, double ck, double cl, double cm, double cn, double co, double cp, double cq, double cr, double cs, double ct, double cu, double cv, double cw, double cx, double cy, double cz);
  
  double* inputs;
  double* outputs;

private:
  int numInputs;
  int numOutputs;
  double* weights;
};

class VX3_DistributedNeuralController
{
public:
  __device__ VX3_DistributedNeuralController(VX3_VoxelyzeKernel* kernel, double a, double b, double c, double d, double e, double f, double g, double h, double i, double j, double k, double l, double m, double n, double o, double p, double q, double r, double s, double t, double u, double v, double w, double x, double y, double z, double aa, double ab, double ac, double ad, double ae, double af, double ag, double ah, double ai, double aj, double ak, double al, double am, double an, double ao, double ap, double aq, double ar, double as, double at, double au, double av, double aw, double ax, double ay, double az, double ba, double bb, double bc, double bd, double be, double bf, double bg, double bh, double bi, double bj, double bk, double bl, double bm, double bn, double bo, double bp, double bq, double br, double bs, double bt, double bu, double bv, double bw, double bx, double by, double bz, double ca, double cb, double cc, double cd, double ce, double cf, double cg, double ch, double ci, double cj, double ck, double cl, double cm, double cn, double co, double cp, double cq, double cr, double cs, double ct, double cu, double cv, double cw, double cx, double cy, double cz);
  __device__ ~VX3_DistributedNeuralController(void);
  
  __device__ double updateVoxelTemp(VX3_Voxel* voxel, VX3_VoxelyzeKernel* kernel);
  __device__ void updateLastSignals(VX3_VoxelyzeKernel* kernel);
  __device__ void getLastSignals(VX3_Voxel* voxel) const;
  __device__ void sense(VX3_Voxel* voxel, VX3_VoxelyzeKernel* kernel);
  __device__ void vote(void) const;
  
  __device__ VX3_Vec3D<float>* getOffset(const linkDirection dir) const;

  VX3_MLP* mlp;
  VX3_dVector<int>* votes;
  VX3_dVector<double>* tempVotes;
  bool firstRightContact;
  bool firstLeftContact;
};

#endif //VX3_DISTRIBUTED_H
