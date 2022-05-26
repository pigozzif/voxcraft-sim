#include "VX_Distributed.h"
#include "Vec3D.h"
#include "VX_Object.h"
#include "VX_Voxel.h"
#include <algorithm>
#include <stdlib.h>
#include <cstdlib>

CVX_Distributed::CVX_Distributed(const int numInputs, const int numVoxels, const std::string weights)
{
  this->numVoxels = numVoxels;
  mlp = new CVX_MLP(numInputs, 6, weights);
  lastSignals = (double**) malloc(sizeof(double*) * numVoxels);
  currSignals = (double**) malloc(sizeof(double*) * numVoxels);
  for (int i = 0; i < numVoxels; ++i) {
    lastSignals[i] = (double*) malloc(sizeof(double) * 4);
    currSignals[i] = (double*) malloc(sizeof(double) * 4);
    //std::fill(lastSignals[i], lastSignals[i] + 4, 0.0);
    for (int j = 0; j < 4; ++j) {
      lastSignals[i][j] = rand() % RAND_MAX;
    }
    std::fill(currSignals[i], currSignals[i] + 4, 0.0);
  }
}

CVX_Distributed::~CVX_Distributed(void)
{
  for (int i = 0; i < numVoxels; ++i) {
    delete lastSignals[i];
    delete currSignals[i];
  }
  delete[] lastSignals;
  delete[] currSignals;
  delete mlp;
}

double CVX_Distributed::UpdateVoxelTemp(CVX_Object* pObj, CVX_Voxel* voxel)
{
  //for (int i = 0; i < (int)pObjUpdate->GetNumMaterials(); ++i) {
    double* sensors = (double*) malloc(sizeof(double));
    sensors[0] = voxel->temp;//pObjUpdate->GetBaseMat(i)->GetCurMatTemp();
    double* signals = GetLastSignals(voxel, pObj);
    double* inputs = new double[mlp->getNumInputs()];
    std::copy(sensors, sensors + mlp->getNumInputs() - 4, inputs);
    std::copy(signals, signals + 4, inputs + mlp->getNumInputs() - 4);
    double* outputs = mlp->Apply(inputs);
    //pObjUpdate->GetBaseMat(i)->SetCurMatTemp(TempBase + outputs[0]);
    std::cout << i << ": ";
    for (int k = 0; k < 6; ++k) {
      std::cout << outputs[k] << " ";
    }
    std::cout << std::endl;
    std::copy(outputs + 2, outputs + mlp->getNumOutputs(), currSignals[i]);
  //}
  return outputs[0];
}

void CVX_Distributed::UpdateLastSignals(void)
{
  for (int i = 0; i < numVoxels; ++i) {
    std::copy(currSignals[i], currSignals[i] + 4, lastSignals[i]);
  }
}

double* CVX_Distributed::GetLastSignals(CVX_Voxel* voxel, CVX_Object* pObj) const
{
  double* signals = (double*) malloc(sizeof(double) * 4);
  //Vec3D<>* currPoint = new Vec3D<>(voxel->ix, voxel->iy, voxel->iz);
  //pObjUpdate->GetXYZ(&currPoint, i);
  for (int dir = 0; dir < 4; ++dir) {
    CVX_Voxel* adjVoxel = voxel->adjacentVoxel(dir);
    signals[dir] = lastSignals[pObj->GetIndex(adjVoxel->ix, adjVoxel->iy, adjVoxel->iz)];
  }
  /*int idx = pObjUpdate->GetIndex(currPoint.x + 1, currPoint.y, currPoint.z);
  if (idx != -1 && pObjUpdate->Structure.GetData(idx) != 0) {
    signals[0] = lastSignals[idx][0];
  }
  else {
    signals[0] = 0.0;
  }
  idx = pObjUpdate->GetIndex(currPoint.x - 1, currPoint.y, currPoint.z);
  if (idx != -1 && pObjUpdate->Structure.GetData(idx) != 0) {
    signals[1] = lastSignals[idx][1];
  }
  else {
    signals[1] = 0.0;
  }
  idx = pObjUpdate->GetIndex(currPoint.x, currPoint.y + 1, currPoint.z);
  if (idx != -1 && pObjUpdate->Structure.GetData(idx) != 0) {
    signals[2] = lastSignals[idx][2];
  }
  else {
    signals[2] = 0.0;
  }
  idx = pObjUpdate->GetIndex(currPoint.x, currPoint.y - 1, currPoint.z);
  if (idx != -1 && pObjUpdate->Structure.GetData(idx) != 0) {
    signals[3] = lastSignals[idx][3];
  }
  else {
    signals[3] = 0.0;
  }*/
  return signals;
}
