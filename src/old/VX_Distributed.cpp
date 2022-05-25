#include "VX_Distributed.h"
#include "Vec3D.h"
#include "VX_Object.h"
#include <algorithm>
#include <stdlib.h>

CVX_Distributed::CVX_Distributed(const int numInputs, const int numMaterials, const std::string weights)
{
  this->numMaterials = numMaterials;
  mlp = new CVX_MLP(numInputs, 6, weights);
  lastSignals = (double**) malloc(sizeof(double*) * numMaterials);
  currSignals = (double**) malloc(sizeof(double*) * numMaterials);
  for (int i = 0; i < numMaterials; ++i) {
    lastSignals[i] = (double*) malloc(sizeof(double) * 4);
    currSignals[i] = (double*) malloc(sizeof(double) * 4);
    std::fill(lastSignals[i], lastSignals[i] + 4, 0.0);
    std::fill(currSignals[i], currSignals[i] + 4, 0.0);
  }
}

CVX_Distributed::~CVX_Distributed(void)
{
  for (int i = 0; i < numMaterials; ++i) {
    delete lastSignals[i];
    delete currSignals[i];
  }
  delete[] lastSignals;
  delete[] currSignals;
  delete mlp;
}

void CVX_Distributed::UpdateMatTemp(CVX_Object* pObjUpdate, double TempBase)
{
  for (int i = 0; i < (int)pObjUpdate->GetNumMaterials(); ++i) {
    double* sensors = (double*) malloc(sizeof(double));
    sensors[0] = pObjUpdate->GetBaseMat(i)->GetCurMatTemp();
    double* signals = GetLastSignals(i, pObjUpdate);
    double* inputs = new double[mlp->getNumInputs()];
    std::copy(sensors, sensors + mlp->getNumInputs() - 4, inputs);
    std::copy(signals, signals + 4, inputs + mlp->getNumInputs() - 4);
    double* outputs = mlp->Apply(inputs);
    pObjUpdate->GetBaseMat(i)->SetCurMatTemp(TempBase + outputs[0]);
    std::copy(outputs + 2, outputs + mlp->getNumOutputs(), currSignals[i]);
  }
  for (int i = 0; i < (int)pObjUpdate->GetNumMaterials(); ++i) {
    std::copy(currSignals[i], currSignals[i] + 4, lastSignals[i]);
  }
}

double* CVX_Distributed::GetLastSignals(int i, CVX_Object* pObjUpdate) const
{
  double* signals = (double*) malloc(sizeof(double) * 4);
  Vec3D<>* currPoint;
  pObjUpdate->GetXYZ(currPoint, i);
  /*int idx = pObjUpdate->GetIndex(currPoint->x + 1, currPoint->y, currPoint->z);
  if (idx != -1 && pObjUpdate->Structure.GetData(idx) != 0) {
    signals[0] = lastSignals[idx][0];
  }
  else {
    signals[0] = 0.0;
  }
  idx = pObjUpdate->GetIndex(currPoint->x - 1, currPoint->y, currPoint->z);
  if (idx != -1 && pObjUpdate->Structure.GetData(idx) != 0) {
    signals[1] = lastSignals[idx][1];
  }
  else {
    signals[1] = 0.0;
  }
  idx = pObjUpdate->GetIndex(currPoint->x, currPoint->y + 1, currPoint->z);
  if (idx != -1 && pObjUpdate->Structure.GetData(idx) != 0) {
    signals[2] = lastSignals[idx][2];
  }
  else {
    signals[2] = 0.0;
  }
  idx = pObjUpdate->GetIndex(currPoint->x, currPoint->y - 1, currPoint->z);
  if (idx != -1 && pObjUpdate->Structure.GetData(idx) != 0) {
    signals[3] = lastSignals[idx][3];
  }
  else {
    signals[3] = 0.0;
  }
  return signals;*/
  return lastSignals[i];
}
