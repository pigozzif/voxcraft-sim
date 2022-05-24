#include <VX_Distributed.h>
#include <algorithm>
#include <stdlib.h>

CVX_Distributed::CVX_Distributed(const int numInputs, const int numOutputs, const int numMaterials)
{
  this->numMaterials = numMaterials;
  this->mlp = new CVX_MLP(numInputs, numOutputs);
  lastSignals = (double**) malloc(sizeof(double*) * numMaterials);
  currSignals = (double**) malloc(sizeof(double*) * numMaterials);
  for (int i = 0; i < numMaterials; ++i)
  {
    lastSignals[i] = (double*) malloc(sizeof(double) * 4);
    currSignals[i] = (double*) malloc(sizeof(double) * 4);
    for (int j = 0; j < 4; ++j)
    {
      lastSignals[i][j] = 0.0;
      currSignals[i][j] = 0.0;
    }
  }
}

CVX_Distributed::~CVX_Distributed(void)
{
  for (int i = 0; i < numMaterials; ++i)
  {
    delete lastSignals[i];
    delete currSignals[i];
  }
  delete[] lastSignals;
  delete[] currSignals;
  delete mlp;
}

void CVX_Distributed::UpdateMatTemp(CVX_Object* pObjUpdate)
{
  for (int i = 0; i < (int)pObjUpdate->GetNumMaterials(); ++i) {
    double sensors[1] = {pObjUpdate->GetBaseMat(i)->GetCurMatTemp()};
    double* signals = lastSignals[i];
    double* inputs = new double[mlp->getNumInputs()];
    std::copy(sensors, sensors + mlp->getNumInputs() - 4, inputs);
    std::copy(signals, signals + 4, inputs + mlp->getNumInputs() - 4);
    double* outputs = mlp->Apply(inputs);
    pObjUpdate->GetBaseMat(i)->SetCurMatTemp(outputs[0] * pObjUpdate->GetBaseMat(i)->GetCurMatTemp());
    std::copy(outputs + 1, outputs + mlp->getNumOutputs(), currSignals[i]);
  }
  for (int i = 0; i < (int)pObjUpdate->GetNumMaterials(); ++i) {
    std::copy(currSignals[i], currSignals[i] + 4, lastSignals[i]);
  }
}
