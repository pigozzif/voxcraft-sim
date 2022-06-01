#include "VX_Distributed.h"
#include "Vec3D.h"
#include "VX_Object.h"
#include "VX_Voxel.h"
#include "Voxelyze.h"
#include "VX_Collision.h"
#include <algorithm>
#include <stdlib.h>
#include <cstdlib>
#include <vector>

CVX_Distributed::CVX_Distributed(const int numVoxels, const std::string weights, CVoxelyze* sim)
{
  this->numVoxels = numVoxels;
  mlp = new CVX_MLP(NUM_SENSORS + NUM_SIGNALS, NUM_SIGNALS + 2, weights);
  lastSignals = (double**) malloc(sizeof(double*) * numVoxels);
  currSignals = (double**) malloc(sizeof(double*) * numVoxels);
  for (int i = 0; i < numVoxels; ++i) {
    lastSignals[i] = (double*) malloc(sizeof(double) * NUM_SIGNALS);
    currSignals[i] = (double*) malloc(sizeof(double) * NUM_SIGNALS);
    //std::fill(lastSignals[i], lastSignals[i] + 4, 0.0);
    for (int j = 0; j < NUM_SIGNALS; ++j) {
      lastSignals[i][j] = 1;
    }
    std::fill(currSignals[i], currSignals[i] + NUM_SIGNALS, 0.0);
  }
  this->sim = sim;
}

CVX_Distributed::~CVX_Distributed(void)
{
  for (int i = 0; i < numVoxels; ++i) {
    free(lastSignals[i]);
    free(currSignals[i]);
  }
  free(lastSignals);
  free(currSignals);
  delete mlp;
  sim = NULL;
}

double CVX_Distributed::UpdateVoxelTemp(CVX_Object* pObj, CVX_Voxel* voxel)
{
  //for (int i = 0; i < (int)pObjUpdate->GetNumMaterials(); ++i) {
    double* sensors = (double*) malloc(sizeof(double) * NUM_SENSORS);
  std::fill(sensors, sensors + NUM_SENSORS, -1.0);
  sense(voxel, sensors);
  
    double* signals = GetLastSignals(voxel, pObj);
    double* inputs = new double[mlp->getNumInputs()];
    std::copy(sensors, sensors + NUM_SENSORS, inputs);
    std::copy(signals, signals + NUM_SIGNALS, inputs + NUM_SENSORS);
    double* outputs = mlp->Apply(inputs);
    //pObjUpdate->GetBaseMat(i)->SetCurMatTemp(TempBase + outputs[0]);
    std::cout << "(" << voxel->pos.x << "," << voxel->pos.y << "," << voxel->pos.z << ") ";
    for (int k = 0; k < NUM_SIGNALS; ++k) {
      std::cout << sensors[k] << " ";
    }
    std::cout << std::endl;
    std::copy(outputs + 2, outputs + mlp->getNumOutputs(), currSignals[pObj->GetIndex(voxel->ix, voxel->iy, voxel->iz)]);
  //}
  
  double actuation = outputs[0];
  free(sensors);
  free(signals);
  free(inputs);
  free(outputs);
  return actuation;
}

void CVX_Distributed::UpdateLastSignals(void)
{
  for (int i = 0; i < numVoxels; ++i) {
    std::copy(currSignals[i], currSignals[i] + NUM_SIGNALS, lastSignals[i]);
  }
}

double* CVX_Distributed::GetLastSignals(CVX_Voxel* voxel, CVX_Object* pObj) const
{
  double* signals = (double*) malloc(sizeof(double) * NUM_SIGNALS);
  //Vec3D<>* currPoint = new Vec3D<>(voxel->ix, voxel->iy, voxel->iz);
  //pObjUpdate->GetXYZ(&currPoint, i);
  for (int dir = 0; dir < NUM_SIGNALS; ++dir) {
    CVX_Voxel* adjVoxel = voxel->adjacentVoxel((CVX_Voxel::linkDirection)dir); 
    signals[dir] = (adjVoxel) ? lastSignals[pObj->GetIndex(adjVoxel->ix, adjVoxel->iy, adjVoxel->iz)][dir] : 0.0;
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

void CVX_Distributed::sense(CVX_Voxel* voxel, double* sensors) const
{
  std::vector<CVX_Collision*> collisions = std::vector<CVX_Collision*>();
  for (CVX_Collision* collision : sim->collisionsList) {
    if (collision->voxel1() == voxel || collision->voxel2() == voxel) {
      collisions.push_back(collision);
    }
  }
  
  for (CVX_Collision* collision : collisions) {
    if (collision->getForce() == Vec3D<float>(0,0,0)) {
      continue;
    }
    for (int i = 0; i < NUM_SENSORS; ++i) {
      Vec3D<float>* offset = getOffset((CVX_Voxel::linkDirection)i);
      double s = voxel->material()->nominalSize();
      CVX_Voxel* other = (collision->voxel1() == voxel) ? collision->voxel2() : collision->voxel1();
      if (Vec3D<float>(other->pos.x / s + offset->x, other->pos.y / s + offset->y, other->pos.z / s + offset->z) == 
          Vec3D<float>(voxel->pos.x / s + offset->x, voxel->pos.y / s + offset->y, voxel->pos.z / s + offset->z)) {
        sensors[i] = 1.0;//touchSensor->sense(voxel, sim->voxel(voxel->pos.x / s + offset->x, voxel->pos.y / s + offset->y, voxel->pos.z / s + offset->z), (CVX_Voxel::linkDirection)i);//voxel->temp;//pObjUpdate->GetBaseMat(i)->GetCurMatTemp();
      }
    }
  }
  
  if (voxel->iz == 0) {
    sensors[5] = (voxel->floorPenetration() >= 0) ? 1.0 : -1.0;
  }
}

Vec3D<float>* CVX_Distributed::getOffset(CVX_Voxel::linkDirection dir) const
{
  switch (dir) {
    case 0:
      return new Vec3D<float>(1,0,0);
    case 1:
      return new Vec3D<float>(-1,0,0);
    case 2:
      return new Vec3D<float>(0,1,0);
    case 3:
      return new Vec3D<float>(0,-1,0);
    case 4:
      return new Vec3D<float>(0,0,1);
    default:
      return new Vec3D<float>(0,0,-1);
  }
}
