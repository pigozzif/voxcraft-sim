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
  touchSensor = new CVX_TouchSensor();
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
  std::cout << sim->voxelCount() << std::endl;
  //for (int i = 0; i < (int)pObjUpdate->GetNumMaterials(); ++i) {
    double* sensors = (double*) malloc(sizeof(double) * NUM_SENSORS);
  std::vector<CVX_Collision*> collisions = std::vector<CVX_Collision*>();
  std::cout << sim->collisionsList << std::endl;
  for (CVX_Collision* collision : sim->collisionsList) {
    if (collision->voxel1() == voxel || collision->voxel2() == voxel) {
      collisions.push_back(collision);
    }
  }
  std::cout << collisions.size() << std::endl;
  for (CVX_Collision* collision : collisions) {
    for (int i = 0; i < NUM_SENSORS; ++i) {
      Vec3D<double>* offset = touchSensor->getOffset((CVX_Voxel::linkDirection)i);
      double s = voxel->material()->nominalSize();
      sensors[i] = 0.0;//touchSensor->sense(voxel, sim->voxel(voxel->pos.x / s + offset->x, voxel->pos.y / s + offset->y, voxel->pos.z / s + offset->z), (CVX_Voxel::linkDirection)i);//voxel->temp;//pObjUpdate->GetBaseMat(i)->GetCurMatTemp();
    }
  }
  
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

CVX_TouchSensor::CVX_TouchSensor(void) {}

CVX_TouchSensor::~CVX_TouchSensor(void) {}

double CVX_TouchSensor::sense(CVX_Voxel* source, CVX_Voxel* target, CVX_Voxel::linkDirection dir) const
{
  if (!target || target->matid == 0) {
    return -1.0;
  }
  std::cout << "we are here:" << " (" << source->ix << "," << source->iy << "," << source->iz << ") " << " (" << target->ix << "," << target->iy << "," << target->iz << ") " << std::endl;
  if (target == source->adjacentVoxel(dir)) {
    return 0.0;
  }
  else if (dir == CVX_Voxel::linkDirection::Z_NEG && source->floorPenetration() >= 0) {
    return 1.0;
  }
  std::cout << "check:" << " (" << target->matid << ") " << std::endl;
  std::cout << "we are here again" << std::endl;
  linkAxis axis = CVX_Voxel::toAxis(dir);
  double baseSize = source->baseSize(axis);
  bool isPositive = CVX_Voxel::isPositive(dir);
  double sourcePos = (isPositive) ? source->pos[axis] : - source->pos[axis];
  double targetPos = (isPositive) ? - target->pos[axis] : target->pos[axis];
  double penetration = baseSize/2 - source->mat->nominalSize()/2 + sourcePos + targetPos;
  return (penetration > 0) ? 1.0 : -1.0;
}

Vec3D<double>* CVX_TouchSensor::getOffset(CVX_Voxel::linkDirection dir) const
{
  switch (dir) {
    case 0:
      return new Vec3D<double>(1,0,0);
    case 1:
      return new Vec3D<double>(-1,0,0);
    case 2:
      return new Vec3D<double>(0,1,0);
    case 3:
      return new Vec3D<double>(0,-1,0);
    case 4:
      return new Vec3D<double>(0,0,1);
    default:
      return new Vec3D<double>(0,0,-1);
  }
}
