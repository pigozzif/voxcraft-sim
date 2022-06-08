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
#include <math.h>
#include <string>
#include <iostream>

CVX_MLP::CVX_MLP(const int numInputs, const int numOutputs, const std::string weights)
{
  this->numInputs = numInputs;
  this->numOutputs = numOutputs;
  setWeights(weights);
}

CVX_MLP::~CVX_MLP(void)
{
  for (int i = 0; i < numOutputs; ++i) {
    free(weights[i]);
  }
  free(weights);
}

void CVX_MLP::setWeights(const std::string weights)
{
  this->weights = (double**) malloc(sizeof(double*) * numOutputs);
  for (int i = 0; i < numOutputs; ++i) {
    this->weights[i] = (double*) malloc(sizeof(double) * (numInputs + 1));
  }
  std::string delim = ",";
  std::size_t start = 0U;
  std::size_t end = weights.find(delim);
  int i = 0;
  int j = 0;
  while (end != std::string::npos) {
    this->weights[i][j++] = atof(weights.substr(start, end - start).c_str());
    if (j >= numInputs + 1) {
      j = 0;
      ++i;
    }
    start = end + delim.length();
    end = weights.find(delim, start);
  }
}

double* CVX_MLP::apply(double* inputs) const
{
  //apply input activation
  for (int i = 0; i < numInputs; ++i)
  {
    inputs[i] = tanh(inputs[i]);
  }
  double* outputs = (double*) malloc(sizeof(double) * numOutputs);
  for (int j = 0; j < numOutputs; ++j)
  {
    double sum = weights[j][0]; //the bias
    for (int k = 1; k < numInputs + 1; ++k)
    {
      sum += inputs[k - 1] * weights[j][k]; //weight inputs
    }
    outputs[j] = tanh(sum); //apply output activation
  }
  return outputs;
}

CVX_Distributed::CVX_Distributed(const int numVoxels, const std::string weights, CVoxelyze* sim)
{
  this->numVoxels = numVoxels;
  mlp = new CVX_MLP(NUM_SENSORS + NUM_SIGNALS, NUM_SIGNALS + 2, weights);
  lastSignals = (double**) malloc(sizeof(double*) * numVoxels);
  currSignals = (double**) malloc(sizeof(double*) * numVoxels);
  for (int i = 0; i < numVoxels; ++i) {
    lastSignals[i] = (double*) malloc(sizeof(double) * NUM_SIGNALS);
    currSignals[i] = (double*) malloc(sizeof(double) * NUM_SIGNALS);
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

double CVX_Distributed::updateVoxelTemp(CVX_Object* pObj, CVX_Voxel* voxel)
{
  std::cout << "start updating" << std::endl;
  double* sensors = (double*) malloc(sizeof(double) * NUM_SENSORS);
  std::fill(sensors, sensors + NUM_SENSORS, -1.0);
  sense(voxel, sensors);
  
  double* signals = getLastSignals(voxel, pObj);
  double* inputs = new double[mlp->getNumInputs()];
  std::copy(sensors, sensors + NUM_SENSORS, inputs);
  std::copy(signals, signals + NUM_SIGNALS, inputs + NUM_SENSORS);
  double* outputs = mlp->apply(inputs);
  
  double actuation = outputs[0];
  free(sensors);
  free(signals);
  free(inputs);
  free(outputs);
  std::cout << "after updating" << std::endl;
  return actuation;
}

void CVX_Distributed::updateLastSignals(void)
{
  for (int i = 0; i < numVoxels; ++i) {
    std::copy(currSignals[i], currSignals[i] + NUM_SIGNALS, lastSignals[i]);
  }
}

double* CVX_Distributed::getLastSignals(CVX_Voxel* voxel, CVX_Object* pObj) const
{
  double* signals = (double*) malloc(sizeof(double) * NUM_SIGNALS);
  for (int dir = 0; dir < NUM_SIGNALS; ++dir) {
    CVX_Voxel* adjVoxel = voxel->adjacentVoxel((CVX_Voxel::linkDirection)dir); 
    signals[dir] = (adjVoxel) ? lastSignals[pObj->GetIndex(adjVoxel->ix, adjVoxel->iy, adjVoxel->iz)][dir] : 0.0;
  }
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
        sensors[i] = 1.0;
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
