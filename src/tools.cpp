#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::GenerateWeights(int n_aug, int lambda)
{
  int n_sigma = 2 * n_aug + 1;
  VectorXd weights = VectorXd(n_sigma);
  double weight_0 = lambda / (lambda + n_aug);
  weights(0) = weight_0;
  for (int i = 1; i < n_sigma; i++)
  {
    double weight = 0.5 / (n_aug + lambda);
    weights(i) = weight;
  }
  return weights;
}

double Tools::NormalizeAngle(double angle)
{
//    std::cout << "angle is" << angle << std::endl;
  while (angle >=  M_PI) angle -= 2. * M_PI;
  while (angle <= -M_PI) angle += 2. * M_PI;
//    std::cout << "returned adjusted..." << angle << std::endl;
  return angle;
}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size() || estimations.size() == 0)
  {
    std::cout << "Invalid estimation or ground_truth data" << std::endl;
    return rmse;
  }

  // accumulate squared residuals
  for (unsigned int i = 0; i < estimations.size(); ++i)
  {
    VectorXd residual = estimations[i] - ground_truth[i];

    // coefficient-wise multiplication
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  // calculate the mean
  rmse = rmse / estimations.size();

  // calculate the squared root
  rmse = rmse.array().sqrt();

  // return the result
  return rmse;
}
