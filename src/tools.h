#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class Tools
{
public:
  /**
  * Constructor.
  */
  Tools();

  /**
  * Destructor.
  */
  virtual ~Tools();

  /**
  * A helper method to calculate RMSE.
  */
  VectorXd CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);

  /**
  * Generate sigma points weights
  */
  VectorXd GenerateWeights(int n_aug, int lambda);

  MatrixXd GenerateAugmentedSigmaPoints(VectorXd x, MatrixXd P, MatrixXd Q, int n_x, int n_aug, int lambda);

  /**
  * Normalize angle
  */
  double NormalizeAngle(double angle);
};

#endif /* TOOLS_H_ */