#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{
  is_initialized_ = false;

  previous_timestamp_ = 0;

  //set state dimension
  n_x_ = 5;

  //set measurement dimension, radar can measure r, phi, and r_dot
  n_z_radar_ = 3;

  //set measurement dimension, laser can measure px, py
  n_z_laser_ = 2;

  // Augmented state dimension
  n_aug_ = 7;

  // define spreading parameter
  lambda_ = 3 - n_x_;

  // Number of sigma points
  n_sigma_ = 2 * n_aug_ + 1;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector: [px, py, v, yaw, yawd]
  x_ = VectorXd(n_x_);
  x_ << 0, 0, 0, 0, 0;

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  P_ <<
    .15,  0,   0,   0,   0,
    0,  .15,   0,   0,   0,
    0,    0,   1,   0,   0,
    0,    0,   0,   1,   0,
    0,    0,   0,   0,   1;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.3;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  Q_ = MatrixXd(2, 2);
  Q_ << std_a_ * std_a_, 0,
      0, std_yawdd_ * std_yawdd_;

  R_radar_ = MatrixXd(n_z_radar_, n_z_radar_);
  R_radar_ << std_radr_ * std_radr_, 0, 0,
      0, std_radphi_ * std_radphi_, 0,
      0, 0, std_radrd_ * std_radrd_;

  R_laser_ = MatrixXd(n_z_laser_, n_z_laser_);
  R_laser_ << std_laspx_ * std_laspx_, 0,
      0, std_laspy_ * std_laspy_;

  weights_ = tools_.Tools::GenerateWeights(n_aug_, lambda_);
}

UKF::~UKF() {}

MatrixXd UKF::GenerateAugmentedSigmaPoints(VectorXd x, MatrixXd P, MatrixXd Q, int n_x, int n_aug, int lambda)
{
  int n_sigma = 2 * n_aug + 1;

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug, n_sigma);
  Xsig_aug.fill(0.0);

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug);
  x_aug.fill(0.0);
  x_aug.head(n_x) = x;
  x_aug(n_x) = 0;
  x_aug(n_x + 1) = 0;

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug, n_aug);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x, n_x) = P;
  P_aug.bottomRightCorner(2, 2) = Q;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug; i++)
  {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda + n_aug) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug) = x_aug - sqrt(lambda + n_aug) * L.col(i);
  }

  return Xsig_aug;
}

MatrixXd UKF::GeneratePredictedSigmaPoints(MatrixXd Xsig_aug, double delta_t, int n_x, int n_aug, int lambda)
{
  int n_sigma = 2 * n_aug + 1;

  MatrixXd Xsig_pred = MatrixXd(n_x, n_sigma);
  Xsig_pred.fill(0.0);

  double dt = delta_t;
  double dt2 = dt * dt;

  for (int column = 0; column < n_sigma; column++)
  {
    double v = Xsig_aug(2, column);
    double yaw = Xsig_aug(3, column);
    double yawd = Xsig_aug(4, column);
    double nu_a = Xsig_aug(5, column);
    double nu_yawdd = Xsig_aug(6, column);

    VectorXd x_k = Xsig_aug.col(column).head(n_x);

    VectorXd delta_x = VectorXd(n_x);
    delta_x.fill(0.0);

    VectorXd noise = VectorXd(n_x);
    noise << 0.5 * dt2 * nu_a * cos(yaw),
        0.5 * dt2 * nu_a * sin(yaw),
        dt * nu_a,
        0.5 * dt2 * nu_yawdd,
        dt * nu_yawdd;

    double delta_px;
    double delta_py;

    if (fabs(yawd) < 0.001)
    {
      delta_px = v * dt * cos(yaw);
      delta_py = v * dt * sin(yaw);
    }
    else
    {
      delta_px = v / yawd * (sin(yaw + yawd * dt) - sin(yaw));
      delta_py = v / yawd * (cos(yaw) - cos(yaw + yawd * dt));
    }

    delta_x << delta_px, delta_py, 0, yawd * dt, 0;

    Xsig_pred.col(column) = x_k + delta_x + noise;
  }

  return Xsig_pred;
}

MatrixXd UKF::PredictedMeanAndCovarianceFromSigmaPoints(MatrixXd Xsig_pred, VectorXd weights, int n_x, int n_aug, int lambda)
{
  int n_sigma = 2 * n_aug + 1;

  VectorXd x = VectorXd(n_x);
  x.fill(0.0);
  for (int i = 0; i < n_sigma; i++)
  {
    x = x + weights(i) * Xsig_pred.col(i);
  }

  //predicted state covariance matrix
  MatrixXd P = MatrixXd(n_x, n_x);
  P.fill(0.0);
  for (int i = 0; i < n_sigma; i++)
  {
    // state difference
    VectorXd x_diff = Xsig_pred.col(i) - x;
    x_diff(3) = tools_.Tools::NormalizeAngle(x_diff(3));
    P = P + weights_(i) * x_diff * x_diff.transpose();
  }

  MatrixXd mean_and_covariance = MatrixXd(n_x, n_x + 1);
  mean_and_covariance.col(0) = x;
  mean_and_covariance.rightCols(n_x) = P;
  return mean_and_covariance;
}

MatrixXd UKF::RadarMeasurementSigmaPointsForPredictedSigmaPoints(MatrixXd Xsig_pred, int n_z, int n_aug)
{
  int n_sigma = 2 * n_aug + 1;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sigma);
  Zsig.fill(0.0);

  //transform sigma points into measurement space
  for (int i = 0; i < n_sigma; i++)
  {
    double p_x = Xsig_pred(0, i);
    double p_y = Xsig_pred(1, i);
    double v = Xsig_pred(2, i);
    double yaw = Xsig_pred(3, i);

    double v1 = cos(yaw) * v;
    double v2 = sin(yaw) * v;

    // measurement model
    Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);                         //r
    Zsig(1, i) = atan2(p_y, p_x);                                     //phi
    Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); //r_dot
  }

  return Zsig;
}

MatrixXd UKF::PredictedMeasurementAndCovarianceForMeasurementSigmaPoints(MatrixXd Zsig, VectorXd weights, MatrixXd R, int n_z, int n_aug)
{
  Tools tools;
  int n_sigma = 2 * n_aug + 1;

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i = 0; i < n_sigma; i++)
  {
    z_pred = z_pred + weights(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for (int i = 0; i < n_sigma; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    z_diff(1) = tools.Tools::NormalizeAngle(z_diff(1));
    S = S + weights(i) * z_diff * z_diff.transpose();
  }

  S = S + R;

  MatrixXd mean_and_covariance = MatrixXd(n_z, n_z + 1);
  mean_and_covariance.fill(0.0);
  mean_and_covariance.col(0) = z_pred;
  mean_and_covariance.rightCols(n_z) = S;
  return mean_and_covariance;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
  if (!is_initialized_)
  {
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      double rhod = meas_package.raw_measurements_[2];
      x_ << rho * cos(phi), rho * sin(phi), 5, phi, 1;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      double px = meas_package.raw_measurements_[0];
      double py = meas_package.raw_measurements_[1];
      x_ << px, py, 1, 1, 0.1;
    }
    previous_timestamp_ = meas_package.timestamp_;
    is_initialized_ = true;
  }
  else
  {
    if (use_radar_ == true && meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      double dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
      previous_timestamp_ = meas_package.timestamp_;
      Predict(dt);
      UpdateRadar(meas_package);
    }
    else if (use_laser_ == true && meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      double dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
      previous_timestamp_ = meas_package.timestamp_;
      Predict(dt);
      UpdateLidar(meas_package);
    }
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Predict(double delta_t)
{
  MatrixXd Xsig_aug = GenerateAugmentedSigmaPoints(x_, P_, Q_, n_x_, n_aug_, lambda_);
  Xsig_pred_ = GeneratePredictedSigmaPoints(Xsig_aug, delta_t, n_x_, n_aug_, lambda_);
  MatrixXd x_and_P = PredictedMeanAndCovarianceFromSigmaPoints(Xsig_pred_, weights_, n_x_, n_aug_, lambda_);
  x_ = x_and_P.leftCols(1);
  P_ = x_and_P.rightCols(n_x_);
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package)
{
  VectorXd z = meas_package.raw_measurements_;
  MatrixXd I = MatrixXd::Identity(n_x_, n_x_);

  MatrixXd H = MatrixXd(2, 5);
  H <<
    1, 0, 0, 0, 0,
    0, 1, 0, 0, 0;

  VectorXd z_pred = H * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H.transpose();
  MatrixXd S = H * P_ * Ht + R_laser_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  x_ = x_ + (K * y);
  P_ = (I - K * H) * P_;

  x_(3) = tools_.Tools::NormalizeAngle(x_(3));
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package)
{
  Zsig_ = RadarMeasurementSigmaPointsForPredictedSigmaPoints(Xsig_pred_, n_z_radar_, n_aug_);
  MatrixXd z_and_S = PredictedMeasurementAndCovarianceForMeasurementSigmaPoints(Zsig_, weights_, R_radar_, n_z_radar_, n_aug_);
  z_pred_ = z_and_S.leftCols(1);
  S_ = z_and_S.rightCols(n_z_radar_);

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z_radar_);
  Tc.fill(0);

  UpdateBelief(meas_package, Tc);
}

void UKF::UpdateBelief(MeasurementPackage meas_package, MatrixXd Tc)
{
  VectorXd z = meas_package.raw_measurements_;

  //calculate cross correlation matrix
  for (int i = 0; i < n_sigma_; i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    x_diff(3) = tools_.Tools::NormalizeAngle(x_diff(3));
    VectorXd z_diff = Zsig_.col(i) - z_pred_;
    z_diff(1) = tools_.Tools::NormalizeAngle(z_diff(1));
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K = Tc * S_.inverse();

  VectorXd z_diff = z - z_pred_;
  z_diff(1) = tools_.Tools::NormalizeAngle(z_diff(1));

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S_ * K.transpose();

  nis_epsilon_ = z_diff.transpose() * S_.inverse() * z_diff;
}
