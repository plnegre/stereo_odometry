#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <ros/ros.h>
#include <boost/thread.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <tf/transform_datatypes.h>

using namespace std;
using namespace boost;

namespace odom
{

class Optimizer
{

public:

  /** \brief Hash class constructor
    */
  Optimizer();

  void setParameters(const cv::Mat camera_matrix, const cv::Mat dist_coef, const double baseline);

  /** \brief Pose optimization
   * \param keypoints of frame A
   * \param world points of frame B
   * \param Camera matrix
   * \param Distortion coefficients
   */
  void poseOptimization(const vector<cv::Point2d> kps_l, const vector<cv::Point2d> kps_r, const vector<cv::Point3d> wps, tf::Transform& pose);

  struct SnavelyReprojectionError {
    SnavelyReprojectionError(double l_observed_x, double l_observed_y, double r_observed_x, double r_observed_y, double* camera_matrix, double* dist_coef)
        : l_observed_x_(l_observed_x), l_observed_y_(l_observed_y), r_observed_x_(r_observed_x), r_observed_y_(r_observed_y), camera_matrix_(camera_matrix), dist_coef_(dist_coef){}

    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const {

      // camera[0,1,2] are the angle-axis rotation.
      T p[3];
      ceres::AngleAxisRotatePoint(camera, point, p);

      // camera[3,4,5] are the translation.
      p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];

      // Left and right point
      T p_l[3];
      T p_r[3];
      p_l[0] = p[0]; p_l[1] = p[1]; p_l[2] = p[2];
      p_r[0] = p[0] - T(camera_matrix_[3]); p_r[1] = p[1]; p_r[2] = p[2];

      // Compute the center of distortion.
      T xp_l = p_l[0] / p_l[2];
      T yp_l = p_l[1] / p_l[2];

      T xp_r = p_r[0] / p_r[2];
      T yp_r = p_r[1] / p_r[2];

      // Apply radial distortion.
      T r2_l = xp_l*xp_l + yp_l*yp_l;
      T r2_r = xp_r*xp_r + yp_r*yp_r;

      T dist_l = (T(1.0) + T(dist_coef_[0])*r2_l + T(dist_coef_[1])*r2_l*r2_l);
      T dist_r = (T(1.0) + T(dist_coef_[0])*r2_r + T(dist_coef_[1])*r2_r*r2_r);

      T xpp_l = xp_l*dist_l + T(2.0)*T(dist_coef_[2])*xp_l*yp_l + T(dist_coef_[3])*(r2_l + T(2.0)*xp_l*xp_l);
      T ypp_l = yp_l*dist_l + T(dist_coef_[2])*(r2_l + T(2.0)*yp_l*yp_l) + T(2.0)*T(dist_coef_[3])*xp_l*yp_l;

      T xpp_r = xp_r*dist_r + T(2.0)*T(dist_coef_[2])*xp_r*yp_r + T(dist_coef_[3])*(r2_r + T(2.0)*xp_r*xp_r);
      T ypp_r = yp_r*dist_r + T(dist_coef_[2])*(r2_r + T(2.0)*yp_r*yp_r) + T(2.0)*T(dist_coef_[3])*xp_r*yp_r;

      // Compute final projected point position.
      T l_predicted_x = T(camera_matrix_[2]) * xpp_l + T(camera_matrix_[0]);
      T l_predicted_y = T(camera_matrix_[2]) * ypp_l + T(camera_matrix_[1]);
      T r_predicted_x = T(camera_matrix_[2]) * xpp_r + T(camera_matrix_[0]);
      T r_predicted_y = T(camera_matrix_[2]) * ypp_r + T(camera_matrix_[1]);

      // The error is the difference between the predicted and observed position.
      residuals[0] = l_predicted_x - T(l_observed_x_);
      residuals[1] = l_predicted_y - T(l_observed_y_);
      residuals[2] = r_predicted_x - T(r_observed_x_);
      residuals[3] = r_predicted_y - T(r_observed_y_);

      // TODO: add the stereo epipolar constraint as a residual
      return true;
    }

     // Factory to hide the construction of the CostFunction object from
     // the client code.
     static ceres::CostFunction* Create(const double l_observed_x,
                                        const double l_observed_y,
                                        const double r_observed_x,
                                        const double r_observed_y,
                                        double* camera_matrix,
                                        double* dist_coef) {
       return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 4, 6, 3>(
                   new SnavelyReprojectionError(l_observed_x, l_observed_y, r_observed_x, r_observed_y, camera_matrix, dist_coef)));
     }

    double l_observed_x_;
    double l_observed_y_;
    double r_observed_x_;
    double r_observed_y_;
    double* camera_matrix_;
    double* dist_coef_;
  };

private:

  double* camera_matrix_; //!> Camera parameters vector [cx, cy, fx, baseline]
  double* dist_coef_; //!> Camera parameters vector [k1, k2, p1, p2]

};

} // namespace

#endif // OPTIMIZER_H