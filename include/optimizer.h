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

  /** \brief Pose optimization
   * \param keypoints of frame A
   * \param world points of frame B
   * \param Camera matrix
   * \param Distortion coefficients
   */
  void poseOptimization(const vector<cv::Point2d> kps, const vector<cv::Point3d> wps, tf::Transform& pose, cv::Mat& camera_matrix, cv::Mat& dist_coef);

  struct SnavelyReprojectionError {
    SnavelyReprojectionError(double observed_x, double observed_y)
        : observed_x_(observed_x), observed_y_(observed_y) {}

    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const {

      // camera[0,1,2] are the angle-axis rotation.
      T p[3];
      ceres::AngleAxisRotatePoint(camera, point, p);

      // camera[3,4,5] are the translation.
      p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];

      // Compute the center of distortion.
      T xp = p[0] / p[2];
      T yp = p[1] / p[2];

      // Apply radial distortion. camera[9,10,11,12] are k1, k2, p1 and p2.
      const T& k1 = camera[9];
      const T& k2 = camera[10];
      const T& p1 = camera[11];
      const T& p2 = camera[12];
      T r2 = xp*xp + yp*yp;
      T dist = (T(1.0) + k1*r2 + k2*r2*r2);
      T xpp = xp*dist + T(2.0)*p1*xp*yp + p2*(r2 + T(2.0)*xp*xp);
      T ypp = yp*dist + p1*(r2 + T(2.0)*yp*yp) + T(2.0)*p2*xp*yp;

      // Compute final projected point position. camera[6,7,8] are focal, cx and cy.
      const T& focal = camera[6];
      const T& cx = camera[7];
      const T& cy = camera[8];
      T predicted_x = focal * xpp + cx;
      T predicted_y = focal * ypp + cy;

      // The error is the difference between the predicted and observed position.
      residuals[0] = predicted_x - T(observed_x_);
      residuals[1] = predicted_y - T(observed_y_);

      // TODO: add the stereo epipolar constraint as a residual
      return true;
    }

     // Factory to hide the construction of the CostFunction object from
     // the client code.
     static ceres::CostFunction* Create(const double observed_x,
                                        const double observed_y) {
       return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 13, 3>(
                   new SnavelyReprojectionError(observed_x, observed_y)));
     }

    double observed_x_;
    double observed_y_;
  };

};

} // namespace

#endif // OPTIMIZER_H