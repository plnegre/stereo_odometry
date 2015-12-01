#include "optimizer.h"

namespace odom
{
  Optimizer::Optimizer()
  {
    camera_matrix_ = new double[4];
    dist_coef_ = new double[4];
  }

  void Optimizer::setParameters(const cv::Mat camera_matrix, const cv::Mat dist_coef, const double baseline)
  {
    camera_matrix_[0] = camera_matrix.at<double>(0,2);
    camera_matrix_[1] = camera_matrix.at<double>(1,2);
    camera_matrix_[2] = camera_matrix.at<double>(0,0);
    camera_matrix_[3] = baseline;

    dist_coef_[0] = dist_coef.at<double>(0);
    dist_coef_[1] = dist_coef.at<double>(1);
    dist_coef_[2] = dist_coef.at<double>(2);
    dist_coef_[3] = dist_coef.at<double>(3);
  }

  void Optimizer::poseOptimization(const vector<cv::KeyPoint> kps_l,
                                   const vector<cv::KeyPoint> kps_r,
                                   const vector<cv::Point3d> wps,
                                   tf::Transform& pose,
                                   const vector<int> mask)
  {
    ROS_ASSERT(kps_l.size() == kps_r.size());
    ROS_ASSERT(kps_l.size() == wps.size());
    if (mask.size() > 0)
      ROS_ASSERT(mask.size() == kps_l.size());

    // Ceres problem
    ceres::Problem problem;

    // Convert quaternion to angle axis
    double angle_axis[3];
    double quaternion[4] = {pose.getRotation().w(), pose.getRotation().x(), pose.getRotation().y(), pose.getRotation().z()};
    ceres::QuaternionToAngleAxis(quaternion, angle_axis);

    // Initial camera parameters
    double* camera_params = new double[6];
    camera_params[0] = angle_axis[0];
    camera_params[1] = angle_axis[1];
    camera_params[2] = angle_axis[2];
    camera_params[3] = pose.getOrigin().x();
    camera_params[4] = pose.getOrigin().y();
    camera_params[5] = pose.getOrigin().z();

    // Prepare the problem
    for (uint i=0; i<kps_l.size(); i++)
    {
      if (mask.size()> 0)
        if (mask[i]==0) continue;

      ceres::CostFunction* cost_function =
          SnavelyReprojectionError::Create(kps_l[i].pt.x,
                                           kps_l[i].pt.y,
                                           kps_r[i].pt.x,
                                           kps_r[i].pt.y,
                                           camera_matrix_,
                                           dist_coef_);

      double* world_point = new double[3];
      world_point[0] = wps[i].x;
      world_point[1] = wps[i].y;
      world_point[2] = wps[i].z;

      problem.AddResidualBlock(cost_function,
                               new ceres::HuberLoss(0.2),
                               camera_params,
                               world_point);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.max_num_iterations = 500;
    options.minimizer_progress_to_stdout = false;
    options.max_solver_time_in_seconds = 0.020; // TODO: update with the frame rate
    options.num_linear_solver_threads = 4;
    options.num_threads = 4;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    ROS_INFO_STREAM("COST: " <<  summary.initial_cost << "  |  " << summary.final_cost);

    // Convert angle axis to quaternion
    double rot[4];
    ceres::AngleAxisToQuaternion(camera_params, rot);

    // Build the transformation
    tf::Quaternion q(rot[1], rot[2], rot[3], rot[0]);
    tf::Vector3 t(camera_params[3], camera_params[4], camera_params[5]);
    pose.setIdentity();
    pose.setRotation(q);
    pose.setOrigin(t);
  }


} //namespace odom