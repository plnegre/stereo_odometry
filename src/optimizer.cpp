#include "optimizer.h"

namespace odom
{
  Optimizer::Optimizer() {}

  void Optimizer::poseOptimization(const vector<cv::Point2d> kps, const vector<cv::Point3d> wps, tf::Transform& pose, cv::Mat& camera_matrix, cv::Mat& dist_coef)
  {
    ROS_ASSERT(kps.size() == wps.size());

    // Ceres problem
    ceres::Problem problem;

    // Convert quaternion to angle axis
    double angle_axis[3];
    double quaternion[4] = {pose.getRotation().w(), pose.getRotation().x(), pose.getRotation().y(), pose.getRotation().z()};
    ceres::QuaternionToAngleAxis(quaternion, angle_axis);

    // Initial camera parameters
    double* camera_params = new double[13];
    camera_params[0] = angle_axis[0];
    camera_params[1] = angle_axis[1];
    camera_params[2] = angle_axis[2];
    camera_params[3] = pose.getOrigin().x();
    camera_params[4] = pose.getOrigin().y();
    camera_params[5] = pose.getOrigin().z();
    camera_params[6] = camera_matrix.at<double>(0,0);
    camera_params[7] = camera_matrix.at<double>(0,2);
    camera_params[8] = camera_matrix.at<double>(1,2);
    camera_params[9]  = dist_coef.at<double>(0);
    camera_params[10] = dist_coef.at<double>(1);
    camera_params[11] = dist_coef.at<double>(2);
    camera_params[12] = dist_coef.at<double>(3);

    // Prepare the problem
    for (uint i=0; i<kps.size(); i++) {

      ceres::CostFunction* cost_function =
          SnavelyReprojectionError::Create(kps[i].x, kps[i].y);

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
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 1000;
    options.minimizer_progress_to_stdout = false;
    options.max_solver_time_in_seconds = 600;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    ROS_INFO_STREAM("COST: " <<  summary.initial_cost << "  |  " << summary.final_cost);

    // Save the optimized camera parameters
    camera_matrix.at<double>(0,0) = camera_params[6];
    camera_matrix.at<double>(1,1) = camera_params[6];
    camera_matrix.at<double>(0,2) = camera_params[7];
    camera_matrix.at<double>(1,2) = camera_params[8];
    dist_coef.at<double>(0) = camera_params[9];
    dist_coef.at<double>(1) = camera_params[10];
    dist_coef.at<double>(2) = camera_params[11];
    dist_coef.at<double>(3) = camera_params[12];

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