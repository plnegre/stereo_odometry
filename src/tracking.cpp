#include "tracking.h"

namespace odom
{

  Tracking::Tracking() : feat_(new Feature), optimizer_(new Optimizer), map_(new Map), first_(true), inliers_for_cam_update_(10) {}

  void Tracking::run()
  {
    ros::NodeHandle nh;
    ros::NodeHandle nhp("~");

    // Camera parameters
    cv::Mat camera_matrix = cv::Mat::eye(3, 3, cv::DataType<double>::type);
    camera_matrix.at<double>(0,0) = 333.535;
    camera_matrix.at<double>(1,1) = 333.535;
    camera_matrix.at<double>(0,2) = 236.379;
    camera_matrix.at<double>(1,2) = 185.615;
    camera_matrix.copyTo(camera_matrix_);

    cv::Mat dist_coef(4, 1, cv::DataType<double>::type);
    dist_coef.at<double>(0) = 0.0;
    dist_coef.at<double>(1) = 0.0;
    dist_coef.at<double>(2) = 0.0;
    dist_coef.at<double>(3) = 0.0;
    dist_coef.copyTo(dist_coef_);

    baseline_ = 0.12;

    // Init
    acc_pose_.setIdentity();

    // Publishers
    stereo_matching_pub_ = nhp.advertise<sensor_msgs::Image>("stereo_matching", 1, true);
    pose_pub_ = nhp.advertise<nav_msgs::Odometry>("odometry", 1);

    // Subscribers
    image_transport::ImageTransport it(nh);
    image_transport::SubscriberFilter left_sub, right_sub;
    message_filters::Subscriber<sensor_msgs::CameraInfo> left_info_sub, right_info_sub;

    // Message sync
    boost::shared_ptr<Sync> sync;
    left_sub      .subscribe(it, params_.camera_name+"/left/image_mono", 1);
    right_sub     .subscribe(it, params_.camera_name+"/right/image_mono", 1);
    sync.reset(new Sync(SyncPolicy(5), left_sub, right_sub) );
    sync->registerCallback(bind(&Tracking::msgsCallback, this, _1, _2));

    ros::spin();
  }

  void Tracking::msgsCallback(const sensor_msgs::ImageConstPtr& l_img_msg,
                              const sensor_msgs::ImageConstPtr& r_img_msg)
  {

    // Convert message to cv::Mat
    cv_bridge::CvImageConstPtr l_img_ptr, r_img_ptr;
    try
    {
      l_img_ptr = cv_bridge::toCvShare(l_img_msg);
      r_img_ptr = cv_bridge::toCvShare(r_img_msg);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("[StereoOdometry:] cv_bridge exception: %s", e.what());
      return;
    }

    ROS_ASSERT(l_img_ptr->image.channels()==3 || l_img_ptr->image.channels()==1);
    ROS_ASSERT(r_img_ptr->image.channels()==3 || r_img_ptr->image.channels()==1);
    ROS_ASSERT(l_img_ptr->image.channels() == r_img_ptr->image.channels());

    cv::Mat l_img, r_img;
    if(l_img_ptr->image.channels()==3)
    {
      cvtColor(l_img_ptr->image, l_img, CV_RGB2GRAY);
      cvtColor(r_img_ptr->image, r_img, CV_RGB2GRAY);
    }
    else if(l_img_ptr->image.channels()==1)
    {
      l_img_ptr->image.copyTo(l_img);
      r_img_ptr->image.copyTo(r_img);
    }

    // Setup the frame
    Frame frame(l_img, r_img, camera_matrix_, dist_coef_, baseline_);
    c_frame_ = frame;
    publishStereoMatches(c_frame_);

    // First iteration
    if (first_)
    {
      // Init feature tools
      feat_->init(camera_matrix_, dist_coef_, baseline_);

      // Store map points
      vector<MapPoint*> mps = c_frame_.getMapPoints();
      map_->addMapPoints(mps);

      // Exit
      p_frame_ = c_frame_;
      first_ = false;
      return;
    }

    vector<cv::Point2d> matched_kps;
    vector<cv::Point3d> matched_wp;
    searchMatches(matched_kps, matched_wp);

    tf::Transform delta;
    int inliers = estimateMotion(matched_kps, matched_wp, delta);

    if (inliers >= 12)
    {
      acc_pose_ *= delta;
    }

    // Publish
    if (pose_pub_.getNumSubscribers() > 0)
    {
      // Publish
      nav_msgs::Odometry odom_msg;
      odom_msg.header.stamp = l_img_msg->header.stamp;
      tf::poseTFToMsg(acc_pose_, odom_msg.pose.pose);
      pose_pub_.publish(odom_msg);
    }

    p_frame_ = c_frame_;
  }

  void Tracking::searchMatches(vector<cv::Point2d>& matched_kps, vector<cv::Point3d>& matched_wp)
  {
    matched_kps.clear();
    matched_wp.clear();

    // Frame to frame feature matching
    vector<cv::DMatch> l_matches, r_matches;
    cv::Mat l_c_desc = c_frame_.getLeftMatchedDesc();
    cv::Mat l_p_desc = p_frame_.getLeftMatchedDesc();
    cv::Mat r_c_desc = c_frame_.getRightMatchedDesc();
    cv::Mat r_p_desc = p_frame_.getRightMatchedDesc();
    // TODO: launch in separate threads
    feat_->ratioMatching(l_c_desc, l_p_desc, 0.8, l_matches);
    feat_->ratioMatching(r_c_desc, r_p_desc, 0.8, r_matches);

    vector<int> cross_matches;
    for (uint i=0; i<l_matches.size(); i++)
    {
      int idx_left = l_matches[i].queryIdx;
      for (uint j=0; j<r_matches.size(); j++)
      {
        if (r_matches[j].queryIdx == idx_left)
        {
          cross_matches.push_back(i);
          break;
        }
      }
    }

    vector<MapPoint*> mps = p_frame_.getMapPoints();
    vector<cv::Point3d> p_world_points;
    for (uint i=0; i<mps.size(); i++)
    {
      MapPoint* mp = mps[i];
      p_world_points.push_back(mp->getWorldPos());
    }

    vector<cv::KeyPoint> l_c_kps = c_frame_.getLeftMatchedKp();
    for (uint i=0; i<cross_matches.size(); i++)
    {
      int idx_c = l_matches[cross_matches[i]].queryIdx;
      int idx_p = l_matches[cross_matches[i]].trainIdx;

      matched_kps.push_back(l_c_kps[idx_c].pt);
      matched_wp.push_back(p_world_points[idx_p]);
    }
  }

  int Tracking::estimateMotion(vector<cv::Point2d> matched_kps, vector<cv::Point3d> matched_wp, tf::Transform& delta)
  {
    // Init
    delta.setIdentity();
    cv::Mat camera_matrix, dist_coef;
    camera_matrix_.copyTo(camera_matrix);
    dist_coef_.copyTo(dist_coef);

    ROS_INFO(" ");
    ROS_INFO(" ");
    ROS_INFO("------------------------------");

    // Initial optimization
    optimizer_->poseOptimization(matched_kps, matched_wp, delta, camera_matrix, dist_coef);

    ROS_INFO_STREAM("MATCHES: " << matched_kps.size());
    ROS_INFO_STREAM("POSE: " << delta.getOrigin().x() << ", " << delta.getOrigin().y() << ", " << delta.getOrigin().z());
    ROS_INFO_STREAM("CAM: " << camera_matrix.at<double>(0,0) << ", " << camera_matrix.at<double>(0,2) << ", " << camera_matrix.at<double>(1,2));
    ROS_INFO_STREAM("DIST: " << dist_coef.at<double>(0) << ", " << dist_coef.at<double>(1) << ", " << dist_coef.at<double>(2) << ", " << dist_coef.at<double>(3));

    // Run 4 optimization
    int inliers = 0;
    const uint opt_n = 3;
    const float err2[opt_n] = {10.60, 9.210, 7.378};
    for (uint i=0; i<opt_n; i++)
    {
      // Remove outliers
      vector<cv::Point2d> matched_kps_filtered;
      vector<cv::Point3d> matched_wp_filtered;
      for (uint j=0; j<matched_wp.size(); j++)
      {
        // Convert point to current camera coordinates
        tf::Vector3 wp_p(matched_wp[j].x, matched_wp[j].y, matched_wp[j].z);
        tf::Vector3 wp_c = delta.inverse() * wp_p;
        cv::Point3d p(wp_c.x(), wp_c.y(), wp_c.z());

        // Compute error
        double error = computeProjectionError(p, matched_kps[j], camera_matrix, dist_coef);

        if (error < err2[i])
        {
          matched_kps_filtered.push_back(matched_kps[j]);
          matched_wp_filtered.push_back(matched_wp[j]);
        }
      }

      matched_kps = matched_kps_filtered;
      matched_wp = matched_wp_filtered;
      inliers = matched_wp.size();

      ROS_INFO_STREAM("INLIERS: " << inliers);

      // Check minimum number of inliers
      if (inliers < 12)
        break;

      // Optimize
      delta.setIdentity();
      optimizer_->poseOptimization(matched_kps, matched_wp, delta, camera_matrix, dist_coef);
    }

    ROS_INFO_STREAM("FINAL POSE: " << delta.getOrigin().x() << ", " << delta.getOrigin().y() << ", " << delta.getOrigin().z());
    ROS_INFO_STREAM("FINAL INLIERS: " << inliers);

    // Store camera information when measurement is good enough
    if (inliers > 25 && inliers > inliers_for_cam_update_)
    {
      camera_matrix.copyTo(camera_matrix_);
      dist_coef.copyTo(dist_coef_);
      inliers_for_cam_update_ = inliers;
    }

    return inliers;
  }

  double Tracking::computeProjectionError(const cv::Point3d wp, const cv::Point2d cp, const cv::Mat camera_matrix, const cv::Mat dist_coef)
  {
    // Project point
    std::vector<cv::Point3d> object_points;
    object_points.push_back(wp);
    cv::Mat rvec = cv::Mat::eye(3, 3, cv::DataType<double>::type);
    cv::Mat rvec_rod(3, 1, cv::DataType<double>::type);
    cv::Rodrigues(rvec, rvec_rod);
    cv::Mat trans = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
    vector<cv::Point2d> projected_points;
    cv::projectPoints(object_points, rvec_rod, trans, camera_matrix, dist_coef, projected_points);
    cv::Point2d proj_wp_c = projected_points[0];

    // Compute error
    return sqrt( (proj_wp_c.x-cp.x)*(proj_wp_c.x-cp.x) + (proj_wp_c.y-cp.y)*(proj_wp_c.y-cp.y));
  }

  void Tracking::publishStereoMatches(Frame frame)
  {
    if (stereo_matching_pub_.getNumSubscribers() > 0)
    {
      cv::Mat imag_matches = frame.drawMatches();
      cv_bridge::CvImage ros_image;
      ros_image.image = imag_matches.clone();
      ros_image.header.stamp = ros::Time::now();
      ros_image.encoding = "bgr8";
      stereo_matching_pub_.publish(ros_image.toImageMsg());
    }
  }


} //namespace odom
