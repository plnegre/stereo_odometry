#include "tracking.h"

namespace odom
{

  Tracking::Tracking() : feat_(new Feature), optimizer_(new Optimizer), map_(new Map), first_(true) {}

  void Tracking::run()
  {
    ros::NodeHandle nh;
    ros::NodeHandle nhp("~");

    // Camera parameters
    cv::Mat camera_matrix = cv::Mat::eye(3, 3, cv::DataType<double>::type);
    camera_matrix.at<double>(0,0) = 343.02;
    camera_matrix.at<double>(1,1) = 343.02;
    camera_matrix.at<double>(0,2) = 235.31;
    camera_matrix.at<double>(1,2) = 194.75;
    camera_matrix.copyTo(camera_matrix_);

    cv::Mat dist_coef(4, 1, cv::DataType<double>::type);
    dist_coef.at<double>(0) = 0.0;
    dist_coef.at<double>(1) = 0.0;
    dist_coef.at<double>(2) = 0.0;
    dist_coef.at<double>(3) = 0.0;
    dist_coef.copyTo(dist_coef_);

    baseline_ = 0.1211;

    // Set optimizer parameters
    optimizer_->setParameters(camera_matrix_, dist_coef_, baseline_);

    // Init accumulated pose
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
    left_sub      .subscribe(it, params_.camera_name+"/left/image_rect", 1);
    right_sub     .subscribe(it, params_.camera_name+"/right/image_rect", 1);
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

    // Frame to frame matching
    vector<cv::Point2d> matched_p_kp_l;
    vector<cv::Point2d> matched_p_kp_r;
    vector<cv::Point3d> matched_c_wp;
    searchMatches(matched_p_kp_l, matched_p_kp_r, matched_c_wp);


    // ******************************************************
    vector<cv::Point2f> matched_kps2;
    vector<cv::Point3f> matched_wp2;
    for (uint i=0; i<matched_c_wp.size(); i++)
    {
      matched_kps2.push_back( cv::Point2f( (float)matched_p_kp_l[i].x, (float)matched_p_kp_l[i].y ) );
      matched_wp2.push_back( cv::Point3f( (float)matched_c_wp[i].x, (float)matched_c_wp[i].y, (float)matched_c_wp[i].z ) );
    }
    vector<int> inliers_spnp;
    cv::Mat rvec, tvec;
    cv::solvePnPRansac(matched_wp2, matched_kps2, camera_matrix_,
        cv::Mat(), rvec, tvec, false,
        100, 1.5, 200, inliers_spnp);
    tf::Vector3 axis(rvec.at<double>(0, 0),
                     rvec.at<double>(1, 0),
                     rvec.at<double>(2, 0));
    double angle = norm(rvec);
    tf::Quaternion quaternion(axis, angle);
    tf::Vector3 translation(tvec.at<double>(0, 0), tvec.at<double>(1, 0),
        tvec.at<double>(2, 0));
    tf::Transform solvepnp(quaternion, translation);
    // ******************************************************

    tf::Transform delta = solvepnp;
    int inliers = estimateMotion(matched_p_kp_l, matched_p_kp_r, matched_c_wp, delta);

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

  void Tracking::searchMatches(vector<cv::Point2d>& matched_p_kp_l, vector<cv::Point2d>& matched_p_kp_r, vector<cv::Point3d>& matched_c_wp)
  {
    matched_p_kp_l.clear();
    matched_p_kp_r.clear();
    matched_c_wp.clear();

    // Frame to frame feature matching
    vector<cv::DMatch> l_matches, r_matches;
    cv::Mat l_c_desc = p_frame_.getLeftMatchedDesc();
    cv::Mat l_p_desc = c_frame_.getLeftMatchedDesc();
    cv::Mat r_c_desc = p_frame_.getRightMatchedDesc();
    cv::Mat r_p_desc = c_frame_.getRightMatchedDesc();

    // TODO: launch in separate threads
    feat_->ratioMatching(l_c_desc, l_p_desc, 0.8, l_matches);
    feat_->ratioMatching(r_c_desc, r_p_desc, 0.8, r_matches);

    // Cross-matches
    vector<int> cross_matches_l;
    vector<int> cross_matches_r;
    for (uint i=0; i<l_matches.size(); i++)
    {
      int idx_left_current = l_matches[i].queryIdx;
      int idx_left_previous = l_matches[i].trainIdx;
      for (uint j=0; j<r_matches.size(); j++)
      {
        if (r_matches[j].queryIdx == idx_left_current && r_matches[j].trainIdx == idx_left_previous)
        {
          cross_matches_l.push_back(i);
          cross_matches_r.push_back(j);
          break;
        }
      }
    }

    // Get current frame world points
    vector<MapPoint*> mps = c_frame_.getMapPoints();
    vector<cv::Point3d> p_world_points;
    for (uint i=0; i<mps.size(); i++)
    {
      MapPoint* mp = mps[i];
      p_world_points.push_back(mp->getWorldPos());
    }

    // Build the matches vectors
    vector<cv::KeyPoint> l_c_kps = p_frame_.getLeftMatchedKp();
    vector<cv::KeyPoint> r_c_kps = p_frame_.getRightMatchedKp();
    for (uint i=0; i<cross_matches_l.size(); i++)
    {
      int idx_c_l = l_matches[cross_matches_l[i]].queryIdx;
      int idx_c_r = r_matches[cross_matches_r[i]].queryIdx;
      int idx_p = l_matches[cross_matches_l[i]].trainIdx;

      matched_p_kp_l.push_back(l_c_kps[idx_c_l].pt);
      matched_p_kp_r.push_back(r_c_kps[idx_c_r].pt);
      matched_c_wp.push_back(p_world_points[idx_p]);
    }
  }

  int Tracking::estimateMotion(vector<cv::Point2d> matched_p_kp_l, vector<cv::Point2d> matched_p_kp_r, vector<cv::Point3d> matched_c_wp, tf::Transform& delta)
  {
    ROS_INFO("------------------------------");
    ROS_INFO_STREAM("MATCHES: " << matched_p_kp_l.size());

    // Initial optimization
    optimizer_->poseOptimization(matched_p_kp_l, matched_p_kp_r, matched_c_wp, delta);

    // Run N optimizations
    const uint opt_n = 3;
    const float chi2[opt_n] = {5.991, 4.605, 2.773}; // Chi-squared distribution
    for (uint i=0; i<opt_n; i++)
    {
      // Remove outliers
      vector<cv::Point2d> matched_kps_filtered_l;
      vector<cv::Point2d> matched_kps_filtered_r;
      vector<cv::Point3d> matched_wp_filtered;
      int inliers = 0;
      for (uint j=0; j<matched_c_wp.size(); j++)
      {
        // Compute error
        cv::Point3d p(matched_c_wp[j].x, matched_c_wp[j].y, matched_c_wp[j].z);
        double error_l = computeProjectionError(delta, p, matched_p_kp_l[j]);
        double error_r = computeProjectionError(delta, p, matched_p_kp_r[j], baseline_);
        if (error_l < chi2[i] && error_r < chi2[2])
        {
          matched_kps_filtered_l.push_back(matched_p_kp_l[j]);
          matched_kps_filtered_r.push_back(matched_p_kp_r[j]);
          matched_wp_filtered.push_back(matched_c_wp[j]);
          inliers++;
        }
      }

      // Check minimum number of inliers
      if (inliers < 12)
        break;

      // Update
      matched_p_kp_l = matched_kps_filtered_l;
      matched_p_kp_r = matched_kps_filtered_r;
      matched_c_wp   = matched_wp_filtered;

      // Optimize
      optimizer_->poseOptimization(matched_p_kp_l, matched_p_kp_r, matched_c_wp, delta);
    }

    double mean_error = 0.0;
    int inliers = 0;
    for (uint j=0; j<matched_c_wp.size(); j++)
    {
      // Compute error
      cv::Point3d p(matched_c_wp[j].x, matched_c_wp[j].y, matched_c_wp[j].z);
      double error_l = computeProjectionError(delta, p, matched_p_kp_l[j]);
      double error_r = computeProjectionError(delta, p, matched_p_kp_r[j], baseline_);
      mean_error += error_l + error_r;
      if (error_l < chi2[2] && error_r < chi2[2])
        inliers++;
    }

    ROS_INFO_STREAM("FINAL POSE: " << delta.getOrigin().x() << ", " << delta.getOrigin().y() << ", " << delta.getOrigin().z());
    ROS_INFO_STREAM("FINAL INLIERS: " << inliers);
    ROS_INFO_STREAM("MEAN ERROR: " << mean_error / (2*matched_c_wp.size()));

    return inliers;
  }

  double Tracking::computeProjectionError(const tf::Transform delta, const cv::Point3d wp, const cv::Point2d cp, double baseline)
  {
    // Decompose motion
    tf::Matrix3x3 rot = delta.getBasis();
    cv::Mat rvec = cv::Mat::zeros(3, 3, cv::DataType<double>::type);
    cv::Mat trans = cv::Mat::zeros(3, 1, cv::DataType<double>::type);
    rvec.at<double>(0,0) = (double)rot[0][0];
    rvec.at<double>(0,1) = (double)rot[0][1];
    rvec.at<double>(0,2) = (double)rot[0][2];
    rvec.at<double>(1,0) = (double)rot[1][0];
    rvec.at<double>(1,1) = (double)rot[1][1];
    rvec.at<double>(1,2) = (double)rot[1][2];
    rvec.at<double>(2,0) = (double)rot[2][0];
    rvec.at<double>(2,1) = (double)rot[2][1];
    rvec.at<double>(2,2) = (double)rot[2][2];
    cv::Mat rvec_rod(3, 1, cv::DataType<double>::type);
    cv::Rodrigues(rvec, rvec_rod);

    trans.at<double>(0) = (double)delta.getOrigin().x() - baseline;
    trans.at<double>(1) = (double)delta.getOrigin().y();
    trans.at<double>(2) = (double)delta.getOrigin().z();

    // Project point
    std::vector<cv::Point3d> object_points;
    object_points.push_back(wp);
    vector<cv::Point2d> projected_points;
    cv::projectPoints(object_points, rvec_rod, trans, camera_matrix_, dist_coef_, projected_points);
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
