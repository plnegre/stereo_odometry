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
    img_w_ = 512;
    img_h_ = 384;

    // Init camera pose
    camera_pose_.setIdentity();

    // Set optimizer parameters
    optimizer_->setParameters(camera_matrix_, dist_coef_, baseline_);

    // Set feature parameters
    feat_->setParameters(camera_matrix_, dist_coef_, baseline_);

    // Set map parameters
    map_->setParameters(camera_matrix_, dist_coef_, img_w_, img_h_);

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
    publishStereoMatches(frame);

    // First iteration
    if (first_)
    {
      // Store map points
      vector<MapPoint*> mps = frame.getMapPoints();
      map_->addMapPoints(mps);

      // Exit
      first_ = false;
      return;
    }

    // Get map points in frustum
    vector<MapPoint*> mps = map_->getMapPointsInFrustum(camera_pose_, 25.0);

    // Matching
    vector<int> matched_map_wps, matched_frame_kps;
    searchMatches(mps, frame, matched_map_wps, matched_frame_kps);

    // Estimate pose
    tf::Transform pose;
    vector<int> inliers;
    estimatePose(mps, frame, matched_map_wps, matched_frame_kps, pose, inliers);

    // Update the map
    updateMap(mps, frame, matched_map_wps, matched_frame_kps, inliers);

    // Publish
    if (pose_pub_.getNumSubscribers() > 0)
    {
      // Publish
      nav_msgs::Odometry odom_msg;
      odom_msg.header.stamp = l_img_msg->header.stamp;
      tf::poseTFToMsg(pose, odom_msg.pose.pose);
      pose_pub_.publish(odom_msg);
    }
  }

  void Tracking::searchMatches(const vector<MapPoint*> mps,
                               Frame frame,
                               vector<int>& matched_map_wps,
                               vector<int>& matched_frame_kps)
  {
    // Extract the map points properties
    cv::Mat map_l_desc, map_r_desc;
    map_->getPointsDesc(mps, map_l_desc, map_r_desc);

    // Extract the current frame properties
    cv::Mat frame_l_desc = frame.getLeftDesc();
    cv::Mat frame_r_desc = frame.getRightDesc();

    // Launch threads to compute in parallel left and right matchings
    vector<cv::DMatch> l_matches, r_matches;
    boost::thread thread_l(&Feature::ratioMatching, *feat_, map_l_desc, frame_l_desc, 0.8, boost::ref(l_matches));
    boost::thread thread_r(&Feature::ratioMatching, *feat_, map_r_desc, frame_r_desc, 0.8, boost::ref(r_matches));
    thread_l.join();
    thread_r.join();

    // Left/right cross matching
    matched_map_wps.clear();
    matched_frame_kps.clear();
    for (uint i=0; i<l_matches.size(); i++)
    {
      int l_map_idx = l_matches[i].queryIdx;
      int l_frame_idx = l_matches[i].trainIdx;
      for (uint j=0; j<r_matches.size(); j++)
      {
        int r_map_idx = r_matches[j].queryIdx;
        int r_frame_idx = r_matches[j].trainIdx;
        if (l_frame_idx == r_frame_idx && l_map_idx == r_map_idx)
        {
          matched_map_wps.push_back(l_map_idx);
          matched_frame_kps.push_back(l_frame_idx);
          break;
        }
      }
    }
  }

  void Tracking::buildMatchedVectors(const vector<MapPoint*> mps,
                                     Frame frame,
                                     const vector<int> matched_map_wps,
                                     const vector<int> matched_frame_kps,
                                     vector<cv::Point3d>& map_wps,
                                     vector<cv::KeyPoint>& frame_l_kps,
                                     vector<cv::KeyPoint>& frame_r_kps)
  {
    ROS_ASSERT(matched_map_wps.size() == matched_frame_kps.size());

    map_wps.clear();
    frame_l_kps.clear();
    frame_r_kps.clear();
    vector<cv::Point3d> map_wps_or;
    map_->getPointsPositions(mps, map_wps_or);
    vector<cv::KeyPoint> frame_l_kps_or = frame.getLeftKp();
    vector<cv::KeyPoint> frame_r_kps_or = frame.getRightKp();
    for (uint i=0; i<matched_map_wps.size(); i++)
    {
      map_wps.push_back(map_wps_or[matched_map_wps[i]]);
      frame_l_kps.push_back(frame_l_kps_or[matched_frame_kps[i]]);
      frame_r_kps.push_back(frame_r_kps_or[matched_frame_kps[i]]);
    }
  }

  void Tracking::estimatePose(const vector<MapPoint*> mps,
                             Frame frame,
                             vector<int> matched_map_wps,
                             vector<int> matched_frame_kps,
                             tf::Transform& pose,
                             vector<int>& inliers)
  {
    ROS_ASSERT(matched_map_wps.size() == matched_frame_kps.size());

    // Extract the matched vectors
    vector<cv::Point3d> map_wps;
    vector<cv::KeyPoint> frame_l_kps, frame_r_kps;
    buildMatchedVectors(mps, frame, matched_map_wps, matched_frame_kps, map_wps, frame_l_kps, frame_r_kps);

    ROS_INFO("------------------------------");
    ROS_INFO_STREAM("MATCHES: " << frame_l_kps.size());

    // Initial optimization
    inliers.resize(map_wps.size(), 1);
    optimizer_->poseOptimization(frame_l_kps, frame_r_kps, map_wps, pose, inliers);

    // Run N optimizations
    const uint opt_n = 3;
    const float chi2[opt_n] = {5.991, 4.605, 2.773}; // Chi-squared distribution
    for (uint i=0; i<opt_n; i++)
    {
      // Remove outliers
      for (uint j=0; j<map_wps.size(); j++)
      {
        if (inliers[j] == 0) continue;

        // Compute error
        cv::Point3d p(map_wps[j].x, map_wps[j].y, map_wps[j].z);
        double error_l = computeProjectionError(pose, p, frame_l_kps[j].pt);
        double error_r = computeProjectionError(pose, p, frame_r_kps[j].pt, baseline_);
        if (error_l >= chi2[i] || error_r >= chi2[i])
          inliers[j] = 0;
      }

      // Count inliers
      int num_inliers = accumulate(inliers.begin(), inliers.end(), 0);

      // Check minimum number of inliers
      if (num_inliers < 12)
        break;

      // Optimize
      optimizer_->poseOptimization(frame_l_kps, frame_r_kps, map_wps, pose, inliers);
    }

    // Count final inliers
    for (uint j=0; j<map_wps.size(); j++)
    {
    	if (inliers[j] == 0) continue;

      // Compute error
      cv::Point3d p(map_wps[j].x, map_wps[j].y, map_wps[j].z);
      double error_l = computeProjectionError(pose, p, frame_l_kps[j].pt);
      double error_r = computeProjectionError(pose, p, frame_r_kps[j].pt, baseline_);
      if (error_l >= chi2[opt_n-1] || error_r >= chi2[opt_n-1])
        inliers[j] = 0;
    }

    ROS_INFO_STREAM("FINAL POSE: " << pose.getOrigin().x() << ", " << pose.getOrigin().y() << ", " << pose.getOrigin().z());
    ROS_INFO_STREAM("FINAL INLIERS: " << accumulate(inliers.begin(), inliers.end(), 0));
  }

  void Tracking::updateMap(const vector<MapPoint*> mps,
                           Frame frame,
                           const vector<int> matched_map_wps,
                           const vector<int> matched_frame_kps,
                           const vector<int> inliers)
  {
    ROS_ASSERT(matched_map_wps.size() == matched_frame_kps.size());
    ROS_ASSERT(matched_map_wps.size() == inliers.size());

    // First, update the existing map points
    for (uint i=0; i<inliers.size(); i++)
    {
      if (inliers[i]==1)
      {
      	// The indices
        int idx_map = matched_map_wps[i];
        int idx_frame = matched_frame_kps[i];

        // Increase the number of times this point have been seen
        MapPoint* mp = mps[idx_map];
        mp->increaseVisible();

        // Update its descriptor if its is better on the current frame
        cv::KeyPoint l_kp = frame.getLeftKp()[idx_frame];
        cv::KeyPoint r_kp = frame.getRightKp()[idx_frame];
        if (mp->getLeftKp().response < l_kp.response &&
            mp->getRightKp().response < r_kp.response)
        {
          cv::Mat l_desc = frame.getLeftDesc().row(idx_frame);
          cv::Mat r_desc = frame.getRightDesc().row(idx_frame);
          mp->setLeftDesc(l_desc);
          mp->setRightDesc(r_desc);
        }
      }
    }

    // Add the new map points
    vector<MapPoint*> frame_mps = frame.getMapPoints();
    int news = 0;
    for (uint i=0; i<mps.size(); i++)
    {
      bool found = false;
      for (uint j=0; j<matched_frame_kps.size(); j++)
      {
        if (matched_frame_kps[j] == i)
        {
          found = true;
          break;
        }
      }
      if (!found)
        news++;
      // int pos = find(matched_frame_kps.begin(), matched_frame_kps.end(), (int)i) - matched_frame_kps.begin();
      // if( pos < matched_frame_kps.size() )
      // {
      //   // if (inliers[pos] == 0)
      //   //   news++;
      //   news = news;
      // }
      // else
      //   news++;
    }

    ROS_INFO_STREAM("TOTAL: " << frame_mps.size() << ". Matched: " << matched_frame_kps.size() << ". Inliers: " << accumulate(inliers.begin(), inliers.end(), 0) << ". New: " << news);

  }

  double Tracking::computeProjectionError(const tf::Transform pose,
                                          const cv::Point3d wp,
                                          const cv::Point2d cp,
                                          const double baseline)
  {
    // Decompose motion
    tf::Matrix3x3 rot = pose.getBasis();
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

    trans.at<double>(0) = (double)pose.getOrigin().x() - baseline;
    trans.at<double>(1) = (double)pose.getOrigin().y();
    trans.at<double>(2) = (double)pose.getOrigin().z();

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
