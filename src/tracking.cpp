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
    vector<MapPoint*> mps = map_->getMapPointsInFrustum(camera_pose_);

    // Matching
    searchMatches(mps, frame);

    // 3. Estimate motion

    // // Publish
    // if (pose_pub_.getNumSubscribers() > 0)
    // {
    //   // Publish
    //   nav_msgs::Odometry odom_msg;
    //   odom_msg.header.stamp = l_img_msg->header.stamp;
    //   tf::poseTFToMsg(acc_pose_, odom_msg.pose.pose);
    //   pose_pub_.publish(odom_msg);
    // }

  }

  void Tracking::searchMatches(vector<MapPoint*> mps, Frame frame)
  {
    // Extract the map points properties
    vector<cv::Point3d> map_wps;
    vector<cv::KeyPoint> map_l_kps, map_r_kps;
    cv::Mat map_l_desc, map_r_desc;
    map_->getPointsProperties(mps, map_wps, map_l_kps, map_r_kps, map_l_desc, map_r_desc);

    // Extract the current frame properties
    cv::Mat frame_l_desc = frame.getLeftMatchedDesc();
    cv::Mat frame_r_desc = frame.getRightMatchedDesc();

    // Launch threads to compute in parallel left and right matchings
    vector<cv::DMatch> l_matches, r_matches;
    boost::thread thread_l(&Feature::ratioMatching, *feat_, map_l_desc, frame_l_desc, 0.8, boost::ref(l_matches));
    boost::thread thread_r(&Feature::ratioMatching, *feat_, map_r_desc, frame_r_desc, 0.8, boost::ref(r_matches));
    thread_l.join();
    thread_r.join();

    // Left/right cross matching
     vector< pair< int,int > > cross_matches;
    for (uint i=0; i<l_matches.size(); i++)
    {
      int l_frame_idx = l_matches[i].trainIdx;
      for (uint j=0; j<r_matches.size(); j++)
      {
        int r_frame_idx = r_matches[j].trainIdx;
        if (l_frame_idx == r_frame_idx && l_matches[i].queryIdx == r_matches[j].queryIdx)
        {
          cross_matches.push_back(make_pair(i, j));
          break;
        }
      }
    }

    // TODO: output matches

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
