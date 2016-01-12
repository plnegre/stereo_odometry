#include "tracker.h"

namespace odom
{

  Tracker::Tracker() : featools_(new Featools), optimizer_(new Optimizer), frame_uid_(0), feat_uid_(0) {}

  void Tracker::run()
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

    // Set feature tools parameters
    featools_->setParameters(camera_matrix_, dist_coef_, baseline_);

    // Publishers
    stereo_matching_pub_ = nhp.advertise<sensor_msgs::Image>("stereo_matching", 1, true);
    num_tracked_pub_ = nhp.advertise<std_msgs::Int8>("tracked_features", 1, true);

    // Subscribers
    image_transport::ImageTransport it(nh);
    image_transport::SubscriberFilter left_sub, right_sub;
    message_filters::Subscriber<sensor_msgs::CameraInfo> left_info_sub, right_info_sub;

    // Message sync
    boost::shared_ptr<Sync> sync;
    left_sub      .subscribe(it, params_.camera_name+"/left/image_rect", 1);
    right_sub     .subscribe(it, params_.camera_name+"/right/image_rect", 1);
    sync.reset(new Sync(SyncPolicy(5), left_sub, right_sub) );
    sync->registerCallback(bind(&Tracker::msgsCallback, this, _1, _2));

    ros::spin();
  }

  void Tracker::msgsCallback(const sensor_msgs::ImageConstPtr& l_img_msg,
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
    Frame frame(featools_, l_img, r_img, frame_uid_, feat_uid_);
    publishStereoMatches(frame);
    c_frame_ = frame;

    // First iteration
    if (frame_uid_ == 0)
    {
      p_frame_ = c_frame_;
      frame_uid_++;
      return;
    }

    // Current / previous matching
    vector< pair<uint, uint> > c_p_matches;
    searchMatches(c_p_matches);

    // Estimate rotation
    estimateRotation(c_p_matches);

    // Update tracks
    updateTracks(c_p_matches);

    // Prepare next iteration
    p_frame_ = c_frame_;
    frame_uid_++;
  }

  void Tracker::searchMatches(vector< pair<uint, uint> >& c_p_matches)
  {
    c_p_matches.clear();

    // Extract the current frame properties
    cv::Mat c_l_desc, c_r_desc;
    vector<Feature*> c_feat = c_frame_.getFeatures();
    buildDescMat(c_feat, c_l_desc, c_r_desc);

    // Extract the previous frame properties
    cv::Mat p_l_desc, p_r_desc;
    vector<Feature*> p_feat = p_frame_.getFeatures();
    buildDescMat(p_feat, p_l_desc, p_r_desc);

    // // Launch threads to compute in parallel left and right matchings
    vector<cv::DMatch> l_matches, r_matches;
    boost::thread thread_l(&Featools::ratioMatching, *featools_, p_l_desc, c_l_desc, 0.8, boost::ref(l_matches));
    boost::thread thread_r(&Featools::ratioMatching, *featools_, p_r_desc, c_r_desc, 0.8, boost::ref(r_matches));
    thread_l.join();
    thread_r.join();

    // Left/right cross matching
    for (uint i=0; i<l_matches.size(); i++)
    {
      int l_p_idx = l_matches[i].queryIdx;
      int l_c_idx = l_matches[i].trainIdx;
      for (uint j=0; j<r_matches.size(); j++)
      {
        int r_p_idx = r_matches[j].queryIdx;
        int r_c_idx = r_matches[j].trainIdx;
        if (l_c_idx == r_c_idx && l_p_idx == r_p_idx)
        {
          c_p_matches.push_back(make_pair(l_c_idx, l_p_idx));
          break;
        }
      }
    }
  }

  void Tracker::updateTracks(const vector< pair<uint, uint> > c_p_matches)
  {
    int num_tracked = 0;
    vector<Feature*> c_features = c_frame_.getFeatures();
    vector<Feature*> p_features = p_frame_.getFeatures();
    for (uint i=0; i<c_p_matches.size(); i++)
    {
      Feature* c_f = c_features[c_p_matches[i].first];
      Feature* p_f = p_features[c_p_matches[i].second];

      // 1) Search this feature into the tracked features
      bool found = false;
      for (uint n=0; n<tracked_feat_.size(); n++)
      {
        vector<uint> uids = tracked_feat_[n]->getUids();
        for (uint m=0; m<uids.size(); m++)
        {
          uint p_uid = p_f->getLastUid();
          if (uids[m] == p_uid)
          {
            found = true;

            // Add the current view into the tracked features
            tracked_feat_[n]->addView(c_f->getLastUid(),
                                      c_f->getNewestFrameUid(),
                                      c_f->getLastWorldPos(),
                                      c_f->getLastLeftKp(),
                                      c_f->getLastRightKp());
            num_tracked++;
            break;
          }
        }
        if (found) break;
      }

      // 2) If not found, add to the tracked features
      if (!found)
      {
        tracked_feat_.push_back(p_f);
        tracked_feat_[tracked_feat_.size()-1]->addView(c_f->getLastUid(),
                                                       c_f->getNewestFrameUid(),
                                                       c_f->getLastWorldPos(),
                                                       c_f->getLastLeftKp(),
                                                       c_f->getLastRightKp());
      }
    }

    // 3) Remove older features
    uint idx = 0;
    bool end = false;
    while (!end)
    {
      if (idx >= tracked_feat_.size())
      {
        end = true;
        continue;
      }

      uint newest_frame_uid = tracked_feat_[idx]->getNewestFrameUid();
      if (frame_uid_ - newest_frame_uid > 5)
      {
        tracked_feat_[idx] = tracked_feat_.back();
        tracked_feat_.pop_back();
      }
      else
        idx++;
    }

    publishNumTracked(num_tracked);
  }

  void Tracker::estimateRotation(const vector< pair<uint, uint> > c_p_matches)
  {
    vector<Feature*> c_features = c_frame_.getFeatures();
    vector<Feature*> p_features = p_frame_.getFeatures();

    // x,y pairing
    cv::Mat c_points_2f, p_points_2f;
    double* c_points = new double[c_p_matches.size()*2];
    double* p_points = new double[c_p_matches.size()*2];
    for (uint i=0; i<c_p_matches.size(); i++)
    {
      cv::KeyPoint c_kp = c_features[c_p_matches[i].first]->getLastLeftKp();
      cv::KeyPoint p_kp = p_features[c_p_matches[i].second]->getLastLeftKp();
      cv::Mat c_tmp = (cv::Mat_<float>(1,2) << c_kp.pt.x, c_kp.pt.y);
      cv::Mat p_tmp = (cv::Mat_<float>(1,2) << p_kp.pt.x, p_kp.pt.y);
      c_points_2f.push_back(c_tmp);
      p_points_2f.push_back(p_tmp);
      c_points[2*i]   = c_kp.pt.x;
      c_points[2*i+1] = c_kp.pt.y;
      p_points[2*i]   = p_kp.pt.x;
      p_points[2*i+1] = p_kp.pt.y;
    }

    // Find essential matrix
    vector<cv::Mat> E; // essential matrix
    vector<cv::Mat> P; // 3x4 projection matrix
    vector<int> inliers;
    bool ret;
    for(uint i=0; i<100; i++) {
      ret = Solve5PointEssential(c_points, p_points, c_p_matches.size(), E, P, inliers);
    }

    if(ret)
    {
      // The best one has the highest inliers
      uint max_inliers_idx = std::distance(inliers.begin(), max_element(inliers.begin(), inliers.end()));
      cv::Mat E_good = E[max_inliers_idx];

      cv::Mat R, t, mask;
      recoverPose(E_good, c_points_2f, p_points_2f, R, t, mask);

    }
    else
    {
      ROS_ERROR("[StereoOdometry:] Could not find a valid essential matrix.");
    }
  }

  int Tracker::recoverPose(const cv::Mat E, const cv::Mat pts1, const cv::Mat pts2, cv::Mat& R, cv::Mat& t, cv::Mat& mask)
  {
    cv::Mat points1, points2;
    pts1.convertTo(points1, CV_64F);
    pts2.convertTo(points2, CV_64F);

    int npoints = points1.checkVector(2);
    CV_Assert( npoints >= 0 && points2.checkVector(2) == npoints && points1.type() == points2.type());

    CV_Assert(camera_matrix_.rows == 3 && camera_matrix_.cols == 3 && camera_matrix_.channels() == 1);

    if (points1.channels() > 1)
    {
      points1 = points1.reshape(1, npoints);
      points2 = points2.reshape(1, npoints);
    }

    double fx = camera_matrix_.at<double>(0,0);
    double fy = camera_matrix_.at<double>(1,1);
    double cx = camera_matrix_.at<double>(0,2);
    double cy = camera_matrix_.at<double>(1,2);

    points1.col(0) = (points1.col(0) - cx) / fx;
    points2.col(0) = (points2.col(0) - cx) / fx;
    points1.col(1) = (points1.col(1) - cy) / fy;
    points2.col(1) = (points2.col(1) - cy) / fy;

    points1 = points1.t();
    points2 = points2.t();

    cv::Mat R1, R2, t1;
    decomposeEssentialMat(E, R1, R2, t1);
    cv::Mat P0 = cv::Mat::eye(3, 4, R1.type());
    cv::Mat P1(3, 4, R1.type()), P2(3, 4, R1.type()), P3(3, 4, R1.type()), P4(3, 4, R1.type());
    P1(cv::Range::all(), cv::Range(0, 3)) = R1 * 1.0; P1.col(3) = t1 * 1.0;
    P2(cv::Range::all(), cv::Range(0, 3)) = R2 * 1.0; P2.col(3) = t1 * 1.0;
    P3(cv::Range::all(), cv::Range(0, 3)) = R1 * 1.0; P3.col(3) = -t1 * 1.0;
    P4(cv::Range::all(), cv::Range(0, 3)) = R2 * 1.0; P4.col(3) = -t1 * 1.0;

    // Do the cheirality check.
    // Notice here a threshold dist is used to filter
    // out far away points (i.e. infinite points) since
    // there depth may vary between postive and negtive.
    double dist = 50.0;
    cv::Mat Q;
    cv::triangulatePoints(P0, P1, points1, points2, Q);
    cv::Mat mask1 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask1 = (Q.row(2) < dist) & mask1;
    Q = P1 * Q;
    mask1 = (Q.row(2) > 0) & mask1;
    mask1 = (Q.row(2) < dist) & mask1;

    triangulatePoints(P0, P2, points1, points2, Q);
    cv::Mat mask2 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask2 = (Q.row(2) < dist) & mask2;
    Q = P2 * Q;
    mask2 = (Q.row(2) > 0) & mask2;
    mask2 = (Q.row(2) < dist) & mask2;

    triangulatePoints(P0, P3, points1, points2, Q);
    cv::Mat mask3 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask3 = (Q.row(2) < dist) & mask3;
    Q = P3 * Q;
    mask3 = (Q.row(2) > 0) & mask3;
    mask3 = (Q.row(2) < dist) & mask3;

    triangulatePoints(P0, P4, points1, points2, Q);
    cv::Mat mask4 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask4 = (Q.row(2) < dist) & mask4;
    Q = P4 * Q;
    mask4 = (Q.row(2) > 0) & mask4;
    mask4 = (Q.row(2) < dist) & mask4;

    mask1 = mask1.t();
    mask2 = mask2.t();
    mask3 = mask3.t();
    mask4 = mask4.t();

    // If _mask is given, then use it to filter outliers.
    if (!mask.empty())
    {
      CV_Assert(mask.size() == mask1.size());
      cv::bitwise_and(mask, mask1, mask1);
      cv::bitwise_and(mask, mask2, mask2);
      cv::bitwise_and(mask, mask3, mask3);
      cv::bitwise_and(mask, mask4, mask4);
    }
    if (mask.empty())
    {
      mask.create(mask1.size(), CV_8U);
    }

    int good1 = cv::countNonZero(mask1);
    int good2 = cv::countNonZero(mask2);
    int good3 = cv::countNonZero(mask3);
    int good4 = cv::countNonZero(mask4);

    if (good1 >= good2 && good1 >= good3 && good1 >= good4)
    {
      R1.copyTo(R);
      t1.copyTo(t);
      mask1.copyTo(mask);
      return good1;
    }
    else if (good2 >= good1 && good2 >= good3 && good2 >= good4)
    {
      R2.copyTo(R);
      t1.copyTo(t);
      mask2.copyTo(mask);
      return good2;
    }
    else if (good3 >= good1 && good3 >= good2 && good3 >= good4)
    {
      t1 = -t1;
      R1.copyTo(R);
      t1.copyTo(t);
      mask3.copyTo(mask);
      return good3;
    }
    else
    {
      t1 = -t1;
      R2.copyTo(R);
      t1.copyTo(t);
      mask4.copyTo(mask);
      return good4;
    }
  }

  void Tracker::decomposeEssentialMat(const cv::Mat E_in, cv::Mat& R1, cv::Mat& R2, cv::Mat& t )
  {
    cv::Mat E = E_in.reshape(1, 3);
    CV_Assert(E.cols == 3 && E.rows == 3);

    cv::Mat D, U, Vt;
    cv::SVD::compute(E, D, U, Vt);

    if (determinant(U) < 0) U *= -1.;
    if (determinant(Vt) < 0) Vt *= -1.;

    cv::Mat W = (cv::Mat_<double>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
    W.convertTo(W, E.type());

    R1 = U * W * Vt;
    R2 = U * W.t() * Vt;
    t = U.col(2) * 1.0;
  }

  void Tracker::publishStereoMatches(Frame frame)
  {
    if (stereo_matching_pub_.getNumSubscribers() > 0)
    {
      cv::Mat imag_matches = frame.drawStereoMatches();
      cv_bridge::CvImage ros_image;
      ros_image.image = imag_matches.clone();
      ros_image.header.stamp = ros::Time::now();
      ros_image.encoding = "bgr8";
      stereo_matching_pub_.publish(ros_image.toImageMsg());
    }
  }

  void Tracker::publishNumTracked(const int num_tracked)
  {
    if (num_tracked_pub_.getNumSubscribers() > 0)
    {
      std_msgs::Int8 msg;
      msg.data = num_tracked;
      num_tracked_pub_.publish(msg);
    }
  }

  void Tracker::buildDescMat(const vector<Feature*> feat, cv::Mat& l_desc, cv::Mat& r_desc)
  {
    l_desc.release();
    r_desc.release();
    for (uint i=0; i<feat.size(); i++)
    {
      l_desc.push_back(feat[i]->getLastLeftDesc());
      r_desc.push_back(feat[i]->getLastRightDesc());
    }
  }


} //namespace odom
