#include "map.h"

namespace odom
{
  Map::Map(){}

  void Map::setParameters(const cv::Mat camera_matrix, const cv::Mat dist_coef, const int img_w, const int img_h)
  {
    camera_matrix.copyTo(camera_matrix_);
    dist_coef.copyTo(dist_coef_);
    img_w_ = img_w;
    img_h_ = img_h;
  }

  void Map::addMapPoint(MapPoint *mp)
  {
    map_points_.insert(mp);
  }

  void Map::addMapPoints(vector<MapPoint*> mps)
  {
    for (uint i=0; i<mps.size(); i++)
      addMapPoint(mps[i]);
  }

  vector<MapPoint*> Map::getMapPointsInFrustum(const tf::Transform camera_pose, const double uncert)
  {
    vector<MapPoint*> mps;

    for(set<MapPoint*>::iterator sit=map_points_.begin(), send=map_points_.end(); sit!=send; sit++)
    {
      cv::Point2d cp = project3dToPixel((*sit)->getWorldPos(), camera_pose);

      if ((cp.x >= -uncert && cp.x < img_w_+uncert) && (cp.y >= -uncert && cp.y < img_h_+uncert))
        mps.push_back(*sit);
    }

    return mps;
  }

  cv::Point2d Map::project3dToPixel(const cv::Point3d point, const tf::Transform camera_pose)
  {
    // Decompose motion
    tf::Matrix3x3 rot = camera_pose.getBasis();
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
    trans.at<double>(0) = (double)camera_pose.getOrigin().x();
    trans.at<double>(1) = (double)camera_pose.getOrigin().y();
    trans.at<double>(2) = (double)camera_pose.getOrigin().z();

    // Project point
    std::vector<cv::Point3d> object_points;
    object_points.push_back(point);
    vector<cv::Point2d> projected_points;
    cv::projectPoints(object_points, rvec_rod, trans, camera_matrix_, dist_coef_, projected_points);
    return projected_points[0];
  }

  void Map::getPointsProperties(const vector<MapPoint*> mps, vector<cv::Point3d>& wps, vector<cv::KeyPoint>& l_kps, vector<cv::KeyPoint>& r_kps, cv::Mat& l_desc, cv::Mat& r_desc)
  {
    // Init
    wps.clear();
    l_kps.clear();
    r_kps.clear();
    l_desc.release();
    r_desc.release();

    for (uint i=0; i<mps.size(); i++)
    {
      MapPoint* mp = mps[i];

      wps.push_back(mp->getWorldPos());
      l_kps.push_back(mp->getLeftKp());
      r_kps.push_back(mp->getRightKp());
      l_desc.push_back(mp->getLeftDesc());
      r_desc.push_back(mp->getRightDesc());
    }
  }

  void Map::getPointsPositions(const vector<MapPoint*> mps, vector<cv::Point3d>& wps)
  {
    wps.clear();
    for (uint i=0; i<mps.size(); i++)
    {
      MapPoint* mp = mps[i];
      wps.push_back(mp->getWorldPos());
    }
  }

  void Map::getPointsDesc(const vector<MapPoint*> mps, cv::Mat& l_desc, cv::Mat& r_desc)
  {
    l_desc.release();
    r_desc.release();
    for (uint i=0; i<mps.size(); i++)
    {
      MapPoint* mp = mps[i];
      l_desc.push_back(mp->getLeftDesc());
      r_desc.push_back(mp->getRightDesc());
    }
  }

} //namespace odom