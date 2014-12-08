#include <iostream>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "overfeat.hpp"
#include "tools/cv2TH.hpp"
#include <signal.h>
#include "std_msgs/String.h"
#include <assert.h>
#include "sensor_msgs/Image.h"
#include "overFeatWrapper/overFeat.h"
#include "overFeatWrapper/detectedObjArray.h"
#include "overFeatWrapper/detectedObj.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ros/ros.h>
#include <Eigen/Core>
//#include <pcl/ros/conversions.h>
//#include <pcl/search/kdtree.h>
//#include <pcl/search/search.h>
/*
#include <pcl/search/pcl_search.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/centroid.h>
#include <pcl/common/pca.h>
#include <pcl/common/eigen.h>
#include <pcl_ros/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/feature.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/sample_consensus/sac_model.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/sac.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
*/
//#include <pcl_conversions/pcl_conversions.h>

namespace enc = sensor_msgs::image_encodings;
using namespace cv;

class Wrapper{
	private:
		//typedef pcl::PointXYZRGB PCLPoint;
		//typedef pcl::PointCloud<PCLPoint> PCLPointCloud;
		ros::NodeHandle node_handle;
		std::string kinect_topic;
		std::string classification_topic;
		std::string pub_topic;
		std::string weight_path_file;
		int network_index;
		//int webcam_index;
		int num_classes;
		double threshold;
		ros::Subscriber kinect_subscriber;
		ros::Publisher classification_pub;
		ros::ServiceServer classify_server;
		VideoCapture* cam;
		Mat_<Vec3b> getCameraFrame_tmp;
		double clustering_tolerance_;
		int cluster_min_size_;
  		int cluster_max_size_;
		
		void init();
		void subscribeToKinect();
		void unsubscribeFromKinect();
		//void publishClassification(const sensor_msgs::PointCloud2::ConstPtr &msg);
		//void filterPointcloud(PCLPointCloud::Ptr& original_pc, PCLPointCloud::Ptr& objects_pointcloud, PCLPointCloud::Ptr& table_pointcloud);
		//void clusterPointcloud(PCLPointCloud::Ptr& cloud, vector<pcl::PointIndices>& cluster_indices);
		//void computeCentroid(const PCLPointCloud& cloud, Eigen::Vector3f& centroid);
		std::vector<std::pair<std::string, float> > overFeatCallBack(sensor_msgs::Image::ConstPtr& img_msg, double tp_left_x, 
			double tp_left_y, double width);
		
	public:
		Wrapper(ros::NodeHandle nh);
		~Wrapper();
		bool handleClassificationServiceCall(overFeatWrapper::overFeat::Request& req, overFeatWrapper::overFeat::Response& res);
		


};
