#include <iostream>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "overfeat.hpp"
#include "tools/cv2TH.hpp"
#include <signal.h>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include <assert.h>
#include "sensor_msgs/PointCloud2.h"
#include "overFeatWrapper/overFeat.h"
#include "overFeatWrapper/detectedObjArray.h"
#include "overFeatWrapper/detectedObj.h"

using namespace cv;

class Wrapper{
	private:
		ros::NodeHandle node_handle;
		std::string kinect_topic;
		std::string classification_topic;
		std::string weight_path_file;
		int network_index;
		int webcam_index;
		int num_classes;
		double threshold;
		ros::Subscriber kinect_subscriber;
		ros::Publisher classification_pub;
		ros::ServiceServer classify_server;
		VideoCapture* cam;
		Mat_<Vec3b> getCameraFrame_tmp;
		
		void init();
		void subscribeToKinect();
		void unsubscribeFromKinect();
		void publishClassification(const sensor_msgs::PointCloud2::ConstPtr &msg);
		void getCameraFrame(int webcamidx, THTensor* output, int w = -1, int h = -1);
		
	public:
		Wrapper(ros::NodeHandle nh);
		~Wrapper();
		bool handleClassificationServiceCall(overFeatWrapper::overFeat::Request& req, overFeatWrapper::overFeat::Response& res);

};
