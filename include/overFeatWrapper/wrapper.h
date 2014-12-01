#include <iostream>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "overfeat.hpp"
#include "tools/cv2TH.hpp"
#include <signal.h>
#include "ros/ros.h"
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

namespace enc = sensor_msgs::image_encodings;
using namespace cv;

class Wrapper{
	private:
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
		
		void init();
		void subscribeToKinect();
		void unsubscribeFromKinect();
		void publishClassification(const sensor_msgs::ImageConstPtr &msg);
		
	public:
		Wrapper(ros::NodeHandle nh);
		~Wrapper();
		bool handleClassificationServiceCall(overFeatWrapper::overFeat::Request& req, overFeatWrapper::overFeat::Response& res);

};
