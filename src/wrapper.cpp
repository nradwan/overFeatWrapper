#include "overFeatWrapper/wrapper.h"


Wrapper::Wrapper(ros::NodeHandle nh):
				node_handle(nh) {
	init();				
}

Wrapper::~Wrapper(){

}

void Wrapper::init(){
	
	kinect_topic = "/camera/rgb/image_rect_color";
	classification_topic = "OverFeat";
	pub_topic = "OverFeatClassif";
	//initialize overfeat parameters
	weight_path_file = "/home/noha/Documents/Hiwi/overfeat/data/default/net_weight_0";
	network_index = 0;
	//webcam_index = 0;
	num_classes = 10;
	threshold = 0.0;
	cam = NULL;
	//announce classification service
	classify_server = node_handle.advertiseService(classification_topic, &Wrapper::handleClassificationServiceCall, this);
	ROS_INFO("Announced service: %s", classification_topic.c_str());
	
	
}

bool Wrapper::handleClassificationServiceCall(overFeatWrapper::overFeat::Request& req, overFeatWrapper::overFeat::Response& res){
	if (req.command == overFeatWrapper::overFeat::Request::OVERFEAT_SUBSCRIBE){
		subscribeToKinect();	
	}
	else if (req.command == overFeatWrapper::overFeat::Request::OVERFEAT_UNSUBSCRIBE) {
		unsubscribeFromKinect();
	} else {
		ROS_ERROR("Cannot recognize command in handleClassificationServiceCall.");
		res.result = overFeatWrapper::overFeatResponse::FAILURE;
		return false;
	}

	res.result = overFeatWrapper::overFeatResponse::SUCCESS;

	return true;
}

void Wrapper::subscribeToKinect(){
	kinect_subscriber = node_handle.subscribe(kinect_topic, 1, &Wrapper::publishClassification, this);
	ROS_INFO("Subscribed to: %s", kinect_topic.c_str());
}

void Wrapper::unsubscribeFromKinect(){
	kinect_subscriber.shutdown();
	ROS_WARN("UNsubscribed from: %s", kinect_topic.c_str());
}

void Wrapper::publishClassification(const sensor_msgs::ImageConstPtr &msg){
	ROS_INFO("Inside publishClassification");
	//convert the kinect image to cv image
	cv_bridge::CvImagePtr cv_ptr;
	try{
      cv_ptr = cv_bridge::toCvCopy(msg, enc::BGR8);
    }
    catch (cv_bridge::Exception& e){
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
   //imwrite("/home/noha/Documents/Hiwi/bowl.jpg", cv_ptr->image);
	try {
    
    // Initialization
    overfeat::init(weight_path_file, network_index);
    THTensor* input = THTensor_(new)();
    THTensor* probas = THTensor_(new)();
    ROS_INFO("Finished initialization");
    
    Mat_<Vec3b> cm_image =  imread("/home/noha/Documents/Hiwi/bowl.jpg");//cv_ptr->image;
    //resize the image
    int cam_w = cm_image.size().width;
    int cam_h = cm_image.size().height;
    Rect myROI(cam_w/2 - cam_h/2, 0, cam_h, cam_h);
    Mat_<Vec3b> im;
  	cm_image = cm_image(myROI);
  	resize(cm_image, im, Size(231, 231), 0, 0, INTER_CUBIC);
    cv2TH(im, input);
    
    cv::imshow("test_window", im);
    cv::waitKey(3);
    
    ROS_INFO("Finished image conversion");

	// Read image from webcam
	// minimum input size : 3x231x231
	//getCameraFrame(webcam_index, input, 231, 231);

	// Extract features and classify
	THTensor* output = overfeat::fprop(input);
	ROS_INFO("Finished extract features and classify");
	
	// Convert output into probabilities
	assert((output->size[1] == 1) && (output->size[2] == 1));
	output->nDimension = 1;
	overfeat::soft_max(output, probas);

	ROS_INFO("Finished converting output to probabilities");
	
	// Create msg and publish the output in it
	std::vector<std::pair<std::string, float> > top_classes = overfeat::get_top_classes(probas, num_classes);
	classification_pub = node_handle.advertise<overFeatWrapper::detectedObjArray> (pub_topic, 1);
	ROS_INFO("Advertised topic: %s", pub_topic.c_str());
	overFeatWrapper::detectedObjArray top_classes_msg;
	for(std::vector<std::pair<std::string, float> >::iterator it = top_classes.begin();
			it != top_classes.end(); ++it){
		//skip objects with probability less than a certain threshold
		//if(it->second < threshold)
			//continue;
		overFeatWrapper::detectedObj curr_obj;
		curr_obj.obj_name = it->first;
		curr_obj.prob = it->second;
		top_classes_msg.detected_classes.push_back(curr_obj);
	}
	classification_pub.publish(top_classes_msg);
	//displayWithConf(input, top_classes);

    THTensor_(free)(input);
    THTensor_(free)(probas);
    overfeat::free();

  } catch (cv::Exception & e) {
    ROS_INFO("CV Error!");
   	//killed(0);
    return;
  }
}


int main(int argc, char** argv){
	
	ros::init(argc, argv, "overFeatWrapper");
	ros::NodeHandle nh;
	Wrapper overfeat_wrapper (nh);
	while (ros::ok()) {
		ros::Duration(0.07).sleep();
		ros::spinOnce();
	}
	
	ros::spin();

	return 0;
}
