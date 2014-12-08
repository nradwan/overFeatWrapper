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
	clustering_tolerance_ = 0.025;
	cluster_min_size_ =  50;
  	cluster_max_size_ = 15000;
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
	//kinect_subscriber = node_handle.subscribe(kinect_topic, 1, &Wrapper::publishClassification, this);
	ROS_INFO("Subscribed to: %s", kinect_topic.c_str());
}

void Wrapper::unsubscribeFromKinect(){
	kinect_subscriber.shutdown();
	ROS_WARN("UNsubscribed from: %s", kinect_topic.c_str());
}

/*void Wrapper::filterPointcloud(PCLPointCloud::Ptr& original_pc, PCLPointCloud::Ptr& objects_pointcloud, PCLPointCloud::Ptr& table_pointcloud){
	double plane_thresh = 0.03;
	pcl::ModelCoefficients coefficients;
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	// Create the segmentation object
	pcl::SACSegmentation<pcl::PointXYZRGB> seg;
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(plane_thresh);
	seg.setInputCloud(original_pc);
	seg.segment(*inliers, coefficients);
	
	pcl::ExtractIndices<pcl::PointXYZRGB> extract;
	// Extract table
	extract.setInputCloud(original_pc);
	extract.setIndices(inliers);
	extract.setNegative(false);
	table_pointcloud->clear();
	extract.filter(*table_pointcloud);

	// Extract objects
	extract.setInputCloud(original_pc);
	extract.setIndices(inliers);
	extract.setNegative(true);
	objects_pointcloud->clear();
	extract.filter(*objects_pointcloud);
}

void Wrapper::clusterPointcloud(PCLPointCloud::Ptr& cloud, std::vector<pcl::PointIndices>& cluster_indices){
	// KdTree object for the search method of the extraction
	pcl::search::KdTree<PCLPoint>::Ptr kdtree_cl(new pcl::search::KdTree<PCLPoint>());
	kdtree_cl->setInputCloud(cloud);
	pcl::EuclideanClusterExtraction<PCLPoint> ec;
	ec.setClusterTolerance(clustering_tolerance_);
	ec.setMinClusterSize(cluster_min_size_);
	ec.setMaxClusterSize(cluster_max_size_);
	ec.setSearchMethod(kdtree_cl);
	ec.setInputCloud(cloud);
	ec.extract(cluster_indices);
}
*/
// Computes the object centroid based on the pointcloud
/*void Wrapper::computeCentroid(const PCLPointCloud& cloud, Eigen::Vector3f& centroid){
	Eigen::Vector4f centr;
	pcl::compute3DCentroid(cloud, centr);
	centroid[0] = centr[0];
	centroid[1] = centr[1];
	centroid[2] = centr[2];
}*/

std::vector<std::pair<std::string, float> > Wrapper::overFeatCallBack(sensor_msgs::Image::ConstPtr& img_msg, double tp_left_x, 
			double tp_left_y, double width){
	
	std::vector<std::pair<std::string, float> > empty_res;
	cv_bridge::CvImagePtr cv_ptr;
	try{
      cv_ptr = cv_bridge::toCvCopy(img_msg, enc::BGR8);
    }
    catch (cv_bridge::Exception& e){
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return empty_res;
    }
   //imwrite("/home/noha/Documents/Hiwi/bowl.jpg", cv_ptr->image);
	try {
    
    // Initialization
    overfeat::init(weight_path_file, network_index);
    THTensor* input = THTensor_(new)();
    THTensor* probas = THTensor_(new)();
    ROS_INFO("Finished initialization");
    
    Mat_<Vec3b> cm_image = cv_ptr->image;//imread("/home/noha/Documents/Hiwi/bowl.jpg");
    //resize the image
    Rect myROI(tp_left_x, tp_left_y, width, width);
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
	
	// get top classes
	std::vector<std::pair<std::string, float> > top_classes = overfeat::get_top_classes(probas, num_classes);

    THTensor_(free)(input);
    THTensor_(free)(probas);
    overfeat::free();
	return top_classes;
	
  } catch (cv::Exception & e) {
    ROS_INFO("CV Error!");
   	//killed(0);
    return empty_res;
  }		
}

//void Wrapper::publishClassification(const sensor_msgs::PointCloud2::ConstPtr &msg){
	// Extract table plane. Return a point cloud of the table and one of the objects.
	/*PCLPointCloud::Ptr objs_cloud(new PCLPointCloud);
	PCLPointCloud::Ptr table_cloud(new PCLPointCloud);
	PCLPointCloud::Ptr n_original_pc(new pcl::PCLPointCloud2());
	pcl::fromROSMsg(*msg, *n_original_pc);
	filterPointcloud(n_original_pc, objs_cloud, table_cloud);
	
	//Cluster the objects
	std::vector<pcl::PointIndices> cluster_indices;
	std::vector<pcl::PointIndices>::const_iterator it;
	clusterPointcloud(objs_cloud, cluster_indices);
	std::vector<PCLPointCloud::Ptr> new_clusters; // collect new clusters
	/*std::vector<Eigen::Vector3f> new_cluster_centroids; // collect new cluster centroids
	for (it = cluster_indices.begin(); it != cluster_indices.end(); ++it) {
		PCLPointCloud::Ptr single_cluster(new PCLPointCloud);
		single_cluster->width = it->indices.size();
		single_cluster->height = 1;
		single_cluster->points.reserve(it->indices.size());
		std::vector<int>::const_iterator pit;
		for (pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
			PCLPoint& pnt = objs_cloud->points[*pit];
			single_cluster->points.push_back(pnt);
		}
    	new_clusters.push_back(single_cluster);
		// Compute centroid
 		Eigen::Vector3f new_centr;
		computeCentroid(*single_cluster, new_centr);
		new_cluster_centroids.push_back(new_centr);
	}
	
	//Get the classification of each clustered object
	overFeatWrapper::detectedObjArray top_classes_msg;
	std::vector<PCLPointCloud::Ptr>::iterator clust_it;
	/*std::vector<Eigen::Vector3f>::iterator centr_it = new_cluster_centroids.begin();
	int idx = 0;
	for(clust_it = new_clusters.begin(); clust_it != new_clusters.end(); ++clust_it){
		//convert from pointcloud to image
		sensor_msgs::Image::ConstPtr img_msg;
		pcl::toROSMsg(**clust_it, *img_msg);
		//get coordinates for cropping
		Eigen::Vector3f centr = *centr_it; 
		double tp_left_x = centr(0) - (*clust_it)->points.size();
		double tp_left_y = centr(1) - (*clust_it)->points.size();
		//call classification method
		std::vector<std::pair<std::string, float> > top_classes = overFeatCallBack(img_msg, tp_left_x, tp_left_y, (*clust_it)->points.size());
		//concatenate result to top_classes_msg
		for(std::vector<std::pair<std::string, float> >::iterator it = top_classes.begin();
			it != top_classes.end(); ++it){
			//skip objects with probability less than a certain threshold
			//if(it->second < threshold)
				//continue;
			overFeatWrapper::detectedObj curr_obj;
			curr_obj.obj_name = it->first;
			curr_obj.prob = it->second;
			curr_obj.cluster_idx = idx;
			top_classes_msg.detected_classes.push_back(curr_obj);
		}
		++idx;
		++centr_it;
	}
	classification_pub.publish(top_classes_msg);*/
//}


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
