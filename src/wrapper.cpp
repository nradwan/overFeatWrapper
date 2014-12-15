#include "overFeatWrapper/wrapper.h"


Wrapper::Wrapper(ros::NodeHandle nh):
				node_handle(nh) {
	init();				
}

Wrapper::~Wrapper(){

}

void Wrapper::init(){
	
	kinect_topic = "/camera/depth_registered/points";
	classification_topic = "OverFeat";
	pub_topic = "OverFeatClassif";
	table_pub_topic = "tableTop";
	clustered_objs_pub_topic = "clusteredObjs";
	frame_id = "/camera_depth_optical_frame";
	//initialize overfeat parameters
	weight_path_file = "/home/radwann/overfeat/data/default/net_weight_0";
	network_index = 0;
	//webcam_index = 0;
	num_classes = 10;
	threshold = 0.0;
	cam = NULL;
	clustering_tolerance_ = 0.025;
	cluster_min_size_ =  50;
  	cluster_max_size_ = 15000;
  	maximum_obj_size = 100; //(50 cm)
	//announce classification service
	classify_server = node_handle.advertiseService(classification_topic, &Wrapper::handleClassificationServiceCall, this);
	ROS_INFO("Announced service: %s", classification_topic.c_str());
	classification_pub = node_handle.advertise<overFeatWrapper::detectedObjArray>(pub_topic, 1);
	ROS_INFO("Publishing classification topic: %s", pub_topic.c_str());
	table_pub = node_handle.advertise<sensor_msgs::PointCloud2>(table_pub_topic, 1);
	ROS_INFO("Publishing table pointcloud topic: %s", table_pub_topic.c_str());
	clustered_objs_pub = node_handle.advertise<visualization_msgs::MarkerArray>(clustered_objs_pub_topic, 1);
	ROS_INFO("Publishing clustered objects topic: %s", clustered_objs_pub_topic.c_str());
	
	temp_pub = node_handle.advertise<PCLPointCloud>("test_publisher", 1);
	
	
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

void Wrapper::filterPointcloud(const sensor_msgs::PointCloud2::ConstPtr& original_pc, PCLPointCloud::Ptr& objects_pointcloud, PCLPointCloud::Ptr& table_pointcloud){
	// Filtering purposes. Remove points with z > max_z
	double max_z = 0.9 + 0.5;
	double min_x = 0.01;// Aachen
	double max_x = 0.65;  
	double min_y = 0.01;// Aachen
	double max_y = 1.5;
	//remove all points with z < table_height
	double table_height = -0.03; // landmark

	//////////////////////////// Remove Outliers ////////////////////////////
	sensor_msgs::PointCloud2 clean_pcl_msg;
 	pcl::StatisticalOutlierRemoval<sensor_msgs::PointCloud2> outlier_rem;
	outlier_rem.setInputCloud(original_pc);
	outlier_rem.setMeanK(50);
	outlier_rem.setStddevMulThresh(1.0);
	outlier_rem.filter(clean_pcl_msg);

	PCLPointCloud::Ptr scene_pcl(new PCLPointCloud);
	pcl::fromROSMsg(clean_pcl_msg, *scene_pcl);

	//////////////////////////// Filter out far away points ////////////////////////////
	// TODO Use table plane normal as the z axis and filter out everything below it.
	PCLPointCloud::Ptr scene_filtered(new PCLPointCloud);
	pcl::PassThrough<pcl::PointXYZRGB> pass;
	pass.setInputCloud(boost::make_shared < pcl::PointCloud<pcl::PointXYZRGB> > (*scene_pcl));
	pass.setFilterFieldName("z");
	pass.setFilterLimits(table_height, max_z);	
	//pass.setFilterLimits(-0.2, max_z);
	pass.filter(*scene_filtered);

//  ROS_WARN("size after z clipping = %u", scene_filtered->points.size());
	
  	pass.setInputCloud(boost::make_shared < pcl::PointCloud<pcl::PointXYZRGB> > (*scene_filtered));
	pass.setFilterFieldName("x");
	pass.setFilterLimits(min_x, max_x);	
	pass.filter(*scene_filtered);

//  ROS_WARN("size after x clipping = %u", scene_filtered->points.size());

  	pass.setInputCloud(boost::make_shared < pcl::PointCloud<pcl::PointXYZRGB> > (*scene_filtered));
	pass.setFilterFieldName("y");
	pass.setFilterLimits(min_y, max_y);	
	pass.filter(*scene_filtered);



//  ROS_WARN("size after y clipping = %u", scene_filtered->points.size());

	if (scene_filtered->empty()) {
		ROS_WARN("No points left after filtering by distance.");
	}

//	sensor_msgs::PointCloud2 fml;
//	pcl::toROSMsg(*scene_filtered, fml);
 // point_cloud_publisher_.publish(fml);
	
	double plane_thresh = 0.03;
	//PCLPointCloud::Ptr pcl_original(new PCLPointCloud);
	//pcl::fromROSMsg(*original_pc, *pcl_original);
	pcl::ModelCoefficients coefficients;
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
	// Create the segmentation object
	pcl::SACSegmentation<PCLPoint> seg;
	seg.setOptimizeCoefficients(true);
	seg.setModelType(pcl::SACMODEL_PLANE);
	seg.setMethodType(pcl::SAC_RANSAC);
	seg.setDistanceThreshold(plane_thresh);
	seg.setInputCloud(scene_filtered);
	seg.segment(*inliers, coefficients);
	
	pcl::ExtractIndices<PCLPoint> extract;
	// Extract table
	extract.setInputCloud(scene_filtered);
	extract.setIndices(inliers);
	extract.setNegative(false);
	table_pointcloud->clear();
	extract.filter(*table_pointcloud);

	// Extract objects
	extract.setInputCloud(scene_filtered);
	extract.setIndices(inliers);
	extract.setNegative(true);
	objects_pointcloud->clear();
	extract.filter(*objects_pointcloud);
	
	
	if(objects_pointcloud->points.empty()){
		std::cout << "no objs. leaving" << std::endl;
		return;
	}
	
	std::vector<int> nan_indices;
	pcl::removeNaNFromPointCloud(*objects_pointcloud,*objects_pointcloud,nan_indices); 
	
	std::cout << "original cloud has size " << scene_filtered->points.size() << std::endl;
	std::cout << "passing on objs cloud of size " << objects_pointcloud->points.size() << std::endl;
	
	/*objects_pointcloud->height = 1;
	objects_pointcloud->width = objects_pointcloud->points.size();*/
	
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

// Computes the object centroid based on the pointcloud
void Wrapper::computeCentroid(const PCLPointCloud& cloud, Eigen::Vector3f& centroid){
	Eigen::Vector4f centr;
	pcl::compute3DCentroid(cloud, centr);
	centroid[0] = centr[0];
	centroid[1] = centr[1];
	centroid[2] = centr[2];
}

std::vector<std::pair<std::string, float> > Wrapper::overFeatCallBack(sensor_msgs::Image::Ptr& img_msg, double tp_left_x, 
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
    cv::imshow("before resize", cm_image);
    cv::waitKey();
    //resize the image
    Rect myROI(tp_left_x, tp_left_y, width, width);
    Mat_<Vec3b> im;
  	cm_image = cm_image(myROI);
  	resize(cm_image, im, Size(231, 231), 0, 0, INTER_CUBIC);
    cv::imshow("test", im);
    cv::waitKey(3);
    //cv2TH(im, input);
    //cv2TH(cm_image, input);
    
    //cv::imshow("test_window", im);
    /*cv::imshow("test_window", cm_image);
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
*/
    THTensor_(free)(input);
    THTensor_(free)(probas);
    overfeat::free();
	//return top_classes;
	return empty_res;
  } catch (cv::Exception & e) {
    ROS_INFO("CV Error!");
   	//killed(0);
    return empty_res;
  }		
}

void Wrapper::publishClassification(const sensor_msgs::PointCloud2::ConstPtr &msg){
	// Extract table plane. Return a point cloud of the table and one of the objects.
	PCLPointCloud::Ptr objs_cloud(new PCLPointCloud);
	PCLPointCloud::Ptr table_cloud(new PCLPointCloud);
	filterPointcloud(msg, objs_cloud, table_cloud);
	if(objs_cloud->points.empty()){
		std::cout << "no objs. leaving" << std::endl;
		return;
	}
		
	//publish the table point cloud
	sensor_msgs::PointCloud2::Ptr table_msg(new sensor_msgs::PointCloud2);
	pcl::toROSMsg(*table_cloud, *table_msg);
//	pcl::toROSMsg(*objs_cloud, *table_msg);
	table_pub.publish(*table_msg);

	visualization_msgs::MarkerArray detected_centroids;
	//Cluster the objects
	std::vector<pcl::PointIndices> cluster_indices;
	std::vector<pcl::PointIndices>::const_iterator it;
	clusterPointcloud(objs_cloud, cluster_indices);
	std::cout << "found " << cluster_indices.size() << " clusters" << std::endl;
	std::vector<PCLPointCloud::Ptr> new_clusters; // collect new clusters
	std::vector<Eigen::Vector3f> new_cluster_centroids; // collect new cluster centroids
	int id = 0;
	
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
		//add object centroid to marker array
		visualization_msgs::Marker curr_obj;
		curr_obj.header.frame_id = frame_id;
		curr_obj.header.stamp = ros::Time();
		curr_obj.id = id;
		curr_obj.type = visualization_msgs::Marker::SPHERE;
		curr_obj.action = visualization_msgs::Marker::ADD;
		curr_obj.pose.position.x = new_centr(0);
		curr_obj.pose.position.y = new_centr(1);
		curr_obj.pose.position.z = new_centr(2);
		curr_obj.scale.x = 0.1;
		curr_obj.scale.y = 0.1;
		curr_obj.scale.z = 0.1;
		curr_obj.color.a = 1.0;
		curr_obj.color.r = 1.0;
		curr_obj.color.g = 0.0;
		curr_obj.color.b = 0.0;
		detected_centroids.markers.push_back(curr_obj);
		
		id++;
	}
	
	//publish the detected clusters
	clustered_objs_pub.publish(detected_centroids);
	
	//Get the classification of each clustered object
	overFeatWrapper::detectedObjArray top_classes_msg;
	std::vector<PCLPointCloud::Ptr>::iterator clust_it;
	std::vector<Eigen::Vector3f>::iterator centr_it = new_cluster_centroids.begin();
	int idx = 0;
	for(clust_it = new_clusters.begin(); clust_it != new_clusters.end(); ++clust_it){
		//convert from pointcloud to image
		sensor_msgs::Image::Ptr img_msg (new sensor_msgs::Image);
		std::cout << "converting pcl tp image" << std::endl;
		PCLPointCloud::Ptr clustered_cloud (new PCLPointCloud);
		*clustered_cloud = **clust_it;
		std::cout << "clustered cloud has " << clustered_cloud->points.size() << " points" << std::endl;
		clustered_cloud->header.frame_id = frame_id;
		pcl::toROSMsg(*clustered_cloud, *img_msg);
		temp_pub.publish(*clustered_cloud);
		ros::Duration(2.0).sleep();
		//get coordinates for cropping
		Eigen::Vector3f centr = *centr_it; 
		double tp_left_x = centr(0) - maximum_obj_size;
		double tp_left_y = centr(1) - maximum_obj_size;
		std::cout << "point has centre: x: " << centr(0) << ", y: " << centr(1) << std::endl;
		std::cout << "top left: " << tp_left_x << ", " << tp_left_y << std::endl;
		//call classification method
		std::vector<std::pair<std::string, float> > top_classes = overFeatCallBack(img_msg, tp_left_x, tp_left_y, maximum_obj_size);
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
	classification_pub.publish(top_classes_msg);

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
