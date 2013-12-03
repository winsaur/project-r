/*
 * BagOnPersonProcessor.cpp
    Copyright (C) 2013  Timothy Sweet

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

//std includes
#include <sstream>

//library includes
#include <ros/ros.h>

//project includes
#include "../include/Utility.h"
#include "../include/ProcessorNode.h"
#include "../include/EventCodes.h"
#include "../include/BlobDescriptorExtended.h"

class BagOnPersonProcessor : public ProcessorNode {
public:
	BagOnPersonProcessor(const std::string& blob_topic,const std::string& event_topic) :
		input_stream_(blob_topic) {
		event_name_ = BagOnPersonName;
		event_code_ = BagOnPersonCode;
		output_topic_ = event_topic;
		init();
		std::cout<<"subscribed to: "<<input_stream_.getTopic()<<std::endl;
	}
	void callback(const ros::TimerEvent& event);

	void init();
private:
	DISALLOW_COPY_AND_ASSIGN(BagOnPersonProcessor);
	DISALLOW_DEFAULT_CONSTRUCTION(BagOnPersonProcessor);
	BlobDescriptorExtendedFetcher input_stream_;

};

int main(int argc, char* argv[]) {
	//connect to ros
	ros::init(argc, argv, "segmentation");
	//set logger level
	utility::setLoggerDebug();
	ros::NodeHandle handle("~");

	BagOnPersonProcessor processor("blobs_in","events_out");
	//loop at 60 hz just because that seems like a pretty good rate given
	//that the camera is running at 30 hz
	ros::Timer timer = handle.createTimer(ros::Duration(1./30.), &BagOnPersonProcessor::callback, &processor);

	ros::spin();

	return 0;
}

void BagOnPersonProcessor::init() {
	setupOutput();
}

void BagOnPersonProcessor::callback(const ros::TimerEvent& event) {
	//for each blob which was updated this frame
	if(input_stream_.blobs_updated_.size()==0) {
		return;
	}
	//TODO: get the size another way
	cv::Mat output(cv::Size(640,480),CV_8UC1,cv::Scalar(0));
	std::vector<int>::iterator blob_iterator = input_stream_.blobs_updated_.begin();
	for(;blob_iterator!=input_stream_.blobs_updated_.end();) {
		//just get this blob as a pointer
		BlobDescriptorExtendedPtr current_blob = input_stream_.data_[*blob_iterator];
		/*
		 * erase this index from the updated list and reset the iterator. This prevents
		 * jumping past the end of the vector when incrementing the iterator
		 */
		input_stream_.blobs_updated_.erase(blob_iterator);

		//TODO: fix the race condition here in which an empty list which
		//has a blob added to it after here will break the loop
		blob_iterator = input_stream_.blobs_updated_.begin();

		//draw the contour on a mat
		//TODO:get the size another way
		cv::Mat blob_visual(cv::Size(640,480),CV_8UC1,cv::Scalar(0));
		{
			ContourList temp_list;
			temp_list.push_back(current_blob->contour_);
			cv::drawContours(blob_visual,temp_list,0,cv::Scalar(255),-1);
		}

		//compute the body axis
		//try computing it as the horizontal projection(as in sum of vertical stacks of pixels)
		utility::floodConcaveRegions(&blob_visual);
		int symmetry_axis;
		{
			std::vector<int> upward_projection;
			utility::getUpwardProjection(blob_visual,&upward_projection);
			int max=0;
			int max_projection_at=0;
			for(int i=0;i<upward_projection.size();i++) {
				if(upward_projection[i] > max) {
					max = upward_projection[i];
					max_projection_at=i;
				}
			}
			//draw max horizontal projection axis
			//cv::line(blob_visual,cv::Point(max_at,0),cv::Point(max_at,480),cv::Scalar(128+64),2);
			symmetry_axis = max_projection_at;
		}


		cv::Mat full_blob(blob_visual.clone());

		//classify each pixel as symmetric or asymmetric
		utility::removeSymmetricRegions(symmetry_axis,&blob_visual);
		utility::recolorNonSymmetricRegions(symmetry_axis,&full_blob); //this is only for the user's visualization

		//show the symmetry line just for the user's edification
		cv::line(full_blob,cv::Point(symmetry_axis,0),cv::Point(symmetry_axis,480),cv::Scalar(200),2);

		//show which regions are symmetric/asymmetric
		cv::imshow("symetric/aysmetric",full_blob);

		//check periodicity and reclassify pixels if necessary
		//algorithm: if there are no similarity measurements for this blob, it must be the first time it's been seen
		if(current_blob->backpack_similarities_.empty()) {
			//this is the first time the blob has been seen
			current_blob->setupPeriodic(blob_visual);
		} else if(utility::changedMoreThanFactor( //the bound has changed a lot since first view, so reset all the periodicity tracking stuff
				current_blob->initial_bound_,
				current_blob->LastRawBound(), 1)) {
			ROS_INFO_STREAM("Blob #"<<current_blob->Id()<<" changed a lot, resetting its track");
			//the blob changed a lot so dump everything and start over
			//this typcially occurs a few times as the blob is entering the scene
			current_blob->setupPeriodic(blob_visual);
		} else {
			//blob has already been seen, just check for periodicity
			Point2fVec dst_points;
			dst_points.push_back(current_blob->getCentroid());
			dst_points.push_back(current_blob->LastRawBound().tl());
			dst_points.push_back(utility::BottomLeft(current_blob->LastRawBound()));

			//get an affine transformation matrix which maps the current view to the template view
			cv::Mat warp_mat = cv::getAffineTransform(dst_points,current_blob->src_points_);

			//warp the visual representation of the non symmetric parts of the blob to the reference frame (ideally the first full view of the blob)
			cv::Mat warped_blob;
			cv::warpAffine(blob_visual,warped_blob,warp_mat,blob_visual.size());


			current_blob->addFrame(warped_blob);

		}

		//store results
	}
	cv::waitKey(1);
}
