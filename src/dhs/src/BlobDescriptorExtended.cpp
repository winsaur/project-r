/*
 * BlobDescriptorExtended.cpp
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

#include "BlobDescriptorExtended.h"
#include "Utility.h"
#include "Types.h"

static const int kMinContourArea(100);

bool isContourSmall(const Contour& rhs) {
	return cv::contourArea(rhs) < kMinContourArea;
}

void BlobDescriptorExtended::setupPeriodic(const cv::Mat& template_view) {
	//store its template
	template_view_ = template_view;
	//store maximum similarity
	backpack_similarities_.clear();
	backpack_similarities_.push_back(0);
	briefcase_similarities_.clear();
	briefcase_similarities_.push_back(0);

	//store its current location
	initial_bound_ = LastRawBound();

	//store the points used for affine warp: center, upper left, lower left
	src_points_.clear();
	src_points_.push_back(getCentroid());
	src_points_.push_back(LastRawBound().tl());
	src_points_.push_back(utility::BottomLeft(LastRawBound()));
}
void BlobDescriptorExtended::addFrame(const cv::Mat& frame) {
	//get just the blob in the image
	cv::Mat warped_blob_cropped(frame,cv::Rect(initial_bound_));
	imshow("1current blob warped",warped_blob_cropped);

	//segment the nonsymmetric regions into hypothesis regions
	cv::Rect backpack_region(0,
							 initial_bound_.height*0.1,
							 initial_bound_.width,
							 initial_bound_.height*0.4);
	cv::Mat backpack_candidate(warped_blob_cropped,backpack_region);
	cv::Mat backpack_template(template_view_,backpack_region);

	cv::Rect briefcase_region(0,
							 initial_bound_.height*0.5,
							 initial_bound_.width,
							 initial_bound_.height*0.5);
	cv::Mat briefcase_candidate(warped_blob_cropped,briefcase_region);
	cv::Mat briefcase_template(template_view_,briefcase_region);


/*	//get contours in backpack and briefcase regions
	ContourList backpack_regions,briecase_regions;
	cv::findContours(backpack_candidate,backpack_regions,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);
	cv::findContours(briefcase_candidate,briecase_regions,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);

	//remove small contours
	backpack_regions.erase(std::remove_if(backpack_regions.begin(),backpack_regions.end(),isContourSmall),
			backpack_regions.end());
	briecase_regions.erase(std::remove_if(briecase_regions.begin(),briecase_regions.end(),isContourSmall),
			briecase_regions.end());
*/
	//compute similarity of backpack and briefcase regions

	cv::Mat backpack_difference = backpack_candidate-backpack_template | backpack_template-backpack_candidate;
	cv::Mat briefcase_difference = briefcase_candidate-briefcase_template | briefcase_template-briefcase_candidate;

	//count the nonblack pixels in the result
	int backpack_nonzero = cv::countNonZero(backpack_difference)+1;
	int briefcase_nonzero = cv::countNonZero(briefcase_difference)+1;
	backpack_similarities_.push_back((double)1./backpack_nonzero);
	briefcase_similarities_.push_back((double)1./briefcase_nonzero);

	utility::visualizeVector("backpack_similarities",backpack_similarities_);
	utility::visualizeVector("briefcase_similarities_",briefcase_similarities_);

	cv::Mat backpack_similarities_mat(1,backpack_similarities_.size(),CV_32FC2,cv::Scalar::all(0));
	cv::Mat briefcase_similarities_mat(1,briefcase_similarities_.size(),CV_32FC2,cv::Scalar::all(0));

	//copy the data to the mat
	for(int i=0;i<backpack_similarities_.size();i++) {
		backpack_similarities_mat.at<cv::Vec2f>(0,i)[0] = (float)backpack_similarities_[i];
		briefcase_similarities_mat.at<cv::Vec2f>(0,i)[0] = (float)briefcase_similarities_[i];
	}
	cv::dft(backpack_similarities_mat,backpack_similarities_mat);
	cv::dft(briefcase_similarities_mat,briefcase_similarities_mat);

	//center data
	int center = backpack_similarities_mat.cols;
	std::cout<<"center: "<<center<<std::endl;
	cv::Mat left(backpack_similarities_mat,cv::Rect(0,0,center/2,1));
	cv::Mat right(backpack_similarities_mat,cv::Rect(center/2,0,center/2-1,1));
	cv::Mat temp;
	right.copyTo(temp);
	left.copyTo(right);
	temp.copyTo(left);

	cv::Mat left2(briefcase_similarities_mat,cv::Rect(0,0,center/2,1));
	cv::Mat right2(briefcase_similarities_mat,cv::Rect(center/2,0,center/2-1,1));
	right2.copyTo(temp);
	left2.copyTo(right2);
	temp.copyTo(left2);

	std::cout<<"backpack similarities fourier: "<<backpack_similarities_mat<<std::endl;
	std::cout<<"briefcase similarities fourier: "<<briefcase_similarities_mat<<std::endl;


	//calculate magnitude of both FT's
	std::vector<float> backpack_magnitude,briefcase_magnitude;
	for(int i=0;i<backpack_similarities_.size();i++) {
		backpack_magnitude.push_back(
				pow(backpack_similarities_mat.at<cv::Vec2f>(0,i)[0],2) +
				pow(backpack_similarities_mat.at<cv::Vec2f>(0,i)[1],2));
		briefcase_magnitude.push_back(
						pow(briefcase_similarities_mat.at<cv::Vec2f>(0,i)[0],2) +
						pow(briefcase_similarities_mat.at<cv::Vec2f>(0,i)[1],2));


	}



	utility::visualizeVector("breifcase fourier",briefcase_magnitude);
	utility::visualizeVector("backpack fourier",backpack_magnitude);

}

BlobDescriptorExtendedFetcher::BlobDescriptorExtendedFetcher(const std::string& topic) {
	blob_subscriber_ = handle_.subscribe(topic,100,&BlobDescriptorExtendedFetcher::receiver,this);
}
std::string BlobDescriptorExtendedFetcher::getTopic() {
	return blob_subscriber_.getTopic();
}

void BlobDescriptorExtendedFetcher::receiver(const dhs::blob& msg) {
	//deserialize the message into the map
	//get colors
	ColorPair colors;
	if(msg.colors.size() >=3) {
		colors.first[0] = msg.colors[0];
		colors.first[1] = msg.colors[1];
		colors.first[2] = msg.colors[2];
	}
	if(msg.colors.size() >=6) {
		colors.second[0] = msg.colors[3];
		colors.second[1] = msg.colors[4];
		colors.second[2] = msg.colors[5];
	}

	//get position/dimensions
	cv::Rect position(msg.filtered_position[0],msg.filtered_position[1],msg.filtered_size[0],msg.filtered_size[1]);
	BlobDescriptorExtendedPtr new_blob(new BlobDescriptorExtended(msg.first_seen,msg.id,colors,position,msg.depth,NULL));
	std::pair<int,BlobDescriptorExtendedPtr> insert_val(msg.id,new_blob);
	std::pair<std::map<int,BlobDescriptorExtendedPtr>::iterator,bool > returned = data_.insert(insert_val);
	//if returned.second then we're done, otherwise need to update the existing blob
	if(!returned.second) {
		returned.first->second->history_.push_back(HistoryDescriptor(msg.last_seen,-1,position));
		returned.first->second->colors_ = colors;
	}

	//deserialize contour
	utility::deSerializeContour(msg.contour,returned.first->second->contour_);

	//get centroid
	returned.first->second->centroid_ = cv::Point2f(msg.centroid[0],msg.centroid[1]);

	//add this id to the blobs_updated array so something else can go do processing on it
	blobs_updated_.push_back(msg.id);
}
