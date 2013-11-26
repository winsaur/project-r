/*
 * types.h
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

#ifndef TYPES_H_
#define TYPES_H_
#include <utility>
#include <vector>
#include <boost/shared_ptr.hpp>

#include <opencv2/opencv.hpp>

class BlobDescriptor;

typedef std::vector<cv::Point> Contour;
typedef Contour::iterator ContourIt;
typedef Contour::const_iterator ContourConstIt;
typedef boost::shared_ptr<Contour> ContourPtr;
typedef std::vector<Contour> ContourList;
typedef ContourList::iterator ContourListIt;
typedef ContourList::const_iterator ContourListConstIt;



typedef std::pair<cv::Scalar,cv::Scalar> ColorPair;

typedef boost::shared_ptr<BlobDescriptor> BlobDescriptorPtr;
typedef std::vector<BlobDescriptorPtr> BlobDescriptorPtrVector;
typedef BlobDescriptorPtrVector::iterator BlobDescriptorPtrVectorIt;
typedef BlobDescriptorPtrVector::const_iterator BlobDescriptorPtrVectorConstIt;

typedef std::vector<cv::Point2f> Point2fVec;
typedef std::map<int,Point2fVec> Point2fVecMap;
typedef std::pair<int,Point2fVec > Point2fVecPair;
typedef Point2fVecMap::iterator Point2fVecMapIt;


#endif /* TYPES_H_ */
