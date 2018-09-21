// VideoStitch.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <stdio.h>
#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/xfeatures2d.hpp"


#include <opencv2/opencv.hpp>
#include "opencv2/stitching.hpp"
#include <opencv/cv.h>

#include <opencv2/core/utility.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <ctype.h>

using namespace cv;
using namespace std;
using namespace xfeatures2d;
void readme();


vector<Mat> images1;
vector<Mat> images2;
Mat H;
std::mutex g_io_mutex;
//std::pthread_mutex_t mutex;
void readframe()
{
	int j = 0;
	int64 old = 0;
	// Load the images
	while (true)
	{
		try
		{
			int64 temps = getTickCount();

			cout << "read image time:" << temps - old << endl;
			old = temps;
			j++;
			char iname[256] = { '0' };
			//sprintf_s(iname, "E:\\Video\\Malibu\\Input\\00000\\%05d.jpg", j);
			sprintf_s(iname, "1.jpg", j);
			Mat image1 = imread(iname);
			if (image1.empty())
			{
				return;
			}

			char iname2[256] = { '0' };
			//sprintf_s(iname2, "E:\\Video\\Malibu\\Input\\00003\\%05d.jpg", j);
			sprintf_s(iname2, "2.jpg", j);
			Mat image2 = imread(iname2);
			if (image2.empty())
			{
				return;
			}
			g_io_mutex.lock();
			images1.push_back(image1);
			images2.push_back(image2);
			g_io_mutex.unlock();
			//cam1 >> image1;
			//cam2 >> image2;
			Mat gray_image1;
			Mat gray_image2;
			// Convert to Grayscale
			cvtColor(image1, gray_image1, CV_RGB2GRAY);
			cvtColor(image2, gray_image2, CV_RGB2GRAY);
			namedWindow("first image", WINDOW_NORMAL);
			//resizeWindow("first image", Size(800, 600));
			imshow("first image", image1);
			waitKey(1);
			namedWindow("second image", WINDOW_NORMAL);
			//resizeWindow("second image", Size(800, 600));
			imshow("second image", image2);
			waitKey(1);
			//			continue;
			cout << "Show Images finshed" << endl;
			if (!gray_image1.data || !gray_image2.data)
			{
				std::cout << " --(!) Error reading images " << std::endl;

			}
		}
		catch (...)
		{
				
		}
		
		
	}
}

void output()
{

	while (true)
	{
		Mat image1, image2;
		g_io_mutex.lock();
		if (images1.empty()||images2.empty())
		{
			g_io_mutex.unlock();
			continue;
		}
		image1 = images1.front();
		images1.pop_back();
		image2 = images2.front();
		images2.pop_back();
		g_io_mutex.unlock();

		int temp = getTickCount();
		//-- Step 1: Detect the keypoints using SURF Detector
		int minHessian = 600;
		Ptr<Feature2D>  extractor = SURF::create();
		//SurfFeatureDetector detector(minHessian);

		std::vector< KeyPoint > keypoints_object, keypoints_scene;

		//detector.detect(gray_image1, keypoints_object);
		//detector.detect(gray_image2, keypoints_scene);

		//-- Step 2: Calculate descriptors (feature vectors)
		//SurfDescriptorExtractor extractor;

		Mat descriptors_object, descriptors_scene;
		static bool flag = true;
		//extractor.compute(gray_image1, keypoints_object, descriptors_object);
		//extractor.compute(gray_image2, keypoints_scene, descriptors_scene);
		if (flag)
		{
			extractor->detectAndCompute(image1, Mat(), keypoints_object, descriptors_object);
			extractor->detectAndCompute(image2, Mat(), keypoints_scene, descriptors_scene);
			//-- Step 3: Matching descriptor vectors using FLANN matcher
			FlannBasedMatcher matcher;
			std::vector< DMatch > matches;
			matcher.match(descriptors_object, descriptors_scene, matches);
			int temp3 = getTickCount();
			cout << "feature time：" << temp3 - temp << endl;
			double max_dist = 0; double min_dist = 100;

			//-- Quick calculation of max and min distances between keypoints
			for (int i = 0; i < descriptors_object.rows; i++)
			{
				double dist = matches[i].distance;
				if (dist < min_dist) min_dist = dist;
				if (dist > max_dist) max_dist = dist;
			}

			printf("-- Max dist : %f \n", max_dist);
			printf("-- Min dist : %f \n", min_dist);

			//-- Use only "good" matches (i.e. whose distance is less than 3*min_dist )
			std::vector< DMatch > good_matches;

			for (int i = 0; i < descriptors_object.rows; i++)
			{
				if (matches[i].distance < 3 * min_dist)
				{
					good_matches.push_back(matches[i]);
				}
			}
			std::vector< Point2f > obj;
			std::vector< Point2f > scene;

			for (unsigned int i = 0; i < good_matches.size(); i++)
			{
				//-- Get the keypoints from the good matches
				obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
				scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
			}
			for (int i = 0; i < obj.size(); i++)
			{
				cv::circle(image1, obj[i], 5, Scalar(100, 200, 100));

			}
			imshow("first image", image1);
			waitKey(1);
			for (int i = 0; i < obj.size(); i++)
			{
				cv::circle(image2, scene[i], 5, Scalar(100, 200, 100));

			}
			imshow("second image", image2);
			waitKey(1);
			// Find the Homography Matrix
			H = findHomography(scene, obj, RANSAC);
			flag = false;
		}
		try
		{
			int64 temp2 = getTickCount();
			// Use the Homography Matrix to warp the images
			cv::Mat result;
			//warpPerspective(image1, result, H, cv::Size(image1.cols + image2.cols, max(image1.rows,image2.rows)));
			warpPerspective(image2, result, H, cv::Size(image1.cols + image2.cols, image1.rows + image2.rows));
			//namedWindow("image1", WINDOW_NORMAL);
			////resizeWindow("image1", Size(1200, 800));
			//imshow("image1", result);

			Mat roi = result(Rect(0, 0, image1.cols, image1.rows));
			image1.copyTo(roi);
			int64 temp3 = getTickCount();
			cout << "warpPerspective" << temp3 - temp2 << endl;
			//result = result + image1;
			namedWindow("result2", WINDOW_NORMAL);
			resizeWindow("result2", Size(1200, 800));
			imshow("result2", result);
			//cv::Mat result2;
			//warpPerspective(image1, result2, H, cv::Size(image1.cols + image2.cols, image1.rows));
			//namedWindow("image2", WINDOW_NORMAL);
			////resizeWindow("image1", Size(1200, 800));
			//imshow("image2", result2);


			/*cv::Mat half1(result, cv::Rect(image2.cols, 0, image1.cols, image1.rows));
			image1.copyTo(half1);
			int temp2 = getTickCount();
			cout << "time :" << temp2 - temp3 << endl;
			namedWindow("Result", WINDOW_NORMAL);
			resizeWindow("Result", Size(1200, 800));
			imshow("Result", result);*/
			waitKey(1);
			//break;
		}
		catch (...)
		{

		}
	}

	
}

	/** @function main */
	int main(int argc, char** argv)
	{
		
		std::thread tread(readframe);
		std::thread tstitch(output);
		tread.join();
		tstitch.join();
			
		return 0;
	}

