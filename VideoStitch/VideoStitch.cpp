// VideoStitch.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include <io.h>
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
#include <mutex>
#include <thread>
using namespace cv;
using namespace std;
using namespace xfeatures2d;
vector<Mat> images1;
vector<Mat> images2;
Mat H;
std::mutex g_io_mutex;

typedef struct
{
	Point2f left_top;
	Point2f left_bottom;
	Point2f right_top;
	Point2f right_bottom;
}four_corners_t;
four_corners_t corners;

void readme();
int file_exists(char *filename)
{
	return (_access(filename, 0) == 0);
}

//计算变形后的图像四个角位置
void CalcCorners(const Mat& H, const Mat& src)
{
	double v2[] = { 0, 0, 1 };//左上角
	double v1[3];//变换后的坐标值
	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量

	V1 = H * V2;
	//左上角(0,0,1)
	cout << "V2: " << V2 << endl;
	cout << "V1: " << V1 << endl;
	corners.left_top.x = v1[0] / v1[2];
	corners.left_top.y = v1[1] / v1[2];

	//左下角(0,src.rows,1)
	v2[0] = 0;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.left_bottom.x = v1[0] / v1[2];
	corners.left_bottom.y = v1[1] / v1[2];

	//右上角(src.cols,0,1)
	v2[0] = src.cols;
	v2[1] = 0;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.right_top.x = v1[0] / v1[2];
	corners.right_top.y = v1[1] / v1[2];

	//右下角(src.cols,src.rows,1)
	v2[0] = src.cols;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.right_bottom.x = v1[0] / v1[2];
	corners.right_bottom.y = v1[1] / v1[2];

}
//根据距离，计算权重，消除拼接处痕迹
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst)
{
	int start = MIN(corners.left_top.x, corners.left_bottom.x);;//开始位置，即重叠区域的左边界  

	double processWidth = img1.cols - start;//重叠区域的宽度  
	int rows = dst.rows;
	int cols = img1.cols; //注意，是列数*通道数
	double alpha = 1;//img1中像素的权重  
	for (int i = 0; i < rows; i++)
	{
		uchar* p = img1.ptr<uchar>(i);  //获取第i行的首地址
		uchar* t = trans.ptr<uchar>(i);
		uchar* d = dst.ptr<uchar>(i);
		for (int j = start; j < cols; j++)
		{
			//如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
			if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
			{
				alpha = 1;
			}
			else
			{
				//img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好  
				alpha = (processWidth - (j - start)) / processWidth;
			}

			d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
			d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
			d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);

		}
	}

}

//std::pthread_mutex_t mutex;
void readframe()
{
	int j = 0;
	int64 old = 0;
	// Load the images
	cv::VideoCapture capture("2.mp4");
	cv::VideoCapture capture2("1.mp4");
	/*cv::VideoCapture capture("result2.avi");
	cv::VideoCapture capture2("result1.avi");*/
	Mat image1,image2;
	while (true)
	{
		try
		{
			int64 temps = getTickCount();

			cout << "read image time:" << temps - old << endl;
			old = temps;
			j++;
			//char iname[256] = { '0' };
			//sprintf_s(iname, "E:\\ImageProcess\\NISwGSP-master\\NISwGSP\\input-42-data\\AANAP-building\\building0.png", j);
			//sprintf_s(iname, "1.jpg", j);
			//Mat image1 = imread(iname);
			capture >> image1;
			if (image1.empty())
			{
				return;
			}

			//char iname2[256] = { '0' };
			//sprintf_s(iname2, "E:\\ImageProcess\\NISwGSP-master\\NISwGSP\\input-42-data\\AANAP-building\\building1.png", j);
			//sprintf_s(iname2, "2.jpg", j);
			//Mat image2 = imread(iname2);
			capture2 >> image2;
			if (image2.empty())
			{
				return;
			}
			g_io_mutex.lock();
			
			int width = image1.cols;
			int height = image1.rows;

			cv::Mat imag1 = Mat::zeros( height / 2, width / 2, CV_8UC3);// = cv::resize(image1, (width / 2, height / 2), cv::INTER_CUBIC);
			cv::Mat imag2 = Mat::zeros(height / 2, width / 2, CV_8UC3); //cv::resize(image2, (width / 2, height / 2), cv::INTER_CUBIC);
			resize(image1, imag1, imag1.size());
			resize(image2, imag2, imag2.size());
			
			images1.push_back(imag1);
			images2.push_back(imag2);
			g_io_mutex.unlock();
			
			Mat gray_image1;
			Mat gray_image2;
			// Convert to Grayscale
			cvtColor(imag1, gray_image1, CV_RGB2GRAY);
			cvtColor(imag2, gray_image2, CV_RGB2GRAY);
		/*	imag1.copyTo(gray_image1);
			imag2.copyTo(gray_image2);*/
			waitKey(25);

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
				if (!file_exists("H_martix.txt"))
				{
					extractor->detectAndCompute(gray_image1, Mat(), keypoints_object, descriptors_object);
					extractor->detectAndCompute(gray_image2, Mat(), keypoints_scene, descriptors_scene);
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
						cv::circle(gray_image1, obj[i], 5, Scalar(100, 200, 100));

					}
					imshow("first image", gray_image1);
					waitKey(1);
					for (int i = 0; i < obj.size(); i++)
					{
						cv::circle(gray_image2, scene[i], 5, Scalar(100, 200, 100));

					}
					imshow("second image", gray_image2);
					waitKey(1);
					// Find the Homography Matrix
					H = findHomography(obj, scene, RANSAC);
					CalcCorners(H, gray_image1);
					cout << H << endl;
					fstream f("H_martix.txt", ios::out);
					for (int i = 0; i<H.rows; i++)
					{
						for (int j = 0; j<H.cols; j++)
						{
							double value = H.at<double>(i, j); //写入数据
							cout << value << endl;;
							f << value << "\n";
						}

					}
					f.close();

					cout << "left_top:" << corners.left_top << endl;
					cout << "left_bottom:" << corners.left_bottom << endl;
					cout << "right_top:" << corners.right_top << endl;
					cout << "right_bottom:" << corners.right_bottom << endl;
					flag = false;
				}
				else
				{
					H.zeros(cv::Size(3, 3), CV_64FC1);
					ifstream ifile1("H_martix.txt", ios::in);
					if (!ifile1)
						cerr << "error_O_L" << endl;
					string lineword;
					int j = 0;
					//std::vector<std::vector<double>> vec;
					std::vector<double> vectemp;
					while (ifile1 >> lineword)
					{
						double temp = atof(lineword.c_str());
						vectemp.push_back(temp);
						j++;
					}
					cv::Mat H1 = cv::Mat{ vectemp };
					H = H1.reshape(0, 3).clone();
					CalcCorners(H, gray_image1);
					ifile1.close();
				}
				cout << H << endl;
				flag = false;
			}
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
	VideoWriter writer;
	while (true)
	{
		Mat img1, img2;
		g_io_mutex.lock();
		if (images1.empty()||images2.empty())
		{
			g_io_mutex.unlock();
			continue;
		}
		img1 = images1.front();		
		images1.erase(images1.begin());
		img2 = images2.front();
		images2.erase(images2.begin());
		g_io_mutex.unlock();
		try
		{

			int64 temp2 = getTickCount();
			// Use the Homography Matrix to warp the images
			Mat imageTransform1, imageTransform2;
			warpPerspective(img1, imageTransform1, H, Size(MAX(corners.right_top.x, corners.right_bottom.x), img2.rows));
			//warpPerspective(image01, imageTransform2, adjustMat*homo, Size(image02.cols*1.3, image02.rows*1.8));
			imshow("直接经过透视矩阵变换", imageTransform1);
			imwrite("trans1.jpg", imageTransform1);
			waitKey(5);

			//创建拼接后的图,需提前计算图的大小
			int dst_width = imageTransform1.cols;  //取最右点的长度为拼接图的长度
			int dst_height = img2.rows;

			Mat dst(dst_height, dst_width, CV_8UC3);
			dst.setTo(0);

			imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
			img2.copyTo(dst(Rect(0, 0, img2.cols, img2.rows)));

			imshow("b_dst", dst);
			OptimizeSeam(img2, imageTransform1, dst);
			imshow("dst", dst);
			imwrite("dst.jpg", dst);
			static bool flag = true;
			if (flag)
			{
				writer = VideoWriter("result.avi", CV_FOURCC('M', 'J', 'P', 'G'), 25, dst.size());
				flag = false;
			}
			writer << dst;
			waitKey(5);
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

