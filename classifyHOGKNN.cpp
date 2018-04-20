#include "opencv/cv.h" 
#include <opencv2\opencv.hpp>  
#include <stdio.h>  
#include <iostream> 
#include <opencv2/highgui/highgui.hpp>  
#include <opencv/ml.h>   
#include <iostream>    
#include <fstream>    
#include <string>    
#include <vector>  
using namespace cv;    
using namespace std;
Mat MoveDetect(Mat background, Mat frame);
Point originalPoint(600, 100); //���ο����  
Point processPoint(1200, 700);
int ImgWidht=28; 
int ImgHeight=28; 
CvKNearest knn;
int FindK=3;  //KNN�㷨����5�������
int choose;
void show();
void train(string s);
int test(Mat s);

int main(int argc, char** argv)      
{
	int flag = 0;
	train("train_car.txt");
	VideoCapture capture("234.mp4");
	int m, n, j;
	m = n = j = 0;
	int dx = 0;
	char ad[128] = { 0 };
	Mat background;//�洢����ͼ��  
	Mat result;//�洢���ͼ��  
	while(1)
	{
		Mat frame;//����һ��Mat���������ڴ洢ÿһ֡��ͼ��
		capture >> frame;  //��ȡ��ǰ֡
		rectangle(frame, originalPoint, processPoint, Scalar(255, 0, 0), 3, 4, 0);
		Mat imageROI = frame(Rect(600, 100, 600, 600));  //(600, 100, 600, 600)
		imshow("��ȡ��Ƶ", frame);  //��ʾ��ǰ֡
		int a = test(imageROI);
		if (a == 0){
			n++;
			flag = 1;
			j = 1;
			imshow("roi", imageROI);
			sprintf_s(ad, "E:\\_Data\\ROI%d.jpg", ++dx);
			imwrite(ad, imageROI);
		}
		else{
			m++;
			if (m > 10){
				m = 0;
				n = 0;
				
			}
			if (flag == 1){

				background = imageROI;
				flag++;
			}
			
		}
		cout << a << endl;
		waitKey(30);  //��ʱ30ms
	}       
    return 0;    
} 
 
void train(string filename)
{
	vector<string> src_img_path;    //ͼƬ����
    vector<int> img_class;          //ͼƬ�������  0����������  1��������
	ifstream svm_train_img(filename); 
	string templine;     
    int trainline = 0;       
    unsigned long n;    
    while(svm_train_img)      //��ȡ�ļ�������vector��
    {    
        if(getline(svm_train_img,templine))    
        {    
            trainline++;    
            if(trainline<201)     //ǰ��201����������������Ϊ0
            {    
                img_class.push_back(0);  
                src_img_path.push_back(templine);//ͼ��·��   
            }    
            else             //֮����Ǹ�����������Ϊ1
            {    
                img_class.push_back(1);  
                src_img_path.push_back(templine);//ͼ��·��   
            }    
        }    
    }    
    svm_train_img.close();    //�ر��ļ�    

    Mat train_hog, train_class;        
    train_class=Mat::zeros(trainline,1,CV_32FC1 );      //����������
	cout<<"��ȡֱ��ͼ����!"<<endl;
    for(string::size_type i=0;i!=src_img_path.size();i++)    
    {    
		cout<<src_img_path[i].c_str()<<endl; 
        Mat src_img=imread(src_img_path[i].c_str(), 1);      
//		cout<<src_img<<endl;
        resize(src_img,src_img,cv::Size(ImgWidht,ImgHeight),0,0,INTER_CUBIC);   //ͼƬ��С����
		cout<<"��ȡͼƬ�����¶����С���!"<<endl;
        HOGDescriptor *hog=new HOGDescriptor(cvSize(ImgWidht,ImgHeight),cvSize(14,14),cvSize(7,7),cvSize(7,7), 9);  //��ȡֱ��ͼ����
		//��һ�������Ǵ��ڴ�С���ڶ����ǿ��С���������ǿ鲽�������ĸ��ǰ�Ԫ��С
		//64*128 16*16 8*8
		//����64-16��/8+1��*����126-16��/8+1��=105
		//(16/8)*(16/8)=4����Ԫ
		//9ά
		//��9*4*105������
        vector<float> hog_result;     
        hog->compute(src_img,hog_result,Size(1,1),Size(0,0)); //ֱ��ͼ����
		cout<<"�ݶ���ɫֱ��ͼ�������!"<<endl;
        if (i==0)    //��һ��ͼƬ��ʱ�����ֱ��ͼMat,ÿһ�д���һ������
        {  
             train_hog = Mat::zeros(trainline,hog_result.size(),CV_32FC1 );  
			 cout<<"��ʼ�������洢����!"<<endl;
        }     
        n=0;    
        for(vector<float>::iterator iter=hog_result.begin();iter!=hog_result.end();iter++)     //���õ���������Mat���鸳ֵ
        {    
            train_hog.at<float>(i,n)=*iter;    
            n++;    
        }      
		cout<<"����Ԫ����ȡ���!"<<endl;
        train_class.at<float>(i,0)=img_class[i];     
    }             
	cout<<"**************************"<<endl;
	cout<<"��ʼѵ��KNN!"<<endl;

    knn.train(train_hog,train_class,Mat(),false,FindK);  

	cout<<"ѵ������!"<<endl;
	cout<<"**************************"<<endl;
}
int test(Mat ROI)
{
	 
    char line[512];    
    ofstream predict("predict_data.txt");    
    //Mat src_image=imread(img_test_path[j].c_str(),1);//����ͼ��     
	Mat src_image = ROI;
    resize(src_image, src_image, cv::Size(ImgWidht,ImgHeight), 0, 0, INTER_CUBIC);//Ҫ���ͬ���Ĵ�С�ſ��Լ�⵽ 

    HOGDescriptor *hog=new HOGDescriptor(cvSize(ImgWidht,ImgHeight),cvSize(14,14),cvSize(7,7),cvSize(7,7),9); 

    vector<float> hog_result;//�������    

    hog->compute(src_image,hog_result,Size(1,1),Size(0,0));      
    Mat test_hog=Mat::zeros(1,hog_result.size(),CV_32FC1); 
    int n=0;
    for(vector<float>::iterator iter=hog_result.begin();iter!=hog_result.end();iter++)    
       {    
            test_hog.at<float>(0,n) = *iter;    
            n++;    
        }    
        int result = knn.find_nearest(test_hog,FindK);     //����֧������������
	    //cout<<"The perdict class: "<<result; 
		return int(result);
}

Mat MoveDetect(Mat background, Mat frame)
{
	Mat result = frame.clone();
	//1.��background��frameתΪ�Ҷ�ͼ  
	Mat gray1, gray2;
	cvtColor(background, gray1, CV_BGR2GRAY);
	cvtColor(frame, gray2, CV_BGR2GRAY);
	//2.��background��frame����  
	Mat diff;
	//GaussianBlur(background, background, Size(21, 21), 0, 0);
	//GaussianBlur(frame, frame, Size(21, 21), 0, 0);
	absdiff(gray1, gray2, diff);
	//imshow("diff", diff);
	//3.�Բ�ֵͼdiff_thresh������ֵ������  
	Mat diff_thresh;
	threshold(diff, diff_thresh, 50, 255, CV_THRESH_BINARY);
	//imshow("diff_thresh", diff_thresh);
	//4.��ʴ  
	Mat kernel_erode = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat kernel_dilate = getStructuringElement(MORPH_RECT, Size(15, 15));
	erode(diff_thresh, diff_thresh, kernel_erode);
	//imshow("erode", diff_thresh);
	//5.����  
	dilate(diff_thresh, diff_thresh, kernel_dilate);
	//imshow("dilate", diff_thresh);
	//6.������������������  
	vector<vector<Point>> contours;
	findContours(diff_thresh, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	drawContours(result, contours, -1, Scalar(0, 0, 255), 2);//��result�ϻ�������  
	//7.��������Ӿ���  
	vector<Rect> boundRect(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		boundRect[i] = boundingRect(contours[i]);
		rectangle(result, boundRect[i], Scalar(0, 255, 0), 2);//��result�ϻ�������Ӿ���  
	}
	return result;//����result  
}