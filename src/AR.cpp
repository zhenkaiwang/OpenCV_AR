#include <string>
#include <thread>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/nonfree/nonfree.hpp>

using namespace cv;
using namespace std;
double fx,fy,cx,cy,k3;
Mat cameraMatrix = Mat::eye(3,3,CV_64FC1);
Mat distCoef(4,1,CV_64FC1);

Mat rvec = Mat::zeros(3, 1, CV_64FC1);          // output rotation vector
Mat tvec = Mat::zeros(3, 1, CV_64FC1);          // output translation vector
Mat pMatrix = Mat::zeros(3 ,4 , CV_64FC1); 

// RANSAC parameters
int iterationsCount = 1200;        // number of Ransac iterations.
double reprojectionError = 3.0;    // maximum allowed distance to consider it an inlier.
//double confidence = 0.95;          // ransac successful confidence.
int minInliersCount = 50;
//double minInliersCount = 0.95

SurfFeatureDetector detector(400);
SurfDescriptorExtractor extractor;


FlannBasedMatcher matcher;
vector<DMatch> matches;
Mat descriptors1;
Mat descriptors2;
vector<KeyPoint> keypoints1;
vector<KeyPoint> keypoints2; 
Mat H;
vector<Point2f> obj_corners(4);
std::vector<Point2f> scene_corners(4);

cv::Point2f backProject3DPoint(const Point3f &point3d)
{
    // 3D point vector [x y z 1]'
    Mat point3d_vec = cv::Mat(4, 1, CV_64FC1);
    point3d_vec.at<double>(0) = point3d.x;
    point3d_vec.at<double>(1) = point3d.y;
    point3d_vec.at<double>(2) = point3d.z;
    point3d_vec.at<double>(3) = 1.0;
    // 2D point vector [u v 1]'
    cv::Mat point2d_vec = Mat(3, 1, CV_64FC1);
    point2d_vec = cameraMatrix * pMatrix * point3d_vec;
    //cout<<"point2d_vec"<<point3d<<endl;
    // Normalization of [u v]'
    cv::Point2f point2d;
    point2d.x = point2d_vec.at<double>(0) / point2d_vec.at<double>(2);
    point2d.y = point2d_vec.at<double>(1) / point2d_vec.at<double>(2);
    return point2d;
}

void initAR(const string &strSettingPath){
    FileStorage fSettings(strSettingPath, FileStorage::READ);
    if(!fSettings.isOpened())
    {
        cerr << "Failed to open settings file at: " << strSettingPath << endl;
        exit(-1);
    }
    fx = fSettings["Camera.fx"];
    fy = fSettings["Camera.fy"];
    cx = fSettings["Camera.cx"];
    cy = fSettings["Camera.cy"];

    cameraMatrix.at<double>(0,0) = fx;
    cameraMatrix.at<double>(1,1) = fy;
    cameraMatrix.at<double>(0,2) = cx;
    cameraMatrix.at<double>(1,2) = cy;

    distCoef.at<double>(0) = fSettings["Camera.k1"];
    distCoef.at<double>(1) = fSettings["Camera.k2"];
    distCoef.at<double>(2) = fSettings["Camera.p1"];
    distCoef.at<double>(3) = fSettings["Camera.p2"];
    const double k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        distCoef.resize(5);
        distCoef.at<double>(4) = k3;
    }
    //cout<<"cameraMatrix"<<cameraMatrix<<endl;

}


int main(int argc, char* argv[])
{
    if(argc!=2){
        cout << "wrong arguments" << endl;
    }
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened()){  // check if we succeeded
        cout << "camera can't open" <<endl;
        return -1;
    }
    Mat sourceIm = imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
    if(!sourceIm.data){
        cout << "image "<< argv[1] <<" not found" <<endl;
        cout << "usage: ./bin/AugmentedReality images/lena.jpg"<< endl;
        return -1;
    }

    //load camera parameters
    string CalibFile = "TUM1.yaml";
    initAR(CalibFile);

    Mat currentFrame;
    Mat frame_vis;
    namedWindow("Augmented Reality",1);
    //double max_dist = 0; double min_dist = 100;
    for(;;)
    {
        //grab image from camera
        // Mat frame;
        cap >>frame_vis;
        cvtColor(frame_vis,currentFrame,CV_BGR2GRAY);
       // frame_vis = frame.clone();

        //detecting keypoints
        detector.detect(sourceIm, keypoints1);
        detector.detect(currentFrame, keypoints2);

        // computing descriptors
        extractor.compute(sourceIm, keypoints1, descriptors1);
        extractor.compute(currentFrame, keypoints2, descriptors2);

        // matching descriptors
        matcher.match(descriptors1, descriptors2, matches);
        double max_dist = 0; double min_dist = 100;
        for( int i = 0; i < descriptors1.rows; i++ )
        { double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        printf("-- Max dist : %f \n", max_dist );
        printf("-- Min dist : %f \n", min_dist );


        //-- Draw only "good" matches (i.e. whose distance is less than 2.5*min_dist )
        vector< DMatch > good_matches;
        for( int i = 0; i < descriptors1.rows; i++ )
        { if( matches[i].distance < 2.5*min_dist )
            { good_matches.push_back( matches[i]); }
        }

        //-- Localize the object
        vector<Point3f> source3d;
        vector<Point2f> scene;
        Point3f tempP;
        for( int i = 0; i < good_matches.size(); i++ )
        {
            //-- Get the keypoints from the good matches
            tempP.x=keypoints1[ good_matches[i].queryIdx ].pt.x;
            tempP.y=keypoints1[ good_matches[i].queryIdx ].pt.y;
            tempP.z=0.0;
            //cout<<tempP<<endl;
            //cout<<"tempPoint:" << tempP<<endl;
            source3d.push_back(tempP);
            scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
        }


        if(good_matches.size()>0){
            Mat inliers;
            cv::solvePnPRansac( source3d, scene, cameraMatrix, distCoef, rvec, tvec,
                            false, iterationsCount, reprojectionError, minInliersCount,
                            inliers, ITERATIVE);

            Rodrigues(rvec,pMatrix);                   // converts Rotation Vector to Matrix
            //pMatrix.push_back(tvec);
            cout<<"pMatrix:" << pMatrix <<endl;
            hconcat(pMatrix,tvec,pMatrix);
            //cout<<"pMatrix:" << pMatrix <<endl;
            double l = 100.0;
            vector<Point2f> pose_points2d;
            pose_points2d.push_back(backProject3DPoint(Point3f(0.0,0.0,0.0)));    // axis center
            pose_points2d.push_back(backProject3DPoint(Point3f(l,0.0,0.0)));    // axis x
            pose_points2d.push_back(backProject3DPoint(Point3f(l,l,0.0)));    // axis y
            pose_points2d.push_back(backProject3DPoint(Point3f(0.0,l,0.0)));    // axis z
            pose_points2d.push_back(backProject3DPoint(Point3f(0,0.0,-l)));    // axis center
            pose_points2d.push_back(backProject3DPoint(Point3f(l,0.0,-l)));    // axis x
            pose_points2d.push_back(backProject3DPoint(Point3f(l,l,-l)));    // axis y
            pose_points2d.push_back(backProject3DPoint(Point3f(0.0,l,-l)));    // axis z
            //cout<<"axis:" << pose_points2d <<endl;
            //draw3DCoordinateAxes(frame_vis, pose_points2d);                                       // draw axes
            int npt[] = { 4 };
            Point rook_points[1][4];
            rook_points[0][0] = pose_points2d[0];
            rook_points[0][1] = pose_points2d[1];
            rook_points[0][2] = pose_points2d[2];
            rook_points[0][3] = pose_points2d[3];
            const Point* ppt[1] = { rook_points[0] };
            fillPoly( frame_vis,ppt,npt,1,Scalar( 255, 0, 0 ),8);
            line( frame_vis, pose_points2d[0], pose_points2d[4], Scalar( 0, 0, 255), 4 );
            line( frame_vis, pose_points2d[4], pose_points2d[5], Scalar( 0, 255, 0), 4 );
            line( frame_vis, pose_points2d[5], pose_points2d[6], Scalar( 0, 255, 0), 4 );
            line( frame_vis, pose_points2d[6], pose_points2d[7], Scalar( 0, 255, 0), 4 );
            line( frame_vis, pose_points2d[7], pose_points2d[4], Scalar(0, 255, 0), 4 );
            line( frame_vis, pose_points2d[1], pose_points2d[5], Scalar( 0, 0, 255), 4 );
            line( frame_vis, pose_points2d[2], pose_points2d[6], Scalar( 0, 0, 255), 4 );
            line( frame_vis, pose_points2d[3], pose_points2d[7], Scalar(0, 0, 255), 4 );
        }
        imshow("Augmented Reality", frame_vis);
        if(waitKey(30) >= 0) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}

