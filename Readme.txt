%--------------------------------------------------------------------------
Compiling and building from source in Linux environment:

    In a terminal, go to the root directory of the folder and run make:

        ~$ cd OpenVC_AR
        ~/OpenCV_AR$ make

    An executable file AugmentedReality will be generated in the bin folder.

%--------------------------------------------------------------------------
Running the executable:

    In a terminal, at the root directory:

        ~/OpenCV_AR$ ./bin/AugmentedReality images/lena.jpg

    Where images/lena.jpg is the source image that you want to detect. You can use your own image to test.

%--------------------------------------------------------------------------
Code Explanation and Design choice:
	1. Initialzation:

		string CalibFile = "TUM1.yaml";
    	initAR(CalibFile);

    Load the calibration file "TUM1.yaml" of the camera. The parameters includes fx, fy, cx, cy and distortion coefficients. To have better performance, users should make own calibration file. 

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

	2. Detector, Extractor and Matcher:

		SurfFeatureDetector detector(400);
		SurfDescriptorExtractor extractor;
		FlannBasedMatcher matcher;

	I use SURF for detector and extractor with 400 keypoints because it is fast and robust. I choose Flann to match scene discriptors and source discriptors because it is faster than BFMatcher. Then following codes are used to detect keypoints, compute descriptors and match descriptors from source image and current fram from camera.

        detector.detect(sourceIm, keypoints1);
        detector.detect(currentFrame, keypoints2);
        extractor.compute(sourceIm, keypoints1, descriptors1);
        extractor.compute(currentFrame, keypoints2, descriptors2);
        matcher.match(descriptors1, descriptors2, matches);

    Accept a matche as a good match if its distance is less then 2.5*min_dist

        double max_dist = 0; double min_dist = 100;
        for( int i = 0; i < descriptors1.rows; i++ )
        { double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        vector< DMatch > good_matches;
        for( int i = 0; i < descriptors1.rows; i++ )
        { if( matches[i].distance < 2.5*min_dist )
            { good_matches.push_back( matches[i]); }
        }


	3. Pose Estimation using PnP + Ransac
	First find the the correspondences of source image and current frame. Save source image as 3D point by setting z to be 0.

	    vector<Point3f> source3d;
        vector<Point2f> scene;
        Point3f tempP;
        for( int i = 0; i < good_matches.size(); i++ )
        {
            tempP.x=keypoints1[ good_matches[i].queryIdx ].pt.x;
            tempP.y=keypoints1[ good_matches[i].queryIdx ].pt.y;
            tempP.z=0.0;
            source3d.push_back(tempP);
            scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
        }

    Use solvePnPRansac to estimate the camera pose and get the rotation matrix and transaltion vector. I use ITERATIVE as the estimation method instead of EPNP and P3P because ITERATIVE is more robust for pose estimation of a planar object.

        solvePnPRansac( source3d, scene, cameraMatrix, distCoef, rvec, tvec,
                            false, iterationsCount, reprojectionError, minInliersCount,
                            inliers, ITERATIVE);
        Rodrigues(rvec,pMatrix);
        hconcat(pMatrix,tvec,pMatrix);

    Other parameters are set as follows:

	    int iterationsCount = 1200;
		double reprojectionError = 3.0;
		int minInliersCount = 50;


	4. 3D Object Drawing
	I use a backProjection3Dpoint() function to project a 3d coordinates into a 2d coordinates through camera parameter matrix and transformation matrix:

		Point2f backProject3DPoint(const Point3f &point3d)
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

	Then draw the cube of length 100 at the origin of the planar object and fill its bottom surface:

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





