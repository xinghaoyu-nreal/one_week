#include "opencv_utility.h"

int COL, ROW;
float FX, FY, CX, CY;
float K1, K2, P1, P2;

void readParam(const std::string &dir_path)
{
    const std::string &config_file = dir_path + "sensor.yaml";
    std::cout << config_file.c_str() << std::endl;

    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings yaml\n";
        return ;
    }

    cv::Mat CV_T, INTRINSICS, DIST, RESOLUTION;
    fsSettings["T_BS"] >> CV_T;
    fsSettings["intrinsics"] >> INTRINSICS;
    fsSettings["distortion_coefficients"] >> DIST;
    fsSettings["resolution"] >> RESOLUTION;

    Eigen::Matrix<double, 3, 4, Eigen::RowMajor> eigen_T;
    cv::cv2eigen(CV_T, eigen_T);
    Eigen::Matrix3d eigen_R = eigen_T.topLeftCorner(3, 3);
    Eigen::Vector3d eigen_t = eigen_T.topRightCorner(3, 1);

    FX = INTRINSICS.at<double>(0, 0);
    FY = INTRINSICS.at<double>(0, 1);
    CX = INTRINSICS.at<double>(0, 2);
    CY = INTRINSICS.at<double>(0, 3);

    K1 = DIST.at<double>(0, 0);
    K2 = DIST.at<double>(0, 1);
    P1 = DIST.at<double>(0, 2);
    P2 = DIST.at<double>(0, 3);

    COL = RESOLUTION.at<double>(0, 0);
    ROW = RESOLUTION.at<double>(0, 1);

    // std::cout << eigen_R << "\n" << eigen_t.transpose() << std::endl;
    // std::cout << FX << " " << FY << " " << CX << " " << CY << std::endl; 
    // std::cout << K1 << " " << K2 << " " << P1 << " " << P2 << std::endl; 

    fsSettings.release();
    return ;
}

void imgUtility(const std::string &dir_path)
{
    cv::Mat img_raw;
    const std::string &img_path = dir_path + "1403637188088318976.png";
    img_raw = cv::imread(img_path, cv::IMREAD_GRAYSCALE);

    cv::Mat img_combine1(ROW, COL * 2, img_raw.type());
    img_raw.copyTo(img_combine1(cv::Rect(0, 0, COL, ROW)));

    /*****像素取反*****/
    cv::Mat img_inverse(ROW, COL, img_raw.type());
    for (int i = 0, iend = img_raw.rows; i < iend; i++)
        for (int j = 0, jend = img_raw.cols; j < jend; j++)
            img_inverse.at<uchar>(i, j) = 255 - img_raw.at<uchar>(i, j);
    img_inverse.copyTo(img_combine1(cv::Rect(COL, 0, COL, ROW)));

    cv::putText(img_combine1,"raw img", cv::Point(40, 40), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255,23,0), 4, 8);
    cv::putText(img_combine1,"inverse img", cv::Point(40 + COL, 40), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255,23,0), 4, 8);
    cv::imwrite(dir_path + "inverse.png", img_combine1);

    /*****去畸变*****/
    cv::Mat img_combine2(ROW, COL * 3, img_raw.type());
    const cv::Mat K = ( cv::Mat_<double> ( 3,3 ) << FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0 );
    const cv::Mat D = ( cv::Mat_<double> ( 5,1 ) << K1, K2, 0.0, P1, P2 );

    cv::Mat map1, map2, img_undistort;
    double alpha = 1;
    cv::Size imageSize(COL, ROW);

    cv::Mat NewCameraMatrix = getOptimalNewCameraMatrix(K, D, cv::Size(COL, ROW), alpha, cv::Size(COL, ROW), 0);
    initUndistortRectifyMap(K, D, cv::Mat(), NewCameraMatrix, imageSize, CV_16SC2, map1, map2);
    remap(img_raw, img_undistort, map1, map2, cv::INTER_LINEAR);
    img_undistort.copyTo(img_combine2(cv::Rect(0, 0, COL, ROW)));

    alpha = 0;
    NewCameraMatrix = getOptimalNewCameraMatrix(K, D, cv::Size(COL, ROW), alpha, cv::Size(COL, ROW), 0);
    initUndistortRectifyMap(K, D, cv::Mat(), NewCameraMatrix, imageSize, CV_16SC2, map1, map2);
    remap(img_raw, img_undistort, map1, map2, cv::INTER_LINEAR);
    img_undistort.copyTo(img_combine2(cv::Rect(COL, 0, COL, ROW)));

    cv::undistort(img_raw, img_undistort, K, D, K);
    img_undistort.copyTo(img_combine2(cv::Rect(COL * 2, 0, COL, ROW)));
    
    cv::putText(img_combine2,"alpha = 1", cv::Point(40, 40), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255,23,0), 4, 8);
    cv::putText(img_combine2,"alpha = 0", cv::Point(40 + COL, 40), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255,23,0), 4, 8);
    cv::putText(img_combine2,"cv::undistort", cv::Point(40 + COL * 2, 40), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(255,23,0), 4, 8);
    cv::imwrite(dir_path + "undistort.png", img_combine2);

    cv::imshow("img1", img_combine1);
    cv::imshow("img2", img_combine2);
    cv::waitKey(0);
}

void featureUtility(const std::string &dir_path, 
    Eigen::Matrix3d &R_1_2, Eigen::Vector3d &t_1_2, 
    std::vector<Eigen::Vector3d> &points3D,
    std::vector<Eigen::Vector2d> &features1,
    std::vector<Eigen::Vector2d> &features2)
{
    cv::Mat img1, img2;
    const std::string &img1_path = dir_path + "1403637188088318976.png";
    const std::string &img2_path = dir_path + "1403637189138319104.png";

    img1 = cv::imread(img1_path, cv::IMREAD_GRAYSCALE);
    img2 = cv::imread(img2_path, cv::IMREAD_GRAYSCALE);

    const cv::Mat K = ( cv::Mat_<double> ( 3,3 ) << FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0 );
    const cv::Mat D = ( cv::Mat_<double> ( 5,1 ) << K1, K2, 0.0, P1, P2 );

    /*****去畸变*****/
    cv::Mat map1, map2, img_undistort1, img_undistort2;
    double alpha = 0;
    cv::Size imageSize(COL, ROW);

    cv::Mat NewCameraMatrix = getOptimalNewCameraMatrix(K, D, cv::Size(COL, ROW), alpha, cv::Size(COL, ROW), 0);
    initUndistortRectifyMap(K, D, cv::Mat(), NewCameraMatrix, imageSize, CV_16SC2, map1, map2);
    remap(img1, img_undistort1, map1, map2, cv::INTER_LINEAR);
    remap(img2, img_undistort2, map1, map2, cv::INTER_LINEAR);

    /*****提取特征点并计算描述子*****/
    cv::Ptr<cv::ORB> detector = cv::ORB::create(400);
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptor1, descriptor2;

    detector->detectAndCompute(img_undistort1, cv::Mat(), keypoints1, descriptor1);
	detector->detectAndCompute(img_undistort2, cv::Mat(), keypoints2, descriptor2);
    cv::imwrite(dir_path + "1.png", img_undistort1);
    cv::imwrite(dir_path + "2.png", img_undistort2);

    /*****特征点匹配*****/
    cv::FlannBasedMatcher fbmatcher(new cv::flann::LshIndexParams(20, 10, 2));
	std::vector<cv::DMatch> matches;
	fbmatcher.match(descriptor1, descriptor2, matches);

    cv::Mat visual_img;
    cv::drawMatches(img_undistort1, keypoints1, img_undistort2, keypoints2, matches, visual_img,
		cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("features", visual_img);
    cv::imwrite(dir_path + "features.png", visual_img);
    cv::waitKey(0);

    /*****RANSAC过滤*****/
    //根据matches将特征点对齐,将坐标转换为float类型
    std::vector<cv::KeyPoint> R_keypoint1, R_keypoint2;
    std::vector<cv::Point2f> p1,p2;

    for (const auto &it : matches)   
    {
        R_keypoint1.push_back(keypoints1[it.queryIdx]);
        R_keypoint2.push_back(keypoints2[it.trainIdx]);

        p1.push_back(keypoints1[it.queryIdx].pt);
        p2.push_back(keypoints2[it.trainIdx].pt);
    }

    //利用基础矩阵剔除误匹配点
    std::vector<uchar> RansacStatus;
    double param1 = 3., param2 = 0.99;
    cv::Mat Fundamental= cv::findFundamentalMat(p1, p2, RansacStatus, cv::FM_RANSAC, param1, param2);

    std::vector<cv::KeyPoint> RR_keypoint1, RR_keypoint2;
    std::vector<cv::DMatch> RR_matches;
    for (int i = 0, index = 0; i < matches.size(); i++)
    {
        if (RansacStatus[i] != 0)
        {
            RR_keypoint1.push_back(R_keypoint1[i]);
            RR_keypoint2.push_back(R_keypoint2[i]);
            matches[i].queryIdx = index;
            matches[i].trainIdx = index;
            RR_matches.push_back(matches[i]);
            index++;
        }
    }

    cv::drawMatches(img_undistort1, RR_keypoint1, img_undistort2, RR_keypoint2, RR_matches, visual_img,
		cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("After RANSAC Image", visual_img);
    cv::imwrite(dir_path + "RANSAC_features.png", visual_img);
    cv::waitKey(0);

    /*****8点法求解相机pose*****/
    Eigen::Matrix3d R1, R2;
    Eigen::Vector3d t;

    int n = RR_keypoint1.size();
    std::vector<Eigen::Vector2d> keypoint1(n), keypoint2(n);
    Eigen::Vector2d xy;
    for (int i = 0; i < n; i++)
    {
        xy.x() = (RR_keypoint1.at(i).pt.x - CX) / FX;
        xy.y() = (RR_keypoint1.at(i).pt.y - CY) / FY;
        keypoint1[i] = xy; 

        xy.x() = (RR_keypoint2.at(i).pt.x - CX) / FX;
        xy.y() = (RR_keypoint2.at(i).pt.y - CY) / FY;
        keypoint2[i] = xy;
    }
    solveEssentialMatrix(keypoint1, keypoint2, R1, R2, t);

    /*****三角化并从四种E分解结果中获得正确位姿*****/
    std::vector<int> ind_set;
    triangulatePoints(R1, R2, t, keypoint1, keypoint2, &R_1_2, &t_1_2, &points3D, ind_set);
    std::cout << "triangle points num: " << points3D.size() << std::endl;

    for (const auto &it : ind_set)
    {
        features1.push_back(keypoint1.at(it));
        features2.push_back(keypoint2.at(it));
    }

    // for (Eigen::Vector3d &it : points3D)
    //     std::cout << it.x() << " " << it.y() << " " << it.z() << std::endl; 

    return ;
}

/*
*利用两帧图像的特征点和内参，恢复本质矩阵E，进而恢复位姿，一些代码来自colmap
*/
void solveEssentialMatrix(const std::vector<Eigen::Vector2d> &keypoints1,
    const std::vector<Eigen::Vector2d> &keypoints2,
    Eigen::Matrix3d &R1, Eigen::Matrix3d &R2, Eigen::Vector3d &t)
{
    
    Eigen::Matrix3d normal_matrix1, normal_matrix2;
    int n = keypoints1.size();
    //将点的x,y归一化到均值为1，求解更加稳定
    std::vector<Eigen::Vector2d> keypointsN1, keypointsN2; //均值化的点
    normalizePoints(keypoints1, keypointsN1, normal_matrix1);
    normalizePoints(keypoints2, keypointsN2, normal_matrix2);
    
    Eigen::MatrixXd A(n, 9);
    for (int i = 0; i < n; i++)
    {
        A.block<1, 3>(i, 0) = keypointsN1.at(i).homogeneous();
        A.block<1, 3>(i, 0) *= keypointsN2.at(i).x();
        A.block<1, 3>(i, 3) = keypointsN1.at(i).homogeneous();
        A.block<1, 3>(i, 3) *= keypointsN2.at(i).y();
        A.block<1, 3>(i, 6) = keypointsN1.at(i).homogeneous();
        // A.row(i) << x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1.0;
    }
    
    Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> A_svd(
      A, Eigen::ComputeFullV);
    const Eigen::VectorXd Amatrix_nullspace = A_svd.matrixV().col(8);
    const Eigen::Map<const Eigen::Matrix3d> Amatrix_t(Amatrix_nullspace.data());

    //De-normalize
    const Eigen::Matrix3d E_raw = normal_matrix2.transpose() *
                                Amatrix_t.transpose() * normal_matrix1;
    Eigen::JacobiSVD<Eigen::Matrix3d> E_raw_svd(
      E_raw, Eigen::ComputeFullU | Eigen::ComputeFullV);

    //Enforcing the internal constraint that two singular values must be equal
    // and one must be zero.
    Eigen::Vector3d singular_values = E_raw_svd.singularValues();
    singular_values(0) = (singular_values(0) + singular_values(1)) / 2.0;
    singular_values(1) = singular_values(0);
    singular_values(2) = 0.0;
    const Eigen::Matrix3d E = E_raw_svd.matrixU() * singular_values.asDiagonal() *
                            E_raw_svd.matrixV().transpose();
    
    /*****E分解得到位姿*****/
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV().transpose();

    if (U.determinant() < 0)
        U *= -1;
    if (V.determinant() < 0)
        V *= -1;

    Eigen::Matrix3d W;
    W << 0, 1, 0, -1, 0, 0, 0, 0, 1;

    R1 = U * W * V;
    R2 = U * W.transpose() * V;
    t = U.col(2).normalized();
    return ;
}

void normalizePoints(const std::vector<Eigen::Vector2d> &xy_set,
    std::vector<Eigen::Vector2d> &xy_normalize_set,
    Eigen::Matrix3d &normal_matrix)
{
    int n = xy_set.size();
    Eigen::Vector2d centroid(0, 0);

    for (const Eigen::Vector2d &it : xy_set)
        centroid += it;
    centroid /= n;

    double rms_mean_dist = 0;
    for (const Eigen::Vector2d &it : xy_set)
        rms_mean_dist += (it - centroid).squaredNorm();
    rms_mean_dist = std::sqrt(rms_mean_dist / n);

    // Compose normalization matrix
    double norm_factor = std::sqrt(2.0) / rms_mean_dist;
    normal_matrix << norm_factor, 0, -norm_factor * centroid(0), 
              0, norm_factor, -norm_factor * centroid(1), 
              0, 0, 1;
    float tx = -norm_factor * centroid(0), ty = -norm_factor * centroid(1);

    Eigen::Vector2d xy_normalize;
    for (const Eigen::Vector2d &it : xy_set)
    {
        xy_normalize.x() = norm_factor * it.x() + tx;
        xy_normalize.y() = norm_factor * it.y() + ty;
        xy_normalize_set.push_back(xy_normalize);
    }
    return ;
}

void triangulatePoints(
    const Eigen::Matrix3d &R1, const Eigen::Matrix3d &R2, const Eigen::Vector3d &t, 
    const std::vector<Eigen::Vector2d> &keypoint1, 
    const std::vector<Eigen::Vector2d> &keypoint2, 
    Eigen::Matrix3d *R_1_2, Eigen::Vector3d *t_1_2,
    std::vector<Eigen::Vector3d> *points3D, std::vector<int> &ind_set)
{
    // Generate all possible projection matrix combinations.
    const std::array<Eigen::Matrix3d, 4> R_cmbs{{R1, R2, R1, R2}};
    const std::array<Eigen::Vector3d, 4> t_cmbs{{t, t, -t, -t}};

    for (int i = 0; i < 4; i++)
    {
        std::vector<Eigen::Vector3d> points3D_cmb;
        std::vector<int> ind;
        CheckCheirality(R_cmbs[i], t_cmbs[i], keypoint1, keypoint2, &points3D_cmb, ind);
        if (points3D_cmb.size() >= points3D->size())
        {
            *R_1_2 = R_cmbs[i];
            *t_1_2 = t_cmbs[i];
            *points3D = points3D_cmb;
            ind_set = std::move(ind);
        }
    }
    return ;
}

/*
*给定R,t，三角化空间点，并计算有效空间点数目
*/
void CheckCheirality(
    const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    std::vector<Eigen::Vector3d>* points3D, std::vector<int> &ind)
{   
    points3D->clear();

    Eigen::Matrix<double, 3, 4> proj_matrix1, proj_matrix2;
    proj_matrix1.setZero();
    proj_matrix1.leftCols<3>() = Eigen::Matrix3d::Identity();
    proj_matrix2.leftCols<3>() = R;
    proj_matrix2.rightCols<1>() = t;

    const double minDepth = std::numeric_limits<double>::epsilon();
    const double maxDepth = 1000.0f * (R.transpose() * t).norm();

    for (int i = 0, iend = points1.size(); i < iend; i++)
    {
        Eigen::Matrix4d A;

        A.row(0) = points1.at(i)(0) * proj_matrix1.row(2) - proj_matrix1.row(0);
        A.row(1) = points1.at(i)(1) * proj_matrix1.row(2) - proj_matrix1.row(1);
        A.row(2) = points2.at(i)(0) * proj_matrix2.row(2) - proj_matrix2.row(0);
        A.row(3) = points2.at(i)(1) * proj_matrix2.row(2) - proj_matrix2.row(1);

        Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
        const Eigen::Vector3d point3D = svd.matrixV().col(3).hnormalized();

        if (point3D.z() > minDepth && point3D.z() < maxDepth)
        {
            double proj_z = proj_matrix2.row(2).dot(point3D.homogeneous());
            //https://github.com/colmap/colmap/issues/787
            proj_z *= proj_matrix2.col(2).norm();
            if (proj_z > minDepth && proj_z < maxDepth)
            {
                points3D->push_back(point3D);
                ind.push_back(i);
            }
                
        }
    }
}

void featureBA()
{

}