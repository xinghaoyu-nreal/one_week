#ifndef OPENCV_UTILITY_H
#define OPENCV_UTILITY_H

#include <Eigen/Dense>
#include <Eigen/Core>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/persistence.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

extern int COL, ROW;
extern float FX, FY, CX, CY;
extern float K1, K2, P1, P2;

void readParam(const std::string &dir_path);
void imgUtility(const std::string &dir_path);
void featureUtility(const std::string &dir_path, 
    Eigen::Matrix3d &R_1_2, Eigen::Vector3d &t_1_2, 
    std::vector<Eigen::Vector3d> &points3D, 
    std::vector<Eigen::Vector2d> &features1,
    std::vector<Eigen::Vector2d> &features2);

void solveEssentialMatrix(const std::vector<Eigen::Vector2d> &keypoints1,
    const std::vector<Eigen::Vector2d> &keypoints2,
    Eigen::Matrix3d &R1, Eigen::Matrix3d &R2, Eigen::Vector3d &t);
void normalizePoints(const std::vector<Eigen::Vector2d> &xy_set, 
    std::vector<Eigen::Vector2d> &xy_normalize_set,
    Eigen::Matrix3d &normal_matrix1);
void CheckCheirality(
    const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2,
    std::vector<Eigen::Vector3d>* points3D, std::vector<int> &ind);
void triangulatePoints(
    const Eigen::Matrix3d &R1, const Eigen::Matrix3d &R2, const Eigen::Vector3d &t, 
    const std::vector<Eigen::Vector2d> &keypoint1, 
    const std::vector<Eigen::Vector2d> &keypoint2,
    Eigen::Matrix3d *R_1_2, Eigen::Vector3d *t_1_2, 
    std::vector<Eigen::Vector3d> *points3D, std::vector<int> &ind_set);

void featureBA();

#endif