#ifndef PROJECTION_H
#define PROJECTION_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include "utility.h"

struct featureProjectionFactor
{
  featureProjectionFactor(Eigen::Vector2d _pt) : pt(_pt){}

  template <typename T>
  bool operator() (const T* const pose,   //7
                   const T* const point,  //3 
                   T* residuals) const
  {
    const Eigen::Matrix<T, 3, 1> t_wc(pose[0], pose[1], pose[2]);
    const Eigen::Quaternion<T> q_wc(pose[6], pose[3], pose[4], pose[5]);
    const Eigen::Matrix<T, 3, 1> pt_w(point[0], point[1], point[2]);

    Eigen::Quaternion<T> q_cw = q_wc.inverse();
    Eigen::Matrix<T, 3, 1> t_cw = -q_wc.toRotationMatrix().transpose() * t_wc;

    Eigen::Matrix<T, 3, 1> pt_cam = q_cw * pt_w + t_cw;
    pt_cam /= pt_cam[2];

    Eigen::Matrix<T, 2, 1> pixel = pt.template cast<T>();

    residuals[0] = (pt_cam[0] - pixel[0]);
    residuals[1] = (pt_cam[1] - pixel[1]);

    return true;

  } 
private:
  Eigen::Vector2d pt;
};

class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 3>
{
  public:
    ProjectionFactor(const Eigen::Vector2d &_pts);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);

    Eigen::Vector2d pt;
    static Eigen::Matrix2d sqrt_info;
    static double sum_t;
};


#endif