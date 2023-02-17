#include "projection.h"
#include "tic_toc.h"

Eigen::Matrix2d ProjectionFactor::sqrt_info = Eigen::Matrix2d::Identity();;
double ProjectionFactor::sum_t;

ProjectionFactor::ProjectionFactor(const Eigen::Vector2d &_pt) : pt(_pt){};

bool ProjectionFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    Eigen::Vector3d t_wc(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond q_wc(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d pt_w(parameters[1][0], parameters[1][1], parameters[1][2]);

    Eigen::Quaterniond q_cw = q_wc.inverse();
    Eigen::Vector3d t_cw = -q_wc.toRotationMatrix().transpose() * t_wc;
    Eigen::Vector3d pt_cam = q_cw * pt_w + t_cw;
    double z = pt_cam[2];

    Eigen::Map<Eigen::Vector2d> residual(residuals);
    residual = pt_cam.head(2)/z - pt;
    residual = sqrt_info * residual;

    if (jacobians)
    {
        Eigen::Matrix<double, 2, 3> reduce(2, 3);
        reduce << 1. / z, 0, -pt_cam(0) / (z * z),
            0, 1. / z, -pt_cam(1) / (z * z);
        reduce = sqrt_info * reduce;

        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose(jacobians[0]);

            Eigen::Matrix<double, 3, 6> jaco_i;
            jaco_i.leftCols<3>() = -q_wc.toRotationMatrix().transpose();
            jaco_i.rightCols<3>() = Utility::skewSymmetric(pt_cam);
            
            jacobian_pose.leftCols<6>() = reduce * jaco_i;
            jacobian_pose.rightCols<1>().setZero();
        }

        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jacobian_feature(jacobians[1]);
            jacobian_feature = reduce * q_wc.toRotationMatrix().transpose();
        }
    }
    
    sum_t += tic_toc.toc();

    return true;
}

void ProjectionFactor::check(double **parameters)
{
    double *res = new double[2];
    double **jaco = new double *[2];
    jaco[0] = new double[7];
    jaco[1] = new double[3];
    Evaluate(parameters, res, jaco);
    puts("check begins");

    puts("my");

    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 1>>(res).transpose() << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jaco[0]) << std::endl
              << std::endl;
    std::cout << Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>(jaco[1]) << std::endl
              << std::endl;

    Eigen::Vector3d t_wc(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond q_wc(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
    Eigen::Vector3d pt_w(parameters[1][0], parameters[1][1], parameters[1][2]);


    Eigen::Vector2d residual;
    Eigen::Quaterniond q_cw = q_wc.inverse();
    Eigen::Vector3d t_cw = -q_wc.toRotationMatrix().transpose() * t_wc;
    Eigen::Vector3d pt_cam = q_cw * pt_w + t_cw;
    double z = pt_cam[2];
    residual = pt_cam.head(2)/z - pt;
    residual = sqrt_info * residual;

    puts("num");
    std::cout << residual.transpose() << std::endl;

    const double eps = 1e-6;
    Eigen::Matrix<double, 2, 10> num_jacobian;
    for (int k = 0; k < 10; k++)
    {
        Eigen::Vector3d t_wc(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond q_wc(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
        Eigen::Vector3d pt_w(parameters[1][0], parameters[1][1], parameters[1][2]);

        int a = k / 3, b = k % 3;
        Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

        if (a == 0)
            t_wc += delta;
        else if (a == 1)
            q_wc = q_wc * Utility::deltaQ(delta);
        else if (a == 2)
            pt_w += delta;

        Eigen::Vector2d tmp_residual;
        Eigen::Quaterniond q_cw = q_wc.inverse();
        Eigen::Vector3d t_cw = -q_wc.toRotationMatrix().transpose() * t_wc;
        Eigen::Vector3d pt_cam = q_cw * pt_w + t_cw;
        double z = pt_cam[2];
        tmp_residual = pt_cam.head(2)/z - pt;

        tmp_residual = sqrt_info * tmp_residual;
        num_jacobian.col(k) = (tmp_residual - residual) / eps;
    }
    std::cout << num_jacobian << std::endl;
}
