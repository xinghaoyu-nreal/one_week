#include "eigen_utility.h"
#include "opencv_utility.h"
#include "pose_local_parameterization.h"
#include "projection.h"

#include <ceres/ceres.h>
#include <ceres/gradient_checker.h>

double param_pose[2][7];
double param_feature[1000][3];

void vector2double(const std::vector<Eigen::Vector3d> &points,
    const Eigen::Matrix3d &R_w_c2, const Eigen::Vector3d &t_w_c2)
{
    param_pose[1][0] = t_w_c2.x();
    param_pose[1][1] = t_w_c2.y();
    param_pose[1][2] = t_w_c2.z();

    Eigen::Quaterniond q{R_w_c2};
    param_pose[1][3] = q.x();
    param_pose[1][4] = q.y();
    param_pose[1][5] = q.z();
    param_pose[1][6] = q.w();

    param_pose[0][0] = param_pose[0][1] = param_pose[0][2];
    q = Eigen::Matrix3d::Identity();
    param_pose[0][3] = q.x();
    param_pose[0][4] = q.y();
    param_pose[0][5] = q.z();
    param_pose[0][6] = q.w();

    for (int i = 0, iend = points.size(); i < iend; i++)
    {
        param_feature[i][0] = points.at(i).x();
        param_feature[i][1] = points.at(i).y();
        param_feature[i][2] = points.at(i).z();
    }
}

void double2vector(Eigen::Matrix3d &R_w_c2, 
    Eigen::Vector3d &t_w_c2, std::vector<Eigen::Vector3d> &points)
{
    Eigen::Quaterniond q{param_pose[1][6], param_pose[1][3], param_pose[1][4], param_pose[1][5]};
    R_w_c2 = q.toRotationMatrix();

    t_w_c2.x() = param_pose[1][0];
    t_w_c2.y() = param_pose[1][1];
    t_w_c2.z() = param_pose[1][2];

    for (int i = 0, iend = points.size(); i < iend; i++)
    {
        points.at(i).x() = param_feature[i][0];
        points.at(i).y() = param_feature[i][1];
        points.at(i).z() = param_feature[i][2];
    }
}

void computeError(const std::string &dir_path, int ind,
    const Eigen::Matrix3d &R_wc, const Eigen::Vector3d &t_wc, 
    const std::vector<Eigen::Vector3d> &points3D,
    const std::vector<Eigen::Vector2d> &features)
{
    std::ofstream fout(dir_path + "repro" + std::to_string(ind) + ".txt", std::ios::out);

    std::cout << "\n=========\n";
    cv::Mat img = cv::imread(dir_path + std::to_string(ind) + ".png", cv::IMREAD_GRAYSCALE);
    cv::cvtColor(img, img, cv::COLOR_GRAY2RGB);
    for (const Eigen::Vector2d &it : features)
    {
        cv::Point a;
        a.x = it.x() * FX + CX;
        a.y = it.y() * FY + CY;

        cv::circle(img, a, 5, cv::Scalar(0, 0, 255), 3, 8);
    }

    Eigen::Vector3d pt_cam;
    std::vector<double> errors;
    double error = 0;
    int n = points3D.size();

    for (int i = 0; i < n; i++)
    {
        pt_cam = R_wc.transpose() * (points3D.at(i) - t_wc);
        pt_cam /= pt_cam[2];

        cv::Point a;
        a.x = pt_cam.x() * FX + CX;
        a.y = pt_cam.y() * FY + CY;
        cv::circle(img, a, 5, cv::Scalar(0, 255, 0), 2, 8);

        error += (pt_cam.head(2) - features.at(i)).norm();

        double m = pt_cam(0) - features.at(i)(0);
        double n = pt_cam(1) - features.at(i)(1);
        fout << m << " " << n << "\n";
        errors.push_back(error);
    }

    error /= n;
    std::cout << "mean value = " << error << std::endl;

    double rms_mean_dist = 0;
    for (const double &it : errors)
        rms_mean_dist += (it - error) * (it - error);
    
    std::cout << "variance = " << rms_mean_dist << std::endl;
    rms_mean_dist = std::sqrt(rms_mean_dist / n);
    std::cout << "rms mean value = " << rms_mean_dist << "\n";

    cv::imwrite(dir_path + "repro" + std::to_string(ind) + ".png", img);
    cv::imshow("1", img);
    cv::waitKey(0);

    fout.close();
    return ;
}

void featureBA(const std::string &dir_path, 
    Eigen::Matrix3d &R_w_c2, Eigen::Vector3d &t_w_c2, 
    std::vector<Eigen::Vector3d> &points3D,
    std::vector<Eigen::Vector2d> &features1,
    std::vector<Eigen::Vector2d> &features2)
{
    vector2double(points3D, R_w_c2, t_w_c2);

    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);

    for (int i = 0; i < 2; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(param_pose[i], 7, local_parameterization);
        
        if (!i)
            problem.SetParameterBlockConstant(param_pose[i]);
    }

    for (int i = 0, iend = points3D.size(); i < iend; i++)
    {
        // ceres::CostFunction* f1 = 
        //         new ceres::AutoDiffCostFunction<featureProjectionFactor, 2, 7, 3>(new featureProjectionFactor(features1.at(i)));
        
        ProjectionFactor *f1 = new ProjectionFactor(features1.at(i));
        problem.AddResidualBlock(f1, loss_function, param_pose[0], param_feature[i]);
        
        // ceres::CostFunction* f2 = 
        //         new ceres::AutoDiffCostFunction<featureProjectionFactor, 2, 7, 3>(new featureProjectionFactor(features2.at(i)));
        
        ProjectionFactor *f2 = new ProjectionFactor(features2.at(i));
        problem.AddResidualBlock(f2, loss_function, param_pose[1], param_feature[i]);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR; 
    options.trust_region_strategy_type = ceres::DOGLEG;
    // options.check_gradients = true;

    options.max_num_iterations = 100;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    double2vector(R_w_c2, t_w_c2, points3D);

    Eigen::Matrix3d R_w_c1 = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_w_c1 = Eigen::Vector3d::Zero();

    computeError(dir_path, 1, R_w_c1, t_w_c1, points3D, features1);
    computeError(dir_path, 2, R_w_c2, t_w_c2, points3D, features2);
}

//gradient check
void checkGradient(int pose_ind, int feature_ind, const Eigen::Vector2d &feature)
{   
    std::vector<double *> parameter_blocks;
    ceres::NumericDiffOptions numeric_diff_options;
    
    ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
    const std::vector<const ceres::LocalParameterization *> local_parameterizations{local_parameterization, nullptr};

    parameter_blocks.push_back(param_pose[pose_ind]);
    parameter_blocks.push_back(param_feature[feature_ind]);
    
    ceres::CostFunction* f1 = 
        new ceres::AutoDiffCostFunction<featureProjectionFactor, 2, 7, 3>(new featureProjectionFactor(feature));
    // ProjectionFactor *f1 = new ProjectionFactor(feature);
    ceres::GradientChecker gradient_checker(f1, &local_parameterizations, numeric_diff_options);
    ceres::GradientChecker::ProbeResults results;

    if (!gradient_checker.Probe(parameter_blocks.data(), 1e-9, &results))
        std::cout << "An error has occurred:\n" << results.error_log;

}

int main()
{
    std::cout << "\n\n=======Transform=======\n";
    transform();

    const std::string &txt_path = "../config/data.txt";
    const std::string &txt2_path = "../config/data2.txt";
    double condition_number, condition_number2;
    
    std::cout << "\n\n=======Ax=b=======\n";
    solveEquation(txt_path, condition_number);
    std::cout << "\n\n=======Ax=b=======\n";
    solveEquation(txt2_path, condition_number2);
    
    const std::string &dir_path = "../config/two_image_pose_estimation/";
    readParam(dir_path);
    imgUtility(dir_path);

    Eigen::Matrix3d R_w_c2, R_w_c1;
    Eigen::Vector3d t_w_c2, t_w_c1;
    R_w_c1 = Eigen::Matrix3d::Identity();
    t_w_c1 = Eigen::Vector3d::Zero();

    std::vector<Eigen::Vector3d> points3D;
    std::vector<Eigen::Vector2d> features1, features2;
    featureUtility(dir_path, R_w_c2, t_w_c2, points3D, features1, features2);

    featureBA(dir_path, R_w_c2, t_w_c2, points3D, features1, features2);

    for (int j = 0, jend = features1.size(); j < jend; j++)
    {
        checkGradient(0, j, features1.at(j));
        checkGradient(1, j, features2.at(j));
    }

    return 0;
}