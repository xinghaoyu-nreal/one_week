#include "eigen_utility.h"
#include "tic_toc.h"

void transform()
{
    //transform from left eye to imu
    Eigen::Quaterniond r_i_l(0.99090224973327068, -0.0084222184858180373, 0.00095051670014565813, 0.13431639597354814);
    Eigen::Vector3d t_i_l{-0.050720060477640147, -0.0017414170413474165, 0.0022943667597148118};

    //transform form right eye to imu
    Eigen::Quaterniond r_i_r(0.99073762672679389, 0.13492462817073628, -0.00013648999867379373, -0.015306242884176362);
    Eigen::Vector3d t_i_r{0.051932496584961352, -0.0011555929083120534, 0.0030949732069645722};

    Eigen::Quaterniond r_l_r;
    Eigen::Vector3d t_l_r;

    r_l_r = r_i_l.inverse() * r_i_r;
    t_l_r = r_i_l.inverse().toRotationMatrix() * (t_i_r - t_i_l);

    std::cout << "R from right eye to left eye\n" << r_l_r.toRotationMatrix() << std::endl;
    std::cout << "t from right eye to left eye\n" << t_l_r.transpose() << std::endl;

    return ;
}

void solveEquation(const std::string &txt_path, double &cond)
{
    std::vector<std::pair<double, double> > xy_set;
    readXY(txt_path, xy_set);
    const int n = xy_set.size();

    Eigen::MatrixXd A(n, 2);
    Eigen::VectorXd b(n);
    for (int i = 0; i < n; i++)
    {
        A(i, 0) = xy_set.at(i).first;
        A(i, 1) = 1;

        b(i) = xy_set.at(i).second;
    }
    
    Eigen::Vector2d solution{4, 1.5};
    Eigen::Vector2d mn;
    TicToc run_time;

    mn = A.colPivHouseholderQr().solve(b);
    std::cout << "The solution using the QR decomposition:\n" << mn.transpose() 
              << "\nRun time: " << run_time.toc() << "s, error: " << (mn - solution).norm() << std::endl;
    
    run_time.tic();
    mn = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    std::cout << "The solution using the SVD decomposition:\n" << mn.transpose() 
              << "\nRun time: " << run_time.toc() << "s, error: " << (mn - solution).norm() << std::endl;

    run_time.tic();
    mn = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    std::cout << "The solution using the (ATA)-1 * (ATb):\n" << mn.transpose() 
              << "\nRun time: " << run_time.toc() << "s, error: " << (mn - solution).norm() << std::endl; 

    //condition number
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A);
    cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1); 
    std::cout << "condition number: " << cond << std::endl;
    return ;
}

void readXY(const std::string &txt_path, std::vector<std::pair<double, double> > &xy_set)
{
    std::ifstream infile;
    infile.open(txt_path, std::ios::in);

    xy_set.clear();

    if (!infile.is_open())
    {
        std::cerr << "file not exist\n";
        return ;
    }

    double x, y;
    while (!infile.eof())
    {
        infile >> x >> y;
        xy_set.emplace_back(std::make_pair(x, y));
    }

    return ;
}

