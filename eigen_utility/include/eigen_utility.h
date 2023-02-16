#ifndef EIGEN_UTILITY_H
#define EIGEN_UTILITY_H

#include <Eigen/Dense>
#include <Eigen/Core>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

void transform();
void solveEquation(const std::string &txt_path, double &cond);
void readXY(const std::string &txt_path, std::vector<std::pair<double, double> > &xy_set);

#endif