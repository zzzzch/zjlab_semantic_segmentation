//
// Created by zhachanghai on 20-9-7.
//

#pragma once
#include <iostream>
#include <Eigen/Dense>



#include <string>
#include <fstream>
#include <memory>
#include <vector>
#include <algorithm>
#include <limits>

class PartialMatchingFacada
{};

class NBO
{
    public:
        NBO(std::shared_ptr<PartialMatchingFacada> pmFacada);
        ~NBO() = default;

        bool hasNBO() const { return nbo_flag;}

    private:

        bool nbo_flag = false;

        const double objectness_weight = 1.0;
        const double position_weight = 1.5;
        const double orient_weight = 1.0;
        const double size_weight = 1.0;
        const double gauss_fuction_center = 0.0;
        const double gauss_fuction_width = 0.05;
        const double max_error_consider_recognized = 0.012;
};