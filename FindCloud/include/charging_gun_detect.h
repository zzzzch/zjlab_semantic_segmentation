#pragma once

#include <Eigen/Eigen>

namespace zjlab{
    namespace pose{
        class CameraInfo{
        public:
            void init(){
                k_rgb_camera << 634.2861328125,  0,      633.5665893554688,
                        0,        632.8214111328125,  404.9833679199219,
                        0,        0,        1;

                k_depth_camera << 634.2861328125,  0,        633.5665893554688,
                        0,        632.8214111328125,  404.9833679199219,
                        0,        0,        1;

//                d_rgb_camera << -0.054005738347768784, 0.061921533197164536, 0.00035799629404209554, -0.00043504239874891937, -0.01914302632212639;
//                d_depth_camera << -0.054005738347768784, 0.061921533197164536, 0.00035799629404209554, -0.00043504239874891937, -0.01914302632212639;
            }

            float get_rgb_fx() {return k_rgb_camera(0,0);}

            float get_rgb_fy() {return k_rgb_camera(1,1);}

            float get_rgb_cx() {return k_rgb_camera(0,2);}

            float get_rgb_cy() {return k_rgb_camera(1,2);}

        private:
            Eigen::Matrix3f k_rgb_camera;
            Eigen::VectorXf d_rgb_camera;

            Eigen::Matrix3f k_depth_camera;
            Eigen::VectorXf d_depth_camera;

        };
    }
}
