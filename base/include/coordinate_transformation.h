//
// Created by zhachanghai on 20-10-22.
//

#pragma once

#include <glog/logging.h>
#include <Eigen/Dense>
#include <string>
#include <vector>

namespace zjlab{
    namespace pose{

#define radianToDegree(radian___) ((radian___)*180.0 / M_PI)
#define degreeToRadian(degree___) ((degree___)*M_PI / 180.0)

    inline Eigen::Matrix4d R2T(const Eigen::Matrix3d& R) {
        Eigen::Matrix4d tf = Eigen::Matrix4d::Identity();
        tf.topLeftCorner<3, 3>() = R;
        return tf;
    }

    inline Eigen::Matrix4d Rt2T(const Eigen::Matrix3d& R,
                                const Eigen::Vector3d& t) {
        Eigen::RowVector4d A14;
        Eigen::Matrix4d tf;
        tf << R, t, A14;
        return tf;
    }

    /**
 * Given yaw,pitch, roll: calculate the extrinsic rotation matrix with ZYX order
 *
 * @param yaw extrinsic rotation along Z axis in radians
 * @param pitch extrinsic rotation along Y axis in radians
 * @param roll extrinsic rotation along X axis in radians
 * @return
 */
        inline Eigen::Matrix3d ypr2RotationZYX(double yaw, double pitch, double roll) {
            return (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                    Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
                    Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()))
                    .toRotationMatrix();
        }
/**
 * Given yaw,pitch, roll: calculate the extrinsic rotation matrix with ZYX order
 *
 * @param yaw extrinsic rotation along Z axis in degrees
 * @param pitch extrinsic rotation along Y axis in degrees
 * @param roll extrinsic rotation along X axis in degrees
 * @return
 */
        inline Eigen::Matrix3d yprDegrees2RotationZYX(double yaw, double pitch,
                                                      double roll) {
            return ypr2RotationZYX(degreeToRadian(yaw), degreeToRadian(pitch),
                                   degreeToRadian(roll));
        }

/**
 * Inverse a tranformation matrix with only rotation and translation, no scaling
 *
 * [ R t ]
 * [ 0 1 ]
 * where R is a strict rotation matrix with R.inverse() = R.transpose();
 *
 * @param matrix
 * @return
 */
        inline Eigen::Matrix4d inverseTransMatrix(const Eigen::Matrix4d& matrix) {
            static const auto A14 = Eigen::RowVector4d(0, 0, 0, 1);

            Eigen::Matrix3d Rt = matrix.topLeftCorner<3, 3>().transpose();
            Eigen::Matrix4d re;
            re << Rt, Rt * -1 * matrix.topRightCorner<3, 1>(), A14;
            return re;
        }


/**
 * Given yaw,pitch, roll: calculate the Reverse extrinsic rotation matrix with
 * ZYX order
 *
 * @param yaw extrinsic rotation along Z axis in radians
 * @param pitch extrinsic rotation along Y axis in radians
 * @param roll extrinsic rotation along X axis in radians
 * @return
 */
        inline Eigen::Matrix3d ypr2RotationZYXInverse(double yaw, double pitch,
                                                      double roll) {
            return (Eigen::AngleAxisd(-roll, Eigen::Vector3d::UnitX()) *
                    Eigen::AngleAxisd(-pitch, Eigen::Vector3d::UnitY()) *
                    Eigen::AngleAxisd(-yaw, Eigen::Vector3d::UnitZ()))
                    .toRotationMatrix();
        }

/**
 * Given yaw,pitch, roll: calculate the INverse extrinsic rotation matrix with
 * ZYX order
 *
 * @param yaw extrinsic rotation along Z axis in degrees
 * @param pitch extrinsic rotation along Y axis in degrees
 * @param roll extrinsic rotation along X axis in degrees
 * @return
 */
        inline Eigen::Matrix3d yprDegrees2RotationZYXInverse(double yaw, double pitch,
                                                             double roll) {
            return ypr2RotationZYXInverse(degreeToRadian(yaw), degreeToRadian(pitch),
                                          degreeToRadian(roll));
        }

    }
}
