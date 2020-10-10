#include <iostream>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/registration/ia_ransac.h>

#include <pcl/segmentation/sac_segmentation.h>

#include "base_io.h"


using namespace zjlab;
using namespace pose;
using namespace std;

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;
    FLAGS_stderrthreshold = 0;

    pcl::PointCloud<pcl::PointXYZ>::Ptr grap_point(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr result_grap_point(new pcl::PointCloud<pcl::PointXYZ>);

    grap_point->push_back(pcl::PointXYZ(0.006973,0.001327,0.016474));
    grap_point->push_back(pcl::PointXYZ(0.009587,0.002154,-0.016744));
    pcl::io::savePCDFileASCII("/home/zhachanghai/duck/grap_point.pcd", *grap_point);


    pcl::PointCloud<pcl::PointXYZ>::Ptr duck_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr back_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    string file_path = "/home/zhachanghai/duck/duck_new.ply";
    string back_path = "/home/zhachanghai/duck/test_table_duck.pcd";
    string origin_scan = "/home/zhachanghai/duck/table.pcd";

    int point_num = 0;
    Eigen::Vector3f *positions = nullptr;
    Eigen::Vector3f *colors = nullptr;
    Eigen::Vector3f *normals = nullptr;
    float *radii = nullptr;

    if (!readPly(point_num, positions, colors, normals, radii, file_path)) {
        LOG(FATAL) << "read ply file error !";
        return -1;
    }

    for (int i = 0; i < point_num; i++) {
        pcl::PointXYZ pt;
        // test for sample match
        pt.x = positions[i][0];
        pt.y = positions[i][1];
        pt.z = positions[i][2];

//        LOG(INFO) << "x " << pt.x << " pt.y " << pt.y << "  pt.z " << pt.z;
        duck_cloud->push_back(pt);
    }

    pcl::io::loadPCDFile<pcl::PointXYZ>(back_path, *back_cloud);

    // down_sampling
    LOG(INFO) << "Downsampling !!";
//    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
//    voxel_grid.setLeafSize(0.012,0.012,0.012);
//    voxel_grid.setInputCloud(duck_cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_src (new pcl::PointCloud<pcl::PointXYZ>());
//    voxel_grid.filter(*cloud_src);
    pcl::copyPointCloud(*duck_cloud,*cloud_src);
    LOG(INFO) << "down size target from " << duck_cloud->size() << " to "<< cloud_src->size();

    //计算表面法线
    pcl::NormalEstimation<pcl::PointXYZ,pcl::Normal> ne_src;
    ne_src.setInputCloud(cloud_src);
    pcl::search::KdTree< pcl::PointXYZ>::Ptr tree_src(new pcl::search::KdTree< pcl::PointXYZ>());
    ne_src.setSearchMethod(tree_src);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_src_normals(new pcl::PointCloud< pcl::Normal>);
    ne_src.setRadiusSearch(0.02);
    ne_src.compute(*cloud_src_normals);

    std::vector<int> indices_tgt;
    pcl::removeNaNFromPointCloud(*back_cloud,*back_cloud, indices_tgt);


//    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_2;
//    voxel_grid_2.setLeafSize(0.01,0.01,0.01);
//    voxel_grid_2.setInputCloud(back_cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tgt (new pcl::PointCloud<pcl::PointXYZ>);
//    voxel_grid_2.filter(*cloud_tgt);
    pcl::copyPointCloud(*back_cloud,*cloud_tgt);
    LOG(INFO) << "down size scene from " << back_cloud->size() << " to "<< cloud_tgt->size();


    pcl::NormalEstimation<pcl::PointXYZ,pcl::Normal> ne_tgt;
    ne_tgt.setInputCloud(cloud_tgt);
    pcl::search::KdTree< pcl::PointXYZ>::Ptr tree_tgt(new pcl::search::KdTree< pcl::PointXYZ>());
    ne_tgt.setSearchMethod(tree_tgt);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_tgt_normals(new pcl::PointCloud< pcl::Normal>);
    //ne_tgt.setKSearch(20);
    ne_tgt.setRadiusSearch(0.02);
    ne_tgt.compute(*cloud_tgt_normals);
    LOG(INFO) << "normal compute over ";

    //计算FPFH
    pcl::FPFHEstimation<pcl::PointXYZ,pcl::Normal,pcl::FPFHSignature33> fpfh_src;
    fpfh_src.setInputCloud(cloud_src);
    fpfh_src.setInputNormals(cloud_src_normals);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_src_fpfh (new pcl::search::KdTree<pcl::PointXYZ>);
    fpfh_src.setSearchMethod(tree_src_fpfh);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_src(new pcl::PointCloud<pcl::FPFHSignature33>());
    fpfh_src.setRadiusSearch(0.05);
    //fpfh_src.setKSearch(20);
    fpfh_src.compute(*fpfhs_src);
    LOG(INFO) << "compute model cloud FPFH over !";

    pcl::FPFHEstimation<pcl::PointXYZ,pcl::Normal,pcl::FPFHSignature33> fpfh_tgt;
    fpfh_tgt.setInputCloud(cloud_tgt);
    fpfh_tgt.setInputNormals(cloud_tgt_normals);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_tgt_fpfh (new pcl::search::KdTree<pcl::PointXYZ>);
    fpfh_tgt.setSearchMethod(tree_tgt_fpfh);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_tgt(new pcl::PointCloud<pcl::FPFHSignature33>());
    fpfh_tgt.setRadiusSearch(0.05);
    //fpfh_tgt.setKSearch(20);
    fpfh_tgt.compute(*fpfhs_tgt);
    LOG(INFO) << "compute target cloud FPFH over !";


    //SAC配准
    pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> scia;
    scia.setInputSource(cloud_src);
    scia.setInputTarget(cloud_tgt);
    scia.setSourceFeatures(fpfhs_src);
    scia.setTargetFeatures(fpfhs_tgt);
    //scia.setMinSampleDistance(1);
    //scia.setNumberOfSamples(2);
    //scia.setCorrespondenceRandomness(20);
    pcl::PointCloud<pcl::PointXYZ>::Ptr sac_result (new pcl::PointCloud<pcl::PointXYZ>);
    scia.align(*sac_result);
    LOG(INFO) << "sac has converged:"<<scia.hasConverged()<<"  score: "<<scia.getFitnessScore();
    Eigen::Matrix4f sac_trans;
    sac_trans = scia.getFinalTransformation();
    LOG(INFO) << sac_trans;
    pcl::io::savePCDFileASCII("/home/zhachanghai/duck/icp_guess.pcd", *sac_result);


    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(sac_result);
    icp.setInputTarget(cloud_tgt);
//    icp.setMaximumIterations (100);
//    icp.setTransformationEpsilon (1e-9);
//    icp.setMaxCorrespondenceDistance (0.05);
//    icp.setEuclideanFitnessEpsilon (1);
//    icp.setRANSACOutlierRejectionThreshold (1.5);

    //icp.setMaxCorrespondenceDistance(0.005);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);

    LOG(INFO) << "has converged:" << icp.hasConverged() << " score: " <<
              icp.getFitnessScore();
    Eigen::Matrix4f icp_trans;
    icp_trans = icp.getFinalTransformation();
    LOG(INFO) << icp_trans;

    pcl::PointCloud<pcl::PointXYZ>::Ptr final_cloud_(new pcl::PointCloud<pcl::PointXYZ>);

    Eigen::Matrix4f rototranslation =  icp_trans * sac_trans;
    LOG(INFO) << rototranslation;

    Eigen::Matrix3f rotation = rototranslation.block<3,3>(0, 0);
    Eigen::Vector3f translation = rototranslation.block<3,1>(0, 3);

    pcl::transformPointCloud(*duck_cloud,*final_cloud_, rototranslation);
    pcl::transformPointCloud(*grap_point,*result_grap_point, rototranslation);

    Eigen::Matrix4f trans_scan;
    trans_scan <<   -0.279421, 0.935276, -0.217215, 0.396542,
                    0.891638, 0.168817, -0.420098, 0.082946,
                    -0.356238, -0.311061, -0.881099, 1.289688,
                    0.000000, 0.000000, 0.000000, 1.000000;


    pcl::PointCloud<pcl::PointXYZ>::Ptr origin_table(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile<pcl::PointXYZ>(origin_scan, *origin_table);
    pcl::PointCloud<pcl::PointXYZ>::Ptr result_table(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(*origin_table,*result_table, trans_scan);
    pcl::io::savePCDFileASCII("/home/zhachanghai/duck/origin_scan.pcd", *result_table);


    printf ("\n");
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
    printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
    printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
    printf ("\n");
    printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));

    pcl::io::savePCDFileASCII("/home/zhachanghai/duck/result.pcd", *final_cloud_);

    pcl::io::savePCDFileASCII("/home/zhachanghai/duck/result_grap.pcd", *result_grap_point);


    return 0;
}


