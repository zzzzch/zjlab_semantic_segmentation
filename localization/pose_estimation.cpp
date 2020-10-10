#include <iostream>
#include <glog/logging.h>
#include <gflags/gflags.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>     //点云类型头文件
#include <pcl/correspondence.h>   //对应表示两个实体之间的匹配（例如，点，描述符等）。
#include <pcl/features/normal_3d_omp.h>   //法线
#include <pcl/features/shot_omp.h>    //描述子
#include <pcl/features/board.h>
#include <pcl/filters/uniform_sampling.h>   //均匀采样
#include <pcl/filters/voxel_grid.h>
#include <pcl/recognition/cg/hough_3d.h>    //hough算子
#include <pcl/recognition/cg/geometric_consistency.h>  //几何一致性
#include <pcl/kdtree/kdtree_flann.h>             //配准方法
#include <pcl/kdtree/impl/kdtree_flann.hpp>      //
#include <pcl/common/transforms.h>             //转换矩阵
#include <pcl/segmentation/extract_clusters.h>

#include "base_io.h"
#include "pcl_tools.h"

DEFINE_string(scene_path,
              "/home/zhachanghai/guochen_test/pcl_point/142 _test.pcd", "scene pcd file");

DEFINE_string(target_path,
              "/home/zhachanghai/guochen_test/model.ply", "target ply file");

DEFINE_string(output_path, "/home/zhachanghai/guochen_test/", "output file path");

using namespace zjlab;
using namespace pose;
using namespace std;

typedef pcl::PointXYZ PointType;  //PointXYZRGBA数据结构
typedef pcl::Normal NormalType;       //法线结构
typedef pcl::ReferenceFrame RFType;    //参考帧
typedef pcl::SHOT352 DescriptorType;   //SHOT特征的数据结构

//Algorithm params
bool show_keypoints_ (false);
bool show_correspondences_ (false);
bool use_cloud_resolution_ (false);
bool use_hough_ (true);
float model_ss_ (0.01f);
float scene_ss_ (0.01f);
float rf_rad_ (0.015f);
float descr_rad_ (0.04f);
float cg_size_ (0.03f);
float cg_thresh_ (1.5f);


double
computeCloudResolution (const pcl::PointCloud<PointType>::ConstPtr &cloud)//计算点云分辨率
{
    double res = 0.0;
    int n_points = 0;
    int nres;
    std::vector<int> indices (2);
    std::vector<float> sqr_distances (2);
    pcl::search::KdTree<PointType> tree;
    tree.setInputCloud (cloud);   //设置输入点云

    for (size_t i = 0; i < cloud->size (); ++i)
    {
        if (! pcl_isfinite ((*cloud)[i].x))
        {
            continue;
        }
        //Considering the second neighbor since the first is the point itself.
        //运算第二个临近点，因为第一个点是它本身
        nres = tree.nearestKSearch (i, 2, indices, sqr_distances);//return :number of neighbors found
        if (nres == 2)
        {
            res += sqrt (sqr_distances[1]);
            ++n_points;
        }
    }
    if (n_points != 0)
    {
        res /= n_points;
    }
    return res;
}

int main (int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;
    FLAGS_stderrthreshold = 0;

    pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ()); //模型点云
    pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());//模型角点
    pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());  //目标点云
    pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());//目标角点
    pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ()); //法线
    pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());
    pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ()); //描述子
    pcl::PointCloud<DescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType> ());
    pcl::PointCloud<PointType>::Ptr result(new pcl::PointCloud<PointType> ()); // 输出结果

    string scene_file_path = FLAGS_scene_path;
    string target_file_path = FLAGS_target_path;

    //读入文件
    {
        pcl::io::loadPCDFile(scene_file_path,*scene);
        //remove far points
        pcl::PointCloud<pcl::PointXYZ>::Ptr distance_scene(new pcl::PointCloud<pcl::PointXYZ>);
        for(int i = 0; i < scene->size(); i++){
            float distance = pow(scene->points[i].x,2)+pow(scene->points[i].y,2)+pow(scene->points[i].z,2);
            if(distance < 2){
                distance_scene->push_back(scene->points[i]);
            }
        }
        scene.swap(distance_scene);


        int point_num = 0;
        Eigen::Vector3f *positions = nullptr;
        Eigen::Vector3f *colors = nullptr;
        Eigen::Vector3f *normals = nullptr;
        float *radii = nullptr;

        if (!readPly(point_num, positions, colors, normals, radii, target_file_path)) {
            LOG(FATAL) << "read ply file error !";
            return -1;
        }

        float x,y,z;
        x =  positions[0][0];
        y = positions[0][1];
        z = positions[0][2];

        for (int i = 0; i < point_num; i++) {
            pcl::PointXYZ pt;
            pt.x = positions[i][0]-x;
            pt.y = positions[i][1]-y;
            pt.z = positions[i][2]-z;
            model->push_back(pt);
        }
    }


    // 设置分辨率
    if (use_cloud_resolution_)
    {
        float resolution = static_cast<float> (computeCloudResolution (model));
        if (resolution != 0.0f)
        {
            model_ss_   *= resolution;
            scene_ss_   *= resolution;
            rf_rad_     *= resolution;
            descr_rad_  *= resolution;
            cg_size_    *= resolution;
        }

        std::cout << "Model resolution:       " << resolution << std::endl;
        std::cout << "Model sampling size:    " << model_ss_ << std::endl;
        std::cout << "Scene sampling size:    " << scene_ss_ << std::endl;
        std::cout << "LRF support radius:     " << rf_rad_ << std::endl;
        std::cout << "SHOT descriptor radius: " << descr_rad_ << std::endl;
        std::cout << "Clustering bin size:    " << cg_size_ << std::endl << std::endl;
    }

    LOG(INFO) << "Downsampling !!";
    pcl::VoxelGrid<pcl::PointXYZ> grid;
    float leaf = 0.005f;
    grid.setLeafSize(leaf, leaf, leaf);
    grid.setInputCloud (scene);
    grid.filter (*scene);

    grid.setInputCloud (model);
    grid.filter (*model);

    LOG(INFO) << "After downsample, scene point number is " << scene->size();
    LOG(INFO) << "After downsample, target point number is " << model->size();

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (scene);

    std::vector<pcl::PointIndices> cluster_indices;

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (0.02);
    ec.setMinClusterSize (100);
    ec.setMaxClusterSize (100000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (scene);
    ec.extract (cluster_indices);

    pcl::PointCloud<pcl::PointXYZ>::Ptr process_scene(new pcl::PointCloud<pcl::PointXYZ>);

    process_scene = getChildCloudByIndicesFromOriginal<pcl::PointXYZ>(scene,cluster_indices[0]);

    LOG(INFO) << "After euclidean cluster, process scene point number is " << process_scene->size();

    scene.swap(process_scene);


    //计算法线 normalestimationomp估计局部表面特性在每个三个特点，分别表面的法向量和曲率，平行，使用OpenMP标准。//初始化调度程序并设置要使用的线程数（默认为0）。
    pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
    norm_est.setKSearch (10);       //设置K邻域搜索阀值为10个点
    norm_est.setInputCloud (model);  //设置输入点云
    norm_est.compute (*model_normals);   //计算点云法线

    norm_est.setInputCloud (scene);
    norm_est.compute (*scene_normals);

    //均匀采样点云并提取关键点
    pcl::UniformSampling<PointType> uniform_sampling;
    uniform_sampling.setInputCloud (model);  //输入点云
    uniform_sampling.setRadiusSearch (model_ss_);   //设置半径
    uniform_sampling.filter (*model_keypoints);   //滤波
    LOG(INFO) << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size ();

    uniform_sampling.setInputCloud (scene);
    uniform_sampling.setRadiusSearch (scene_ss_);
    uniform_sampling.filter (*scene_keypoints);
    LOG(INFO) << "Scene total points: " << scene->size () << "; Selected Keypoints: " << scene_keypoints->size ();


    //为关键点计算描述子
    pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
    descr_est.setRadiusSearch (descr_rad_);  //设置搜索半径

    descr_est.setInputCloud (model_keypoints);  //输入模型的关键点
    descr_est.setInputNormals (model_normals);  //输入模型的法线
    descr_est.setSearchSurface (model);         //输入的点云
    descr_est.compute (*model_descriptors);     //计算描述子

    descr_est.setInputCloud (scene_keypoints);  //同理
    descr_est.setInputNormals (scene_normals);
    descr_est.setSearchSurface (scene);
    descr_est.compute (*scene_descriptors);

    //  使用Kdtree找出 Model-Scene 匹配点
    pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

    pcl::KdTreeFLANN<DescriptorType> match_search;   //设置配准的方法
    match_search.setInputCloud (model_descriptors);  //输入模板点云的描述子

//每一个场景的关键点描述子都要找到模板中匹配的关键点描述子并将其添加到对应的匹配向量中。
    for (size_t i = 0; i < scene_descriptors->size (); ++i)
    {
        std::vector<int> neigh_indices (1);   //设置最近邻点的索引
        std::vector<float> neigh_sqr_dists (1); //申明最近邻平方距离值
        if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])) //忽略 NaNs点
        {
            continue;
        }
        int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
//scene_descriptors->at (i)是给定点云 1是临近点个数 ，neigh_indices临近点的索引  neigh_sqr_dists是与临近点的索引

        if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) // 仅当描述子与临近点的平方距离小于0.25（描述子与临近的距离在一般在0到1之间）才添加匹配
        {
//neigh_indices[0]给定点，  i  是配准数  neigh_sqr_dists[0]与临近点的平方距离
            pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
            model_scene_corrs->push_back (corr);   //把配准的点存储在容器中
        }
    }

    LOG(INFO) << "Correspondences found: " << model_scene_corrs->size ();


    //  实际的配准方法的实现
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
    std::vector<pcl::Correspondences> clustered_corrs;

    //  使用 Hough3D算法寻找匹配点
    if (use_hough_)
    {
        //
        //  Compute (Keypoints) Reference Frames only for Hough
        //计算参考帧的Hough（也就是关键点）
        pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
        pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());
        //特征估计的方法（点云，法线，参考帧）
        pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
        rf_est.setFindHoles (true);
        rf_est.setRadiusSearch (rf_rad_);   //设置搜索半径

        rf_est.setInputCloud (model_keypoints);  //模型关键点
        rf_est.setInputNormals (model_normals); //模型法线
        rf_est.setSearchSurface (model);    //模型
        rf_est.compute (*model_rf);      //模型的参考帧

        rf_est.setInputCloud (scene_keypoints);  //同理
        rf_est.setInputNormals (scene_normals);
        rf_est.setSearchSurface (scene);
        rf_est.compute (*scene_rf);

        //  Clustering聚类的方法

        //对输入点与的聚类，以区分不同的实例的场景中的模型
        pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
        clusterer.setHoughBinSize (cg_size_);//霍夫空间设置每个bin的大小
        clusterer.setHoughThreshold (cg_thresh_);//阀值
        clusterer.setUseInterpolation (true);
        clusterer.setUseDistanceWeight (false);

        clusterer.setInputCloud (model_keypoints);
        clusterer.setInputRf (model_rf);   //设置输入的参考帧
        clusterer.setSceneCloud (scene_keypoints);
        clusterer.setSceneRf (scene_rf);
        clusterer.setModelSceneCorrespondences (model_scene_corrs);//model_scene_corrs存储配准的点

        //clusterer.cluster (clustered_corrs);辨认出聚类的对象
        clusterer.recognize (rototranslations, clustered_corrs);
    }
    else // Using GeometricConsistency  或者使用几何一致性性质
    {
        pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer;
        gc_clusterer.setGCSize (cg_size_);   //设置几何一致性的大小
        gc_clusterer.setGCThreshold (cg_thresh_); //阀值

        gc_clusterer.setInputCloud (model_keypoints);
        gc_clusterer.setSceneCloud (scene_keypoints);
        gc_clusterer.setModelSceneCorrespondences (model_scene_corrs);

        //gc_clusterer.cluster (clustered_corrs);
        gc_clusterer.recognize (rototranslations, clustered_corrs);
    }

    //输出的结果  找出输入模型是否在场景中出现
    LOG(INFO) << "Model instances found: " << rototranslations.size ();
    for (size_t i = 0; i < rototranslations.size (); ++i)
    {
        std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
        std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size () << std::endl;

        // 打印处相对于输入模型的旋转矩阵与平移矩阵
        Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);
        Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);

        printf ("\n");
        printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
        printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
        printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
        printf ("\n");
        printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));


        pcl::transformPointCloud(*model,*result,rototranslations[i]);

        pcl::io::savePCDFileASCII(FLAGS_output_path + to_string(i) + "_seg_result.pcd", *result);

    }


    pcl::PointCloud<PointType>::Ptr off_scene_model (new pcl::PointCloud<PointType> ());
    pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints (new pcl::PointCloud<PointType> ());

    if (show_correspondences_ || show_keypoints_)  //可视化配准点
    {
        //  We are translating the model so that it doesn't end in the middle of the scene representation
        //就是要对输入的模型进行旋转与平移，使其在可视化界面的中间位置
        pcl::transformPointCloud (*model, *off_scene_model, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
        pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));  //因为模型的位置变化了，所以要对特征点进行变化

    }


    for (size_t i = 0; i < rototranslations.size (); ++i)
    {
        pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
        pcl::transformPointCloud (*model, *rotated_model, rototranslations[i]);
        //把model转化为rotated_model  <Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >   rototranslations是射影变换矩阵
        std::stringstream ss_cloud;
        ss_cloud << "instance" << i;


//        if (show_correspondences_)   //显示配准连接
//        {
//            for (size_t j = 0; j < clustered_corrs[i].size (); ++j)
//            {
//                std::stringstream ss_line;
//                ss_line << "correspondence_line" << i << "_" << j;
//                PointType& model_point = off_scene_model_keypoints->at (clustered_corrs[i][j].index_query);
//                PointType& scene_point = scene_keypoints->at (clustered_corrs[i][j].index_match);
//
//                //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
//                viewer.addLine<PointType, PointType> (model_point, scene_point, 0, 255, 0, ss_line.str ());
//            }
//        }
    }


    return (0);
}