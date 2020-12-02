#include <stdio.h>
//#include <cuda_runtime.h>
//#include <pcl/io/pcd_io.h>

//__global__ void kernal()
//{
//    printf("hello from GPU\n");
//}

int main()
{
//    printf("hello from CPU\n");
//    kernal << <1, 1 >> > ();
    return 0;
}


//13 int main(int argc, char** argv)
//14 {
//15     pcl::PointCloud<pcl::PointXYZRGB> cloud;
//16     pcl::gpu::DeviceArray<pcl::PointXYZRGB> cloud_device;
//17
//18
//19     cloud.width = 1;
//20     cloud.height =1;
//21     cloud.is_dense=false;
//22     cloud.points.resize(cloud.width*cloud.height);
//23
//24     std::vector<float> point_val;
//25
//26     for(size_t i=0; i<3*cloud.points.size(); ++i)
//27     {
//28         point_val.push_back(1024*rand()/(RAND_MAX+1.0f));
//29     }
//30
//31     for (size_t i = 0; i < cloud.points.size(); ++i) {
//32         cloud.points[i].x = point_val[3 * i];
//33         cloud.points[i].y = point_val[3 * i + 1];
//34         cloud.points[i].z = point_val[3 * i + 2];
//35     }
//36
//37     std::cout<<"cloud.points="<<cloud.points[0]<<std::endl;
//38
//39     cloud_device.upload(cloud.points);
//40
//41     cloud2GPU(cloud_device);
//42
//43     cloud_device.download(cloud.points);
//44
//45     std::cout<<"cloud.points="<<cloud.points[0]<<std::endl;
//46     return (0);
//47 }
