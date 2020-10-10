#include <iostream>
#include <glog/logging.h>


int main (int argc, char **argv)
{

    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;
    FLAGS_stderrthreshold = 0;


    return (0);
}