#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/types_c.h"
#include <opencv2/highgui/highgui_c.h>

#include "base_io.h"

using namespace zjlab;
using namespace pose;
using namespace std;

int main(int argc, char** argv)
{
    string infile_pic = "/home/zhachanghai/test_grap/video/";
    string outfile_video = "/home/zhachanghai/test_grap/demo_show.avi";
    image2Video(infile_pic,outfile_video);

    return 0;
}
