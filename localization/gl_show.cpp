#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
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
using json = nlohmann::json;

void readJsonFile(const string& json_path){
    json j;
    j["x"] = "100";
    j["y"] = "200";
    ofstream json_file(json_path);
    json_file << j;
    json_file.close();
}



int main(int argc, char** argv)
{
    string infile_pic = "/home/zhachanghai/test_grap/video/";
    string outfile_video = "/home/zhachanghai/test_grap/demo_show.avi";
//    image2Video(infile_pic,outfile_video);
    string json_file = "/home/zhachanghai/test.json";
    readJsonFile(json_file);
    return 0;
}
