//
// Created by zhachanghai on 20-9-27.
//

#include "base_io.h"

namespace zjlab{
    namespace pose{
        using namespace std;

        bool readPly(
                int &pt_num,
                     Eigen::Vector3f *&positions,
                     Eigen::Vector3f *&colors,
                     Eigen::Vector3f *&normals,
                     float *&radii,
                     string &file_path) {

            ifstream infile;
            infile.open(file_path.data());
            string line_data;


            bool has_position = false;
            bool has_color = false;
            bool has_normal = false;
            bool has_radius = false;
            bool finish_head = false;

            try {
                if (!infile) {
                    throw std::runtime_error("Error: ply file not found!");
                }

                //
                getline(infile, line_data);
                if (line_data != "ply") {
                    throw std::runtime_error("Error: wrong file format!");
                }

                getline(infile, line_data);
                bool isASCII = false;
                if (line_data.find("ascii") == std::string::npos) {
                    if (line_data.find("binary") == std::string::npos)
                        throw std::runtime_error("Error: wrong file format!");
                    else
                        isASCII = false;
                } else
                    isASCII = true;

                while (!finish_head) {
                    getline(infile, line_data);
                    if (!has_position) {
                        if (line_data.find("float x") != std::string::npos)
                            has_position = true;
                    }
                    if (!has_color) {
                        if (line_data.find("uchar red") != std::string::npos)
                            has_color = true;
                    }
                    if (!has_normal) {
                        if (line_data.find("float nx") != std::string::npos)
                            has_normal = true;
                    }
                    if (!has_radius) {
                        if (line_data.find("float radius") != std::string::npos)
                            has_radius = true;
                    }
                    if (pt_num == 0) {
                        if (line_data.find("vertex") != std::string::npos) {
                            std::istringstream tempLine(line_data);
                            std::string tempLinePrefix;
                            tempLine >> tempLinePrefix;
                            tempLine >> tempLinePrefix;
                            tempLine >> pt_num;
                        }
                    }
                    if (line_data.find("end") != std::string::npos)
                        finish_head = true;
                }
                if (!has_position)
                    throw std::runtime_error("Error: this ply file doesn't have a position property!");

                //TODO zch : meshlab output ply color may be something error
                has_color = false;

                positions = new Eigen::Vector3f[pt_num];
                colors = has_color ? new Eigen::Vector3f[pt_num] : nullptr;
                normals = has_normal ? new Eigen::Vector3f[pt_num] : nullptr;
                radii = has_radius ? new float[pt_num] : nullptr;

                // point num without nan
                int real_pt_num = 0;
                if (isASCII) {
                    //TODO zch: add ascii data read code
                    //real_pt_num = readAsciiData(pointNum, positions, colors, normals, radii);
                } else {
                    real_pt_num = pt_num;
                    for (int i = 0; i < real_pt_num;) {
                        infile.read(reinterpret_cast<char *>(&positions[i][0]), sizeof(float));
                        infile.read(reinterpret_cast<char *>(&positions[i][1]), sizeof(float));
                        infile.read(reinterpret_cast<char *>(&positions[i][2]), sizeof(float));

                        if (has_color) {
                            unsigned char tempColor;
                            infile.read(reinterpret_cast<char *>(&tempColor), sizeof(unsigned char));
                            colors[i][0] = tempColor;
                            infile.read(reinterpret_cast<char *>(&tempColor), sizeof(unsigned char));
                            colors[i][1] = tempColor;
                            infile.read(reinterpret_cast<char *>(&tempColor), sizeof(unsigned char));
                            colors[i][2] = tempColor;
                        }

                        if (has_normal) {
                            infile.read(reinterpret_cast<char *>(&normals[i][0]), sizeof(float));
                            infile.read(reinterpret_cast<char *>(&normals[i][1]), sizeof(float));
                            infile.read(reinterpret_cast<char *>(&normals[i][2]), sizeof(float));
                            if (std::isnan(normals[i][0]) || std::isnan(normals[i][1]) || std::isnan(normals[i][2])) {
                                --real_pt_num;
                                continue;
                            }
                            normals[i] *= -1;
                        }

                        if (has_radius)
                            infile.read(reinterpret_cast<char *>(&radii[i]), sizeof(float));
                        ++i;
                    }

                    if (pt_num != real_pt_num) {
                        //TOOD zch : deal with vector size of positions & colors & normals
                        clog << "point num has nan data !" << endl;
                    }
                }
            }
            catch (std::runtime_error err) {
                cout << "open file error" << endl;
                cerr << err.what() << endl;
                infile.close();
                return false;

            }
            infile.close();
            return true;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr depth2cloud(cv::Mat depth_image, float fx, float fy, float cx, float cy) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>());
            cloud_ptr->width = depth_image.cols;
            cloud_ptr->height = depth_image.rows;
            cloud_ptr->is_dense = false;
            for (int y = 0; y < depth_image.rows; ++y) {
                for (int x = 0; x < depth_image.cols; ++x) {
                    pcl::PointXYZ pt;
                    if (depth_image.at<unsigned short>(y, x) != 0) {
                        if (depth_image.at<unsigned short>(y, x) > 2000) {
                            pt.z = 0;
                            pt.x = 0;
                            pt.y = 0;
                        } else {
                            pt.z = depth_image.at<unsigned short>(y, x);
                            pt.z /= 1000;
                            pt.x = (x - cx) * pt.z / fx;
                            pt.y = (y - cy) * pt.z / fy;
                        }
                        cloud_ptr->points.push_back(pt);
                    } else {
                        pt.z = 0;
                        pt.x = 0;
                        pt.y = 0;
                        cloud_ptr->points.push_back(pt);
                    }
                }
            }
            return cloud_ptr;
        }


        void getVideoFrame2Cloud(const string &video_input, const intrinsicsDepth &intrinsics_depth, const string &output_path) {

            int total = 0;
            openni::OpenNI::initialize();
            openni::Device fromonifile;
            fromonifile.open(video_input.data());
            openni::PlaybackControl *pController = fromonifile.getPlaybackControl();

            openni::VideoStream streamDepth;
            openni::VideoFrameRef frameDepth;


            //验证是否有彩色传感器（是否有彩色视频）和建立与设备想关联的视频流
            if (fromonifile.hasSensor(openni::SENSOR_COLOR)) {
                if (streamDepth.create(fromonifile, openni::SENSOR_COLOR) == openni::STATUS_OK) {
                    LOG(INFO) << "建立视频流成功";
                } else {
                    LOG(FATAL) << "建立视频流没有成功";
                }
            } else if (fromonifile.hasSensor(openni::SENSOR_DEPTH)) {
                if (streamDepth.create(fromonifile, openni::SENSOR_DEPTH) == openni::STATUS_OK) {
                    LOG(INFO) << "建立视频流成功";
                } else {
                    LOG(FATAL) << "建立视频流没有成功";
                }
            } else {
                LOG(FATAL) << "视频流既没有RGB也没有深度图像，请检查视频 ！！";
            }

            //获取总的视频帧数并将该设备的速度设为-1以便能留出足够的时间对每一帧进行处理、显示和保存
            total = pController->getNumberOfFrames(streamDepth);
            pController->setSpeed(-1);

            LOG(INFO) << " frames number is " << total;
            //开启视频流
            streamDepth.start();
            for (int i = 1; i <= total; ++i) {
                //读取视频流的当前帧
                streamDepth.readFrame(&frameDepth);

                LOG_EVERY_N(INFO, 20) << "当前正在读的帧数是： " << frameDepth.getFrameIndex() << "  当前的循环次数是：  " << i;

                const cv::Mat mImageDepth(frameDepth.getHeight(), frameDepth.getWidth(), CV_16UC1,
                                          (void *) frameDepth.getData());
                int iMaxDepth = streamDepth.getMaxPixelValue();
                cv::Mat mScaledDepth;
                mImageDepth.convertTo(mScaledDepth, CV_8U, 255.0 / iMaxDepth);

//        for(int m = 0; m < mImageDepth.cols; m++){
//            for(int n = 0; n < mImageDepth.rows; n++){
//                if (mImageDepth.at<unsigned short>(n,m) > 80){
//                    LOG(INFO) << "depth is " << mImageDepth.at<unsigned short>(n,m);
//                }
//            }
//        }

                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
                cloud = depth2cloud(mImageDepth, intrinsics_depth.fx, intrinsics_depth.fy, intrinsics_depth.cx,
                                    intrinsics_depth.cy);

                pcl::io::savePCDFileASCII(output_path + "pcl_point/" + to_string(i) + " _test.pcd", *cloud);


                cv::imwrite(output_path + "image/" + to_string(i) + "_depth.png", mScaledDepth);

            }
        }

        void image2Video(const string &input_path, const string &output_video_path) {
            string s_image_name;
            cv::VideoWriter writer;
            int isColor = 1;//不知道是干啥用的
            int frame_fps = 20;
            int frame_width = 640;
            int frame_height = 480;
            writer = cv::VideoWriter(output_video_path, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), frame_fps, cv::Size(frame_width,frame_height), isColor);
            cout << "frame_width is " << frame_width << endl;
            cout << "frame_height is " << frame_height << endl;
            cout << "frame_fps is " << frame_fps << endl;
            int num = 90;//输入的图片总张数
            int i = 1;
            cv::Mat img;
            while (i <= num)
            {
                s_image_name = input_path + to_string(i) + "_image.jpg";
                img = cv::imread(s_image_name);//读入图片
                if (!img.data)//判断图片调入是否成功
                {
                    cout << s_image_name << endl;
                    cout << "Could not load image file...\n" << endl;
                }
                //写入
                writer.write(img);
//        if (i == 1 || i == 12 ||i == 19 ||i == 25 ||i == 35 ||i == 41 ||i == 51 ||i == 62 ||i == 71 ||i == 81){
//            for(int n = 0 ; n < 10 ; n++){
//                writer.write(img);
//            }
//        }
                if (cv::waitKey(30) == 27 || i == 160)
                {
                    cout << "按下ESC键" << endl;
                    break;
                }
                i++;
            }
        }

        Eigen::Vector3f rotationAboutX(Eigen::Vector3f &origin, float &theta){
            Eigen::Matrix3f rotation_x;
            rotation_x << 1, 0, 0,
                          0, cos(theta), -sin(theta),
                          0, sin(theta), cos(theta);
            Eigen::Vector3f output_vector;
            output_vector = rotation_x * origin;
            return output_vector;
        }

        Eigen::Vector3f rotationAboutY(Eigen::Vector3f &origin, float &theta){
            Eigen::Matrix3f rotation_y;
            rotation_y << cos(theta), 0, sin(theta),
                    0, 1, 0,
                    -sin(theta), 0, cos(theta);
            Eigen::Vector3f output_vector;
            output_vector = rotation_y * origin;
            return output_vector;
        }

        Eigen::Vector3f rotationAboutZ(Eigen::Vector3f &origin, float &theta){
            Eigen::Matrix3f rotation_z;
            rotation_z << cos(theta), -sin(theta), 0,
                          sin(theta), cos(theta), 0,
                          0, 0, 1;
            Eigen::Vector3f output_vector;
            output_vector = rotation_z * origin;
            return output_vector;

        }

    }
}
