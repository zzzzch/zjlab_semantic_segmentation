/*
 * @Description: 
 * @Version: 1.0
 * @Autor: wxchen
 * @Date: 2020-09-20 15:41:19
 * @LastEditTime: 2020-09-22 20:01:28
 */
#include <iostream>
#include <chrono>
#include <queue>
#include <map>
#include <unistd.h>
#include <GxIAPI.h>
#include <DxImageProc.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <thread>
#include <mutex>

int64_t PayLoadSize;
int64_t Width;
int64_t Height;
uint64_t FrameID;
uint64_t TimeStamp;
double s=1000000000;
cv::Mat mImg;

char *BufferRaw = NULL;
char *BufferRGB = NULL;

std::queue<cv::Mat> img_queue;
std::queue<std::pair<std::string,cv::Mat>> img_queue_info;
std::mutex img_mutex;


void run()
{
    int idx = 0;
    std::string str = "/home/zhachanghai/daheng_callback_test/image/";
    int i = 1;
    while (i > 0)
    {
        std::cout << "queue " << img_queue_info.size() << std::endl;
        if(!img_queue_info.empty())
        {
            cv::Mat image;
            img_mutex.lock();

            char tmp[20];
            sprintf(tmp, "%06d", idx++);
            std::string id(tmp);

//             std::cout << "writing " << idx -1 << " to " <<  file_path <<  std::endl;
            std::string timestamp = img_queue_info.front().first;
            std::string file_path = str + id + "_" + timestamp  + ".png";
            image = img_queue_info.front().second;

            img_queue_info.pop();
            img_mutex.unlock();
            cv::imwrite(file_path, image);
//            i++;
//            std::cout << "number " << i << std::endl;
        }
        std::this_thread::sleep_for (std::chrono::milliseconds(2));
    }
}


//图 像 回 调 处 理 函 数
static void GX_STDC OnFrameCallbackFun(GX_FRAME_CALLBACK_PARAM* pFrame)
{
if (pFrame->status == GX_FRAME_STATUS_SUCCESS)
{
//对 图 像 进 行 某 些 操 作
    auto start = std::chrono::steady_clock::now();
    FrameID = pFrame->nFrameID;
    TimeStamp = pFrame->nTimestamp;
    double t = static_cast<double>(TimeStamp) / s;
    std::ostringstream tmpTime;
    tmpTime << t;

    img_mutex.lock();

    // std::cout << Width << " " << Height << std::endl;

    memcpy(BufferRaw, pFrame->pImgBuf, pFrame->nImgSize);
    // std::cout << FrameID << std::endl;
    DxRaw8toRGB24(BufferRaw, BufferRGB, Width, Height, RAW2RGB_NEIGHBOUR,DX_PIXEL_COLOR_FILTER(BAYERBG),false);


    // cv::Mat mImg_tmp;
    // mImg_tmp.create(Height, Width, CV_8UC3);
    memcpy(mImg.data, BufferRGB, Width*Height*3);
    std::string str = "/home/zhachanghai/daheng_callback_test/image/";
    std::ostringstream tmp;
    tmp << FrameID;
    str += tmp.str();

//     cv::imwrite(str+"_.png", mImg);
    auto end0 = std::chrono::steady_clock::now();
    auto ms0 = std::chrono::duration_cast<std::chrono::milliseconds>(end0-start).count();
    // std::cout << "cb0 time cost: " << ms0 << "ms." << std::endl;
    auto data = std::make_pair(tmpTime.str(), mImg.clone());
    img_queue_info.push(data);
    img_mutex.unlock();

//    cv::namedWindow("test");
//    cv::imshow("test", mImg);
//    cv::waitKey(3);
    // auto end = std::chrono::steady_clock::now();
    // auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    // std::cout << "cb time cost: " << ms << "ms." << std::endl;


}
return;
}
int main(int argc, char* argv[])
{
GX_STATUS status = GX_STATUS_SUCCESS;
GX_DEV_HANDLE hDevice = NULL;
GX_OPEN_PARAM stOpenParam;
uint32_t nDeviceNum = 0;
//初 始 化 库
status = GXInitLib();
if (status!= GX_STATUS_SUCCESS)
{
return 0;
}
//枚 举 设 备 列 表
status = GXUpdateDeviceList(&nDeviceNum, 1000);
if ((status!= GX_STATUS_SUCCESS)||(nDeviceNum<= 0))
{
return 0;
}

std::thread save_thread(run);

//打 开 设 备
stOpenParam.accessMode = GX_ACCESS_EXCLUSIVE;
stOpenParam.openMode = GX_OPEN_INDEX;
stOpenParam.pszContent = "1";
status = GXOpenDevice(&stOpenParam, &hDevice);

status = GXSetEnum(hDevice, GX_ENUM_DEVICE_LINK_THROUGHPUT_LIMIT_MODE, GX_DEVICE_LINK_THROUGHPUT_LIMIT_MODE_OFF);
status = GXSetInt(hDevice, GX_INT_WIDTH, 1920);
status = GXSetInt(hDevice, GX_INT_HEIGHT, 1200);
status = GXSetFloat(hDevice, GX_FLOAT_EXPOSURE_TIME, 2000.0);
// gain; balance_white; frame_rate
status = GXSetEnum(hDevice, GX_ENUM_GAIN_AUTO, GX_GAIN_AUTO_ONCE);
status = GXSetEnum(hDevice,GX_ENUM_BALANCE_WHITE_AUTO,GX_BALANCE_WHITE_AUTO_ONCE);
status = GXSetEnum(hDevice, GX_ENUM_ACQUISITION_FRAME_RATE_MODE,GX_ACQUISITION_FRAME_RATE_MODE_ON);
status = GXSetFloat(hDevice, GX_FLOAT_ACQUISITION_FRAME_RATE, 140.0);

// get size
status = GXGetInt(hDevice, GX_INT_PAYLOAD_SIZE, &PayLoadSize);

//get width
status = GXGetInt(hDevice, GX_INT_WIDTH, &Width);

//get height
status = GXGetInt(hDevice, GX_INT_HEIGHT, &Height);

mImg.create(Height, Width, CV_8UC3);


BufferRGB = new char[Width*Height*3];

BufferRaw = new char[PayLoadSize];

if (status == GX_STATUS_SUCCESS)
{
//注 册 图 像 处 理 回 调 函 数
status = GXRegisterCaptureCallback(hDevice, NULL,OnFrameCallbackFun);
//发 送 开 采 命 令
status = GXSendCommand(hDevice, GX_COMMAND_ACQUISITION_START);
//---------------------
//
//在 这 个 区 间 图 像 会 通 过 OnFrameCallbackFun 接 口 返 给 用 户
//save_thread.join();
    usleep(8000000);
//getchar();
std::cout << FrameID << std::endl;
//
//---------------------
//发 送 停 采 命 令
status = GXSendCommand(hDevice, GX_COMMAND_ACQUISITION_STOP);
//注 销 采 集 回 调
status = GXUnregisterCaptureCallback(hDevice);
}
status = GXCloseDevice(hDevice);
status = GXCloseLib();

//    run();
save_thread.join();
return 0;
}