#ifndef CINCLUDE
#define CINCLUDE
#include <stdio.h>
#include <iostream>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <cstring>
#include <fstream>
#include <istream>
#include <vector>
#include <array>
#include <list>
#include <queue>
#include <map>
#include <thread>
#include <future>
#include <mutex>
#include <cassert>
#include <ios>
#include <functional>
#include <atomic>
#include <condition_variable>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cvconfig.h>
#include <opencv2/imgcodecs.hpp>

#endif

#ifndef CVPATH
#define CVPATH
class CVTypeAlien
{
public:
    CVTypeAlien() noexcept = default;

    virtual ~CVTypeAlien() noexcept = default;

    using usize = std::size_t;
    using size = ssize_t;
    using spath = std::string;
    using path = const char*;
    //! 图像精度
    using precision = _Float32;

    //! 图像CHW尺寸别名
    using Shapes = std::array<usize, 3>;
    using HWShapes = std::array<usize, 2>;
    //! cv图像对象
    using CMat = cv::Mat;
    using RGBMat = cv::Mat;
    using GrayMat = cv::Mat;
    //! cv图像展平容器
    using Datas = std::vector<precision>;

    using _mtx = std::mutex;
    using _conv = std::condition_variable;
    using _lock = std::unique_lock<_mtx>;
};

//! \brief 原图像与掩码图像精度
#define PRECISION  CV_32F
#define MATTYPE  CV_32FC3
#define MASKTYPE  CV_32FC1

class cvbase: virtual public CVTypeAlien
{
public:
    cvbase() = default;

    virtual ~cvbase() = default;

    using CVTypeAlien::CMat;
    using CVTypeAlien::RGBMat;
    using CVTypeAlien::GrayMat;
    using CVTypeAlien::path;
    using CVTypeAlien::Shapes;
    using CVTypeAlien::HWShapes;

    //! \brief 显示hwc格式图片 
    void Show(CMat& src) const noexcept;

    //! \brief 保存hwc格式图像
    void SaveMat(path save, CMat& value) const noexcept;

    //! \brief 加载hwc图像
    void LoadMat(path load, CMat& src);

    //! \brief 裁剪到指定尺寸
    void ResizeMat(CMat& src, Shapes CHW) const noexcept;

public:
    virtual void FlattenCHW(RGBMat& src, Datas& datas) const = 0;

    virtual void RecoverCHW(Datas& datas, HWShapes HW, GrayMat& value) const = 0;

    using CVTypeAlien::usize;
    using CVTypeAlien::size;
    using CVTypeAlien::precision;
protected:
    virtual void ConvertMat(CMat& src, Shapes CHW, CMat& value) const = 0;
};

#endif

#ifndef CVMAT
#define CVMAT
/**
 * \brief 图像预处理对象
 * \brief 图像hwc转换chw 数据展平归一化处理 
*/
class cvmat: virtual public cvbase
{
public:
    using CVTypeAlien::CMat;
    using CVTypeAlien::RGBMat;
    using CVTypeAlien::GrayMat;
    using CVTypeAlien::Shapes;
    using CVTypeAlien::HWShapes;
    using CVTypeAlien::Datas;

    cvmat() = default;

    ~cvmat() = default;

    //! \brief hwc格式展平归一化处理 返回chw转换后对象
    //! \param hwc格式mat对象 引用未处理容器 图形尺寸
    void FlattenCHW(RGBMat& srchwc, Datas& dataschw) const override;

    //! \brief 无需重写
    void RecoverCHW(Datas& datas, HWShapes HW, GrayMat& value) const override
    { return ; }

    using CVTypeAlien::usize;
    using CVTypeAlien::size;
    using CVTypeAlien::precision;
protected:
    //! \brief 彩色图hwc to chw 保持尺寸一致
    //! \param 引用hwc对象 引用chw对象
    void ConvertMat(RGBMat& srchwc, Shapes CHW, RGBMat& valuechw) const override;
};

#endif

#ifndef CVMASK
#define CVMASK
/**
 * \brief 原图像预掩码交互对象
 * \brief 图像chw转换hwc 原图与掩码交互
*/
class cvmask: virtual public cvbase
{
public:
    cvmask() = default;

    ~cvmask() = default;

    using CVTypeAlien::CMat;
    using CVTypeAlien::RGBMat;
    using CVTypeAlien::GrayMat;
    using CVTypeAlien::Shapes;
    using CVTypeAlien::HWShapes;
    using CVTypeAlien::Datas;

    //! \brief 无需重写
    void FlattenCHW(RGBMat& src, Datas& datas) const override
    { return ; }

    //! \brief 将推理后展平数据重塑为Mat对象
    //! \param 引用展平后连续数据 引用chw对象 图像尺寸
    void RecoverCHW(Datas& dataschw, HWShapes HW, GrayMat& valuehwc) const override;

    //! \brief 原图像与掩码交互 保持尺寸一致
    //! \param 引用hwc格式原图与掩码图 输出结果
    void MaskSegmentation(RGBMat& srcmat, GrayMat& srcmask, CMat& value) const;

    using CVTypeAlien::usize;
    using CVTypeAlien::size;
    using CVTypeAlien::precision;
protected:
    //! \brief 灰度图chw to hwc 保持尺寸一致
    //! \param 引用chw对象 引用hwc对象
    void ConvertMat(GrayMat& srcchw, Shapes CHW, GrayMat& valuehwc) const override;
};

#endif

#ifndef CTENSORRT
#define CTENSORRT
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d, code=%d(%s)\n", __FILE__, __LINE__, \
               err, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

class logger: public nvinfer1::ILogger
{
public:
logger() = default;

~logger() = default;

    //重写虚函数
    void log(Severity severity, const char* msg) noexcept override 
    {
    if(severity !=Severity::kINFO 
    && severity != Severity::kWARNING 
    && severity != Severity::kERROR 
    && severity != Severity::kINTERNAL_ERROR
    && severity != Severity::kVERBOSE)
        return;
    switch (severity)
    {
    case Severity::kINFO:
        printf("kINFO info is: %s\n",msg);
        break;
    case Severity::kWARNING:
        printf("kWARNING warning is: %s\n",msg);
        break;
    case Severity::kERROR:
        printf("kERROR error is: %s\n",msg);
        break;
    case Severity::kINTERNAL_ERROR:
        printf("kINTERNAL_ERROR internal error is: %s\n",msg);
        break;
    case Severity::kVERBOSE:
        printf("kVERBOSE is: %s\n",msg);
    default:
        break;
    }
    }
};

#endif
