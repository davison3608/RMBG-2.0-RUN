#include "cvpacked.h"
//
//! cvbase对象
void cvbase::Show(CMat& src) const noexcept
{
    using namespace cv;
    if (src.channels() != 3) return ;
    CMat showmat;
    cvtColor(src, showmat, COLOR_RGB2BGR);
    // 分离线程
    std::thread([showmat]() {
    const char *winname="Display RGB Image";
    namedWindow(winname, WINDOW_AUTOSIZE);
    imshow(winname, showmat);
    // 异步等待按键事件
    waitKey(0);
    // 按键后关闭窗口
    destroyWindow(winname);
    }).detach(); 
}

void cvbase::SaveMat(path save, CMat& value) const noexcept
{
    using namespace std;
    ofstream file(save);
    if (!file.is_open()) 
    { cout<<"保存路径无效\n"; return ; }
    file.close();

    using namespace cv;
    if (!imwrite(save, value)) {
    cerr<<"推理叠加图保存失败 !\n";
    return ;
    }
    return ;
}

void cvbase::LoadMat(path load, CMat& src)
{
    using namespace std;
    ifstream file(load);
    if (!file.is_open()) 
    { src=cv::Mat(); return ; }
    
    src=cv::imread(load); 
    CMat src0;
    CMat src1;
    
    //rgb cv_32fc3
    cv::cvtColor(src, src0, cv::COLOR_BGR2RGB);
    src0.convertTo(src1, PRECISION);
    src=src1;
    return ;
}

void cvbase::ResizeMat(CMat& src, Shapes CHW) const noexcept
{ 
    if (src.empty()) {
    std::cerr<<"传入图像空！";
    return ;
    }
    //扭曲方式变形
    cv::Size size(CHW[2], CHW[1]); 
    cv::resize(src, src, size); 
}
//
//! cvmat 对象
void cvmat::ConvertMat(RGBMat& srchwc, Shapes CHW, RGBMat& valuechw) const
{
    if (srchwc.channels() != 3) return ;
    if (srchwc.channels() != CHW[0]) return ;
    if (srchwc.rows != CHW[1]) return ;
    if (srchwc.cols != CHW[2]) return ;

    usize c=CHW[0];
    usize h=CHW[1];
    usize w=CHW[2];
    using namespace cv;
    using namespace std;

    valuechw=Mat();
    if (srchwc.type() != MATTYPE) return ;

    // 每个元素为单通道图
    vector<Mat> channelmats(c);
    // 分离
    cv::split(srchwc, channelmats);
    // 创建矩阵 chw格式 每个元素为单个通道的值
    valuechw.create(c, h * w, MASKTYPE);
    // 填充所有通道
    for (int i=0; i<c; i++) {
    // 单通道展平
    Mat flatten=channelmats[i].reshape(1, 1);
    // 获取待填充图像单个通道
    Mat singlechannelmat=valuechw.row(i);
    // 填充到这个通道
    flatten.copyTo(singlechannelmat);
    }
}

void cvmat::FlattenCHW(RGBMat& srchwc, Datas& dataschw) const
{
    using namespace std;
    using namespace cv;
    // 均值 方差参数
    constexpr array<precision, 3> mean = {0.485f, 0.456f, 0.406f};
    constexpr array<precision, 3> stddev = {0.229f, 0.224f, 0.225f};

    if (srchwc.channels() != 3) return ;
    if (srchwc.type() != MATTYPE) return ;
    usize row=srchwc.rows;
    usize col=srchwc.cols;
    usize c=srchwc.channels();

    // 创建参数对象
    Mat meanMat(1, 1, CV_64F, Scalar(mean[0], mean[1], mean[2]));
    Mat stdMat(1, 1, CV_64F, Scalar(stddev[0], stddev[1], stddev[2]));
    // 归一化到副本 
    Mat normat;
    srchwc.convertTo(normat, CV_64F, 1.0f/255.0f);

    // 标准化处理
    cv::subtract(normat, meanMat, normat);
    cv::divide(normat, stdMat, normat);
    normat.convertTo(normat, CV_32FC3);

    // 创建矩阵 chw格式
    Mat normalizematchw;
    // 转换成chw
    this->ConvertMat(normat, {c, row, col}, normalizematchw);

    // 开辟容器
    vector<precision> normalizedatas;
    normalizedatas.reserve(c * row * col);

    auto normalizedatas_itor=normalizedatas.begin();
    for (int i=0; i<c; i++) {
        // 每单个通道指针
        precision *singlechannelptr=normalizematchw.ptr<precision>(i);
        // 插入到容器中
        normalizedatas_itor=normalizedatas.insert
        (normalizedatas_itor, singlechannelptr, singlechannelptr + (row * col));
    }

    //移交资源
    if (normalizedatas.size() != c * row * col) 
    { cerr<<"展平归一化到容器数据计算失败\n"; dataschw=move(vector<float>()); }
    dataschw=move(normalizedatas);
}
//
//! cvmask对象
void cvmask::ConvertMat(GrayMat& srcchw, Shapes CHW, GrayMat& valuehwc) const 
{       
    usize channels = CHW[0];
    usize row = CHW[1];
    usize col = CHW[2];
    valuehwc=GrayMat();

    // 重塑为 HWC 格式（row×col×1）
    valuehwc=srcchw.reshape(channels, row).clone();
    return ;
}

void cvmask::RecoverCHW(Datas& dataschw, HWShapes HW, GrayMat& valuehwc) const 
{
    usize row=HW[0];
    usize col=HW[1];
    usize channels=1;
    using namespace std;
    using namespace cv;

    GrayMat denorchw=GrayMat();
    //确保单通道一维数据
    if (dataschw.size() != row*col) { return ; }
    //创建chw单通道矩阵
    denorchw.create(channels, row * col, MASKTYPE);

    constexpr array<precision, 3> mean = {0.485f, 0.456f, 0.406f};
    constexpr array<precision, 3> stddev = {0.229f, 0.224f, 0.225f};

    precision* denorchw_ptr=denorchw.ptr<precision>(0);  // 获取第 0 行指针
    for (int h=0; h<row; ++h) {
    for (int w=0; w<col; ++w) {
        usize idx=h * col + w;  // 计算空间索引（通道=0）
        precision prob=static_cast<precision>(min(max(dataschw[idx] * 255.0f, 0.0f), 255.0f));
        //覆盖原像素
        denorchw_ptr[idx]=prob;
    }
    }
    //转换回hwc格式
    this->ConvertMat(denorchw, {channels, row, col}, valuehwc);
    valuehwc.convertTo(valuehwc, CV_8UC1);
    //namedWindow("Mask Result", WINDOW_NORMAL);
    //resizeWindow("Mask Result", 512, 512);
    //imshow("Mask Result", valuehwc);
    //waitKey(0);
    return ;
}

void cvmask::MaskSegmentation(RGBMat& srcmat, GrayMat& srcmask, CMat& value) const
{
    usize row=srcmat.rows;
    usize col=srcmat.cols;
    if (row != srcmask.rows) return ;
    if (col != srcmask.cols) return ;
    value=CMat();

    using namespace cv;
    //原图转换
    cvtColor(srcmat, srcmat, COLOR_BGR2RGB);
    //创建hwc 4通道矩阵
    value.create(row, col, CV_8UC3);
    //原图 mask转为uchar
    srcmat.convertTo(srcmat, CV_8UC3);
    srcmask.convertTo(srcmask, CV_8UC1);

    for (int r=0; r<row; r++) {
        //当前行指针
        const uchar *srcmat_r_p=srcmat.ptr<uchar>(r);
        const uchar *srcmask_r_p=srcmask.ptr<uchar>(r);
        uchar *value_r_p=value.ptr<uchar>(r);
    for (int c=0; c<col; c++) {
        //计算当前列像素索引
        int src_idx=c * 3;
        int value_idx=c * 3;
        //比较掩码值
        uchar mask=srcmask_r_p[c];
        //复制像素值
        if (mask >= 128.0f) {
        value_r_p[value_idx]=srcmat_r_p[src_idx]; //r
        value_r_p[value_idx + 1]=srcmat_r_p[src_idx + 1]; //g
        value_r_p[value_idx + 2]=srcmat_r_p[src_idx + 2]; //b
        }
        //后景白色
        else 
        value_r_p[value_idx]=value_r_p[value_idx + 1]=value_r_p[value_idx + 2]=255;
    }
    }
    namedWindow("Seg Result", WINDOW_NORMAL);
    resizeWindow("Seg Result", 512, 512);
    //imshow("Seg Result", value);
    //waitKey(3500);
    destroyWindow("Seg Result");
    return ;
}
