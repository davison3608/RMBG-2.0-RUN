#include "cvpacked.h"
#include "qtpacked.h"

#ifndef NVMOD
#define NVMOD
//! \struct 队列节点描述
struct Qnode
{
    using CMat = CVTypeAlien::CMat;
    using RGBMat = CVTypeAlien::RGBMat;
    using GrayMat = CVTypeAlien::GrayMat;

    using size = CVTypeAlien::size;
    using spath = CVTypeAlien::spath;

    //! 该节点数据结构体
    SQMat val;    

    //! 指向下一节点指针
    Qnode *next;

    //! 原图像 
    RGBMat src;

    //! 保存路径
    spath save;

    //! 根据val初始化
    void init() noexcept;

    Qnode():
    next(nullptr), save("")
    { this->val.reset(); src=CMat(); }
};

/**
 * \brief 待推理的sqmat队列  
*/
class Qdeque: public CVTypeAlien
{
public:
    using CVTypeAlien::CMat;
    using CVTypeAlien::RGBMat;
    using CVTypeAlien::GrayMat;

    using _thread = std::thread;
    using CVTypeAlien::_mtx;
    using CVTypeAlien::_conv;    
    using CVTypeAlien::_lock;

    Qdeque();
 
    ~Qdeque();
    
    //! 异步接收缓冲区结构体
    void async_push();

    //! 返回首元素指针 size为0返回null
    Qnode *front_p();

    //! 插入元素 从队尾插入
    void push(SQMat& val);

    //! 删除首元素
    void front_pop(); 
    
    using CVTypeAlien::usize;
    using CVTypeAlien::size;
    
    //! 有效的节点数量
    usize node_size;

private:
    //! 指向首节点前一位指针 尾节点指针
    Qnode *first;
    Qnode *last;
    //! 线程锁
    _mtx que_mt;

private:
    //! 计算进程共享对象
    ProcessShare *_share;
};

//! \brief cu别名类
class RTTypeAlias: public CVTypeAlien
{
public:
    RTTypeAlias() noexcept = default;

    ~RTTypeAlias() noexcept = default;

    using _mtx = CVTypeAlien::_mtx;
    using _conv = CVTypeAlien::_conv;

    using _str = CVTypeAlien::spath;
    using _precision = CVTypeAlien::precision;
    using _flag = std::atomic<bool>;
    using _shapes = std::array<int, 3>;

    using _Datas = CVTypeAlien::Datas;
    using _CDatas = std::vector<char>;

    using _Builder = nvinfer1::IBuilder;
    using _Engine = nvinfer1::ICudaEngine;
    using _Context = nvinfer1::IExecutionContext;
    using _Run = nvinfer1::IRuntime;
    using _buffer = void*;

    using _custr = cudaStream_t;
    using _cuevt = cudaEvent_t;
};

//! \struct context对象描述
struct ContextStream 
{
    using _precision = RTTypeAlias::_precision;
    using _flag = RTTypeAlias::_flag;
    using _str = RTTypeAlias::_str;
    using _usize = std::size_t;
    using _size = ssize_t;
    using _shapes = RTTypeAlias::_shapes;
    using _CDatas = RTTypeAlias::_CDatas;

    using _custr = RTTypeAlias::_custr;
    using _cuevt = RTTypeAlias::_cuevt;

    using _Builder = RTTypeAlias::_Builder;
    using _Engine = RTTypeAlias::_Engine;
    using _Context = RTTypeAlias::_Context; 
    using _buffer = RTTypeAlias::_buffer;
    //! 输入输出信息描述
    const _str in="input";
    const _str out="output";
    _shapes chw_in={3, 1024, 1024};
    _shapes chw_out={1, 1024, 1024};
    _usize batchsize=1;

    //! 上下文指针 对象由trt创建
    _Context *context=nullptr;
    //! 缓冲区
    _buffer buffers[2]={nullptr};

    //! cuda流
    _custr stream;
    _cuevt start; 
    _cuevt end;

    //! 有效标志
    _flag isinit;
    //! 空闲标志
    _flag isfree;

    ContextStream() = default;
    ContextStream(const ContextStream&) = delete;
    ContextStream& operator=(const ContextStream&) = delete;

    //! 手动获取context指针
    _Context *&get();
    //! 手动创建
    void init(_Engine *&eng);
    //! 手动释放
    void destroy();
};

/**
 * \brief engine创建对象  
*/
class basemod: public RTTypeAlias
{
public:
    using RTTypeAlias::_Engine;
    using RTTypeAlias::_Run;
    using RTTypeAlias::_str;
    using RTTypeAlias::usize;
    using RTTypeAlias::_CDatas;

    basemod();

    ~basemod();

    //! \brief 返回引擎指针
    _Engine *&get() 
    { return this->engine; }

protected:
    _str path;
    logger log;
    _Run *run=nullptr;
    _Engine *engine=nullptr;
};

#endif

#ifndef THREADING 
#define THREADING
/**
 * \brief 并发线程池 
*/
template<int NUMTHREADS>
class basets: public RTTypeAlias
{
public:
    using RTTypeAlias::usize;
    using RTTypeAlias::size;

    using RTTypeAlias::_flag;
    using RTTypeAlias::_mtx;
    using RTTypeAlias::_lock;
    using RTTypeAlias::_conv;

    using RTTypeAlias::CMat;
    using RTTypeAlias::RGBMat;
    using RTTypeAlias::_shapes;

    basets() = delete;
    //! \brief new接口 启用async_recv线程
    basets(usize num);

    ~basets(); 

    using _thread = std::thread;
    using _base_func_p = void (basets::*)();
    using _run_func = std::function<void()>;
    using _TDatas = std::vector<_thread>;

    //! \brief 线程池执行
    virtual void init() {}

    //! \brief 线程并发函数
    virtual void Concurrent_exec() {}

    //! 待推理队列
    Qdeque que;
protected:
    //! \brief 线程推理关键函数
    virtual void exec_kernel(Qnode& qnode, usize& id) {}

private:
    //! \brief 绑定当前对象函数 返回执行函数
    //! \param 当前对象函数指针
    _run_func basets_funct_bind(_base_func_p func);

    //! \brief 异步启动 不断地接收待推理队列数据
    void async_recv();

protected:
    //! 线程容器
    _TDatas atts_vec;

    //! context数目 锁 通知信号 
    usize num;
    _mtx mt;
    _conv cv;

    //! 是否有空闲推理节点
    _flag isspare;
    //! 原图像处理接口对象
    cvmat *cvmat_handle=nullptr;
    //! 掩码处理接口对象
    cvmask *cvmask_handle=nullptr;
};

template<int NUMTHREADS>
class rtts: public basets<NUMTHREADS>
{
public:
    using typename basets<NUMTHREADS>::usize;
    using typename basets<NUMTHREADS>::size;
    using typename basets<NUMTHREADS>::_flag;
    using typename basets<NUMTHREADS>::_conv;
    using typename basets<NUMTHREADS>::_mtx;
    using typename basets<NUMTHREADS>::_lock;
    
    using typename basets<NUMTHREADS>::CMat;
    using typename basets<NUMTHREADS>::RGBMat;
    using typename basets<NUMTHREADS>::_shapes;

    using RTTypeAlias::_str;
    using RTTypeAlias::_precision;
    using RTTypeAlias::_buffer;
    using RTTypeAlias::_Datas;

    using RTTypeAlias::_Context;
    using RTTypeAlias::_custr;
    using RTTypeAlias::_cuevt;
    
    using _idx = std::atomic<usize>;
    using _rtts_func_p = void (rtts::*)(Qnode&, usize&);
    using _rtts_vfunc_p = void (rtts::*)();

    using typename basets<NUMTHREADS>::_thread;
    using typename basets<NUMTHREADS>::_run_func;
    
    rtts() = delete;
    //! \brief context序列全部init
    rtts(usize numcontexts);

    ~rtts();

    //! \brief 线程池执行
    void init() override;

    //! \brief 线程并发函数
    void Concurrent_exec() override;

private:
    //! \brief 绑定当前对象函数 返回执行函数
    //! \param 当前对象函数指针
    _run_func rtts_func_bind(_rtts_func_p func, Qnode& Node, usize& id);
    _run_func rtts_func_bind(_rtts_vfunc_p func);

    //! \brief 线程推理关键函数 待推理数据对象 空闲context索引
    void exec_kernel(Qnode& qn, usize& id) override;

    //! \brief 异步启动 监控所有ContextStream状态
    void async_monitor();

    //! runtime engine对象
    basemod *nvbuilder;
    //! context序列容器 
    ContextStream cs_vec[CONTEXTS];
};

#endif
