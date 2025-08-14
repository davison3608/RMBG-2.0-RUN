#include "rtpacketed.h"
//
//! runtime
basemod::basemod()
{
    assert(cudaSetDevice(0) == cudaSuccess);
    this->path="../file/engine.trt";

    using namespace std;
    using namespace nvinfer1;

    ifstream file(this->path, ios::binary);
    if (!file.is_open()) cerr<<"访问trt文件失败 \n";
    
    file.seekg(0, ios::end);
    usize size=file.tellg();
    file.seekg(0, ios::beg);

    _CDatas datas;
    datas.resize(size);

    //反序列化引擎
    cout<<"准备反序列化引擎文件"<<this->path<<" \n";
    run=createInferRuntime(this->log);
    run->setMaxThreads(2);
    
    file.read(datas.data(), size);
    engine=run->deserializeCudaEngine(datas.data(), size);
    assert(engine != nullptr);
    cout<<"已返回序列化trt文件到engine \n";
}

basemod::~basemod()
{
    this->engine->destroy();
    this->run->destroy();
}
//
//! 线程池对象
template<>
basets<THREADS>::_run_func basets<THREADS>::basets_funct_bind(_base_func_p func)
{
    _run_func recv_f=std::bind(func, this);
    return recv_f;
}

template<>
void basets<THREADS>::async_recv()
{
    using namespace std;
    using namespace std::this_thread;
    cout<< "计算线程池async_push线程已启动，开始持续接收队列首数据..."<<endl;
    //当Qdeque对象析构时一同结束
    while (this->que.node_size >= 0) {
    //0.35s 检查
    sleep_for(chrono::milliseconds(350));
    
    //锁住
    _lock lock(this->mt);

    Qnode *node=this->que.front_p();

    //如果队列首空
    if (node == nullptr) {
        //解锁后重新等待
        lock.unlock();
        continue;
    }
    cout<<"发现队首有数据 等待空闲推理节点\n";
    //等待标志 直到有空闲推理节点
    while (1) {
    if (this->isspare.load()) {
        cout<<"存在空闲线程 通知执行\n";
        break;
    }
    this_thread::yield();
    }

    //通知计算线程处理
    lock.unlock();
    this->cv.notify_one();
    continue;
    }
    return ;
}

template<>
basets<THREADS>::basets(usize num):
num(num), cvmat_handle(new cvmat()), cvmask_handle(new cvmask()) 
{
    //线程预分配
    this->atts_vec.resize(THREADS);
    //获取异步队首检测函数
    _run_func recv_f_t=basets_funct_bind(&basets<THREADS>::async_recv);
    //分离一个线程执行
    _thread t(recv_f_t);
    if (t.joinable()) 
    t.detach();
}

template<>
basets<THREADS>::~basets()
{ delete this->cvmat_handle; delete this->cvmask_handle; }
//
//! 
template<>
rtts<THREADS>::_run_func rtts<THREADS>::rtts_func_bind(_rtts_func_p func, Qnode& Node, usize& id)
{
    using namespace std;
    _run_func exec_f=bind(func, this, ref(Node), ref(id));
    return exec_f;
}

template<>
rtts<THREADS>::_run_func rtts<THREADS>::rtts_func_bind(_rtts_vfunc_p func)
{
    using namespace std;
    _run_func monitor_t=bind(func, this);
    return monitor_t;
}

template<>
void rtts<THREADS>::async_monitor()
{
    using namespace std;
    cout<< "计算线程池async_monitor线程已启动，监控所有推理节点状态..."<<endl;
    //当Qdeque对象析构时一同结束
    while (this->que.node_size >= 0) {
    //1.5s检查
    this_thread::sleep_for(chrono::milliseconds(1500));

    //无需锁 独立查看
    bool ss=false;
    for (int i=0; i<CONTEXTS; i++) {
    if (this->cs_vec[i].isfree.load()) {
        ss=true;
        break;
    }
    }
    this->isspare.store(ss);

    //检查完返回
    continue;
    }
    return ;
}

template<>
void rtts<THREADS>::exec_kernel(Qnode& qn, usize& id)
{
    //关键数据对象
    auto src=qn.src.clone();
    _str save="";
    save=qn.save;
    //本次节点
    ContextStream *curr;
    curr=&this->cs_vec[id];
    _Context *con;
    con=curr->get();
    //辅助信息
    usize rows=curr->chw_in[1];
    usize cols=curr->chw_in[2];
    usize can_in=curr->chw_in[0];
    usize can_ou=curr->chw_out[0];
    //cuda
    _custr *cu_str=&curr->stream;
    _cuevt *cu_t_s=&curr->start;
    _cuevt *cu_t_e=&curr->end;

    using namespace std;
    //尺寸变形
    this->cvmat_handle->ResizeMat(src, {can_in, rows, cols}); 
    //展平到容器
    //batchsize = 1
    Datas src_datas;
    this->cvmat_handle->FlattenCHW(src, src_datas); 
    //拷贝显存
    CUDA_CHECK(cudaMemcpyAsync(
        curr->buffers[0],
        src_datas.data(),
        sizeof(precision) * src_datas.size(),
        cudaMemcpyHostToDevice,
        *cu_str
    ));   
    cudaStreamSynchronize(*cu_str);

    //执行
    cudaEventRecord(*cu_t_s);
    cudaEventSynchronize(*cu_t_s);

    assert(con->enqueueV3(*cu_str));
    cudaStreamSynchronize(*cu_str);

    cudaEventRecord(*cu_t_e);
    cudaEventSynchronize(*cu_t_e);

    float time;
    cudaEventElapsedTime(&time, *cu_t_s, *cu_t_e);
    cerr<<save<<"已完成推理 计时 "<<time<<" ms\n";
    //拷贝回数据
    Datas value_datas;
    value_datas.resize(can_ou * rows * cols * curr->batchsize);
        CUDA_CHECK(cudaMemcpyAsync(
        value_datas.data(),
        curr->buffers[1],
        sizeof(precision) * value_datas.size() * curr->batchsize,
        cudaMemcpyDeviceToHost,
        *cu_str
    )); 
    cudaStreamSynchronize(*cu_str);
    
    //重塑
    GrayMat mask;
    this->cvmask_handle->RecoverCHW(value_datas, {rows, cols}, mask);
    //掩码重叠
    CMat value;
    this->cvmask_handle->MaskSegmentation(src, mask, value);
    //保存
    this->cvmat_handle->SaveMat(save.c_str(), value);
    cerr<<"推理叠加图已保存\n";
    return ;
}

template<>
void rtts<THREADS>::Concurrent_exec()
{
    using namespace std;
    cout<<"推理线程启动，等待任务..."<<endl;

    //当Qdeque析构一同结束
    while (this->que.node_size >= 0) {
    //锁住
    _lock lock(this->mt);

    //总是等待被通知
    this->cv.wait(lock);
    cout<<this_thread::get_id()<<"被通知 准备推理\n";

    //队首需要处理 立即寻找索引
    usize id=0;
    for (int i=0; i<CONTEXTS; i++) {
    if (this->cs_vec[i].isfree.load())
        break;
    id++;
    }
    //标记对应推理节点忙
    this->cs_vec[id].isfree.store(false);

    Qnode *first=this->que.front_p();
    //临时对象 只复制关键数据
    Qnode tmp;
    tmp.src=first->src.clone();
    assert(!tmp.src.empty());
    tmp.save=first->save;

    //移除队首 解锁后异步推理
    this->que.front_pop();
    lock.unlock();

    //获取推理执行函数
    std::cout<<"\n"<<tmp.save<<"分配context 开始推理\n";
    _run_func exec=this->rtts_func_bind(&rtts::exec_kernel, tmp, id);
    exec();    
    
    lock.lock();
    //执行完毕 标记推理节点空闲
    this->cs_vec[id].isfree.store(true);
    lock.unlock();
    continue;
    }
    return ;
}

template<>
void rtts<THREADS>::init()
{
    //开启所有推理线程
    for (int i=0; i<THREADS; i++) {
    //获取Concurrent_exec
    auto func=this->rtts_func_bind(&rtts<THREADS>::Concurrent_exec);
    //执行 
    this->atts_vec[i]=_thread(func);
    }
    //分离
    for (auto& e: this->atts_vec) {
    if (e.joinable())
        e.detach();
    }
}

template<>
rtts<THREADS>::rtts(usize numcontexts):
basets(numcontexts), nvbuilder(new basemod())
{
    //手动初始化所有推理节点
    assert(numcontexts == CONTEXTS);
    for (int i=0; i<numcontexts; i++) {
    this->cs_vec[i].init(this->nvbuilder->get());
    std::cout<<"推理节点"<<i<<"已初始化 并持有显存区\n";
    }

    this->isspare.store(false);
    //启用监控线程 查看是否有空闲节点
    _run_func mon_t=this->rtts_func_bind(&rtts<THREADS>::async_monitor);
    _thread t(mon_t);
    if (t.joinable())
    t.detach();
    std::cout<<"计算进程线程池已建立\n";
}

template<>
rtts<THREADS>::~rtts()
{
    for (auto& e: this->cs_vec)
    e.destroy();

    delete this->nvbuilder;
}
