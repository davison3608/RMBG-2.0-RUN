#include "rtpacketed.h"
//
//! ProcessShare对象
TypeAlias::_status ProcessShare::pop(SQMat& _pop) {
    sembuf sb;
    sb.sem_num = 0;
    sb.sem_flg = 0; 
    sb.sem_op = -1;  // 等待"有数据"通知（来自push的唤醒）

    // 若缓存无效（无数据），必须等待push写入并发送通知（信号量>0）
    if (!this->iscache.load()) {
        // 此处阻塞，直到push执行sem_op=1（信号量从1→0）
        if (semop(this->sem_cache, &sb, 1) != 0) {
            std::cerr << "pop等待push唤醒失败\n";
            return -1;
        }
    }
    
    // 加锁读取数据（此时iscache必为true）
    pthread_mutex_lock(&this->send_mt);
    _pop.reset();
    _pop.deepcopy(this->cache);
    this->cache.reset();
    this->iscache.store(false);  // 标记无数据
    pthread_mutex_unlock(&this->send_mt);

    // 通知push：可写入新数据（信号量从0→1）
    sb.sem_op = 1;
    if (semop(this->sem_cache, &sb, 1) != 0) {
        std::cerr << "pop通知push失败\n";
        return -1;
    }
    std::cout << "pop成功：" << _pop.matpath << "，已通知push\n";
    return 1;
}
//
//! Qnode对象
void Qnode::init() noexcept
{
    using namespace cv;

    //加载mat
    spath path=this->val.matpath;
    this->src=imread(path);
    assert(!this->src.empty());

    cvtColor(this->src, this->src, COLOR_BGR2RGB);
    this->src.convertTo(this->src, MATTYPE);

    //设置保存路径
    this->save=this->val.matname;
}

//
//! Qdeque对象
Qdeque::Qdeque():
node_size(0), first(new Qnode())
{ 
    this->last=this->first;
    this->first->next=nullptr; 
    std::cout<<"计算线程池推理队列已就绪\n";

    //this->_share=ProcessShare::get(); 
    //分离一个线程执行
    //_thread t(&Qdeque::async_push, this);
    //if (t.joinable())
    //t.detach();
}

Qdeque::~Qdeque()
{
    //锁住释放所有节点
    _lock lock(this->que_mt);
    for (int i=0; i<this->node_size; i++) {
        auto old_node=this->first->next;
        auto old_node_next=old_node->next;

        //删除首节点
        this->first->next=old_node_next;
        delete old_node;
    }
    lock.unlock();

    //强制修改
    this->node_size=-99;
    //引用次数递减
    this->_share->destroy();
}

void Qdeque::async_push()
{
    using namespace std;
    cout<< "async_push线程已启动，开始等待共享内存数据..."<<endl;

    while (this->node_size >= 0) {    
    using namespace std::this_thread;
    //锁住
    _lock lock(this->que_mt);

    //查看是否取出
    while (this->_share->iscache.load()) {
    //从尾列入新节点
    auto new_node=new Qnode();
    auto old_node=this->last;
    old_node->next=new_node;
    this->last=new_node;
    this->node_size++;

    //新节点的val复制
    this->_share->pop(this->last->val);
    //复制后节点初始化
    this->last->init();

    cout<<this->last->val.matpath<<"已列入待推理队列\n";
    lock.unlock();
    continue;
    }
    
    //无缓存取出
    lock.unlock();
    continue;
    }
}

Qnode *Qdeque::front_p()
{ 
    //自动释放
    std::lock_guard<_mtx> lock(this->que_mt);
    if (this->node_size == 0) 
    return nullptr;
    return this->first->next; 
}

void Qdeque::push(SQMat& val)
{
    Qnode *new_node=new Qnode();
    //深拷贝节点值
    new_node->val.deepcopy(val);
    //根据节点值手动初始化节点
    new_node->init();
    new_node->next=nullptr;     // 新节点为尾节点，next置空

    // 2. 加锁保证线程安全（与front_pop使用相同的锁）
    _lock lock(this->que_mt);

    // 3. 将新节点添加到队列尾部
    this->last->next = new_node;  // 原尾节点的next指向新节点
    this->last = new_node;        // 更新尾指针为新节点
    this->node_size++;            // 节点计数+1
    lock.unlock();
    std::cout<<val.matpath<<"已列入待推理队列\n";
    return ;
}

void Qdeque::front_pop()
{
    //锁住
    _lock lock(this->que_mt);

    if (this->node_size == 0) 
    { lock.unlock(); return ; }

    auto old_node=this->first->next;
    auto old_node_next=old_node->next;

    //删除首节点
    this->first->next=old_node_next;
    delete old_node;
    this->node_size--;

    if (this->node_size == 0) 
    { this->last=this->first; }

    std::cout<<"已删除队列首节点数据 \n";
    lock.unlock();
    return ;
}
//
//! ContextStream对象
RTTypeAlias::_Context *&ContextStream::get()
{ return this->context; }

void ContextStream::init(_Engine *&eng)
{
    assert(eng);
    assert(eng->getNbIOTensors() == 2);
    using namespace nvinfer1;

    Dims ind=eng->getTensorShape(this->in.c_str());
    Dims outd=eng->getTensorShape(this->out.c_str());
    assert(ind.nbDims > 0);
    assert(outd.nbDims > 0);

    //cuda初始化
    cudaStreamCreateWithFlags(&this->stream, cudaStreamNonBlocking);
    cudaEventCreate(&this->start);
    cudaEventCreate(&this->end);

    //接收context指针与开辟显存
    this->context=eng->createExecutionContext();
    assert(this->context);
    assert(this->context->setInputShape(
        this->in.data(),
        Dims4{batchsize, chw_in[0], chw_in[1], chw_in[2]}
    ));

    _usize Tensor_elements_in=this->batchsize * chw_in[0] * chw_in[1] * chw_in[2];
    _usize Tensor_elements_ou=this->batchsize * chw_out[0] * chw_out[1] * chw_out[2];

    CUDA_CHECK(cudaMallocAsync(
        &this->buffers[0],
        sizeof(_precision) * Tensor_elements_in,
        this->stream
    ));

    CUDA_CHECK(cudaMemsetAsync(
        this->buffers[0],
        0,
        sizeof(_precision) * Tensor_elements_in,
        this->stream
    ));

    CUDA_CHECK(cudaMallocAsync(
        &this->buffers[1],
        sizeof(_precision) * Tensor_elements_ou,
        this->stream
    ));

    CUDA_CHECK(cudaMemsetAsync(
        this->buffers[1],
        0,
        sizeof(_precision) * Tensor_elements_ou,
        this->stream
    ));

    //绑定context tensor
    assert(this->context->setTensorAddress(
        this->in.c_str(),
        this->buffers[0]
    ));
    
    assert(this->context->setTensorAddress(
        this->out.c_str(),
        this->buffers[1]
    ));

    //标志状态
    this->isinit.store(true);
    this->isfree.store(true);    
}

void ContextStream::destroy()
{
    //显存释放
    CUDA_CHECK(cudaFreeAsync(this->buffers[0], this->stream));
    CUDA_CHECK(cudaFreeAsync(this->buffers[1], this->stream));
    cudaStreamDestroy(this->stream);
    //cuda释放
    cudaStreamDestroy(this->stream);
    cudaEventDestroy(this->start);
    cudaEventDestroy(this->end);

    this->context->destroy();
}

