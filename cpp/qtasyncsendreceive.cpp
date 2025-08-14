#include "qtpacked.h"
#include "rtpacketed.h"
#include <fstream>
//
//! SQMat对象
const SQMat *SQMat::init(TypeAlias::_ctexts& path, TypeAlias::_ctexts& name)
{
    //传输图像路径字节尺寸
    this->matpath_len=strlen(path.data());
    this->matpath=path;

    //传输图像名称字节尺寸
    this->matname_len=strlen(name.data());
    this->matname=name;

    std::ifstream file(path.data(), std::ios::binary);
    if (!file.is_open()) 
        return nullptr;
    
    //传输图像实际数据流字节尺寸
    file.seekg(0, file.end);
    this->mat_len=file.tellg();
    file.seekg(0, file.beg);
    //动态分配图像实际数据
    this->matdatas=new char[this->mat_len];
    file.read(this->matdatas, this->mat_len);

    //传输总字节尺寸
    this->totall_len+=sizeof(this->matpath_len); 
    this->totall_len=this->totall_len + this->matpath_len;
    this->totall_len+=sizeof(this->matname_len);  
    this->totall_len=this->totall_len + this->matname_len;
    this->totall_len+=sizeof(this->mat_len);  
    this->totall_len=this->totall_len + this->mat_len;

    this->totall_len+=sizeof(this->totall_len);  

    //测试失效
    this->test_str="";

    return this;
}

void SQMat::deepcopy(SQMat& m)
{
    this->totall_len=m.totall_len;
    this->matpath_len=m.matpath_len;
    this->matname_len=m.matname_len;
    this->mat_len=m.mat_len;
    this->matpath=m.matpath;
    this->matname=m.matname;
}

void SQMat::reset()
{
    this->matpath_len=0;
    this->matpath="";
    
    this->matname_len=0;
    this->matname="";

    this->mat_len=0;
    if (this->matdatas)
        delete[] this->matdatas;
    else
        this->matdatas=nullptr;

    this->totall_len=0;

    this->test_str="test_strings";
}

//
//! talkscreen对象
//! \brief 接收结构体容器
std::vector<SQMat> sqmat_vec(99);
//! \brief 接收索引
size_t sqmat_id=0;

//! \brief 通过标志位表明接收的是枚举还是拆分数据
SQMatStatus sqmat_tus=SQMatStatus::TOTALL_LEN;
bool isenum=true;

bool asyncreceive_single(int& _recvsocket, char *buffer, size_t allsize)
{
    //已接收的数据字节大小
    size_t recvsize=0;

    while (recvsize < allsize)
    {
        ssize_t n=::recv(
            _recvsocket,
            buffer + recvsize,
            allsize - recvsize,
            0
        );

        //对于已经发送的数据必须保证传输正常
        if (n <= 0) {
            if (errno == ACCEPTNONE)
                return false; //说明当前无接收数据 随即continue
            else {
                std::cerr<<"qt端接收推理结果失败 拆分的数据协议接收失败 !";
                exit(EXIT_FAILURE); //不同于发送端 接收时严格处理
            }   
        }

        recvsize+=n;
    }
    
    return true;
}

bool blocksend_single(SQMatStatus sqtus, int& _sendsocket, const char *buffer, size_t allsize)
{
    //已发送的总字节数
    size_t sendsize=0;
    
    //发送枚举指示
    int isqtus=static_cast<int>(sqtus);
    char *isqtus_p=reinterpret_cast<char*>(&isqtus);
    while (sendsize < sizeof(isqtus)) {
        ssize_t n=::send(
            _sendsocket,
            isqtus_p + sendsize,
            sizeof(isqtus_p) - sendsize,
            0
        );

        if (n == 0) {
            qDebug()<<"发送状态指示期间服务器断开连接 发送失败 !";
            return false;
        }
        else if (n < 0) {
            if (errno == ACCEPTNONE) {
                qDebug()<<"发送状态指示错误 发送缓冲区满 发送失败 !";        
                return false;
            }
            qDebug()<<"发送状态指示错误 连接状态有误 发送失败 !";
            return false;
        }
        sendsize+=n;
    }
    
    sendsize=0;
    //发送字节数 到达指定发送字节数
    while (sendsize < allsize) {
        ssize_t n=::send(
            _sendsocket, //指定的已连接socket
            buffer + sendsize, //从上次发送完位置开始
            allsize - sendsize, //剩余发送大小
            0
        );
 
        if (n == 0) {
            qDebug()<<"发送拆分数据期间服务器断开连接 发送失败 !";
            return false;
        }
        else if (n < 0) {
            if (errno == ACCEPTNONE) {
                qDebug()<<"发送拆分数据错误 发送缓冲区满 发送失败 !";        
                return false;
            }
            qDebug()<<"发送拆分数据错误 连接状态有误 发送失败 !";
            return false;
        }
        sendsize+=n;
    }
    return true;
}

void talkscreen::asyncreceive()
{
    //根据已连接状态为循环接收依据
    while (this->iscontected.load()) {
    //优先接收枚举变量
    if (isenum) {
        //当前接收枚举状态
        int e;
        char *e_p=reinterpret_cast<char*>(&e);
        _usize allsize=sizeof(e);
        _size sendsize=0;
        while (sendsize < allsize) {
            _size n=::recv(
                this->cus_socket,
                e_p + sendsize,
                allsize - sendsize,
                0
            );
            
            //对于对象已经发送的数据必须保证传输正常
            if (n <= 0) {
                if (errno == ACCEPTNONE) {
                    std::this_thread::sleep_for(std::chrono::microseconds(100)); //1ms
                    continue; //表明当前无数据接收 重新recv
                }
                else {
                    std::cerr<<"server listen接收期间socket断开或其他错误 ! 终止服务端";
                    exit(EXIT_FAILURE);
                }
            }
            sendsize+=n;
        }

        //解析接收的枚举值
        sqmat_tus=static_cast<SQMatStatus>(e);
        //标识下一次接收实际数据
        isenum=false;
        continue ;
    }
    
    //下一次接收实际拆分数据
    char *buffer=nullptr;
    _usize allsize=0;
    //根据缓存的枚举匹配拆分数据
    switch (sqmat_tus)
    {
    case SQMatStatus::TOTALL_LEN:
        //协议首部 先重置对应的对象
        sqmat_vec[sqmat_id].reset();
        buffer=reinterpret_cast<char*>(&sqmat_vec[sqmat_id].totall_len);
        allsize=sizeof(sqmat_vec[sqmat_id].totall_len);
        if (!asyncreceive_single(this->cus_socket, buffer, allsize)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); //1ms
            continue; //说明枚举接收完后没有实际数据
        }
        else {
            //表示下一次应该接收枚举
            isenum=true;
            continue;
        }

    case SQMatStatus::MATPATH_LEN:
        buffer=reinterpret_cast<char*>(&sqmat_vec[sqmat_id].matpath_len);
        allsize=sizeof(sqmat_vec[sqmat_id].matpath_len);
        if (!asyncreceive_single(this->cus_socket, buffer, allsize)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); //1ms
            continue; //说明枚举接收完后没有实际数据
        }
        else {
            isenum=true;
            continue;
        }

    case SQMatStatus::MATNAME_LEN:
        buffer=reinterpret_cast<char*>(&sqmat_vec[sqmat_id].matname_len);
        allsize=sizeof(sqmat_vec[sqmat_id].matname_len);
        if (!asyncreceive_single(this->cus_socket, buffer, allsize)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); //1ms
            continue; //说明枚举接收完后没有实际数据
        }
        else {
            isenum=true;
            continue;
        }

    case SQMatStatus::MAT_LEN:
        buffer=reinterpret_cast<char*>(&sqmat_vec[sqmat_id].mat_len);
        allsize=sizeof(sqmat_vec[sqmat_id].mat_len);
        if (!asyncreceive_single(this->cus_socket, buffer, allsize)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); //1ms
            continue; //说明枚举接收完后没有实际数据
        }
        else {
            isenum=true;
            continue;
        }

    case SQMatStatus::MATPATH:
        //buffer=sqmat_vec[sqmat_id].matpath.c_str();
        allsize=sqmat_vec[sqmat_id].matpath_len;
        if (!asyncreceive_single(this->cus_socket, buffer, allsize)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); //1ms
            continue; //说明枚举接收完后没有实际数据
        }
        else {
            isenum=true;
            continue;
        }

    case SQMatStatus::MATNAME:
        //buffer=sqmat_vec[sqmat_id].matname;
        allsize=sqmat_vec[sqmat_id].matname_len;
        if (!asyncreceive_single(this->cus_socket, buffer, allsize)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); //1ms
            continue; //说明枚举接收完后没有实际数据
        }
        else {
            isenum=true;
            continue;
        }
    
    case SQMatStatus::MATDATAS:
        //动态分配
        sqmat_vec[sqmat_id].matdatas=new char[sqmat_vec[sqmat_id].mat_len];
        buffer=sqmat_vec[sqmat_id].matdatas;
        allsize=sqmat_vec[sqmat_id].mat_len;
        if (!asyncreceive_single(this->cus_socket, buffer, allsize)) {
            delete[] sqmat_vec[sqmat_id].matdatas;
            sqmat_vec[sqmat_id].matdatas=nullptr;
            //无数据接收释放内存
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); //1ms
            continue;
        }
        else {
            std::cout<<"成功接收SQMat数据并传递给计算进程 \n";
            std::cout<<"路径 "<<_ctexts(sqmat_vec[sqmat_id].matpath)<<'\n';
            std::cout<<"名称 "<<_ctexts(sqmat_vec[sqmat_id].matname)<<'\n';
            //协议尾部则索引自增
            sqmat_id++;
            isenum=true;
            continue;
        }

    default:
        //意外分支
        goto label_cus_recv_down;
    }
    //while single down
    }
    label_cus_recv_down:

    qDebug()<<"连接断开 已退出异步接收数据线程";
    return ;
}

bool talkscreen::blocksend(const SQMat *&_sqmat, int& _sendsocket)
{
    bool status;

    //发送总长度
    TypeAlias::_usize total_head_s=sizeof(_sqmat->totall_len);
    const char *total_head=(const char*)&_sqmat->totall_len;
    status=blocksend_single(SQMatStatus::TOTALL_LEN, this->cus_socket, total_head, total_head_s);
    if (!status)
        return status;

    //发送路径长度 
    TypeAlias::_usize path_head_s=sizeof(_sqmat->matpath_len);
    const char *path_head=(const char*)&_sqmat->matpath_len;
    status=blocksend_single(SQMatStatus::MATPATH_LEN, this->cus_socket, path_head, path_head_s);
    if (!status)
        return status;

    //名称长度
    TypeAlias::_usize name_head_s=sizeof(_sqmat->matname_len);
    const char *name_head=(const char*)&_sqmat->matname_len;
    status=blocksend_single(SQMatStatus::MATNAME_LEN, this->cus_socket, name_head, name_head_s);
    if (!status)
        return status;

    //实际图像数据长度
    TypeAlias::_usize mat_head_s=sizeof(_sqmat->mat_len);
    const char *mat_head=(const char*)&_sqmat->mat_len;
    status=blocksend_single(SQMatStatus::MAT_LEN, this->cus_socket, mat_head, mat_head_s);
    if (!status)
        return status;

    //发送路径内容 只发送有效长度
    TypeAlias::_usize path_s=_sqmat->matpath_len;
    //const char *path=_sqmat->matpath;
    //status=blocksend_single(SQMatStatus::MATPATH, this->cus_socket, path, path_s);
    if (!status)
        return status;

    //发送名称内容 只发送有效长度
    TypeAlias::_usize name_s=_sqmat->matname_len; 
    //const char *name=_sqmat->matname;
    //status=blocksend_single(SQMatStatus::MATNAME, this->cus_socket, name, name_s);
    if (!status)
        return status;

    //发送实际图像数据    
    TypeAlias::_usize mat_s=_sqmat->mat_len;
    const char *mat_data=(const char*)&_sqmat->matdatas;
    status=blocksend_single(SQMatStatus::MATDATAS, this->cus_socket, mat_data, mat_s);
    if (!status)
        return status;

    return true;
}

void talkscreen::on_sendbutton_clicked()
{
    qDebug()<<"\n\n触发事件 发送数据\n\n";

    //读取lineedit
    _qtexts qmatpath=this->talkformui->picturepathlineedit->text();
    _qtexts qmatname=this->talkformui->namelineedit->text();
    _ctexts matpath=qmatpath.toStdString();
    _ctexts matname=qmatname.toStdString();

    //声明临时结构体
    SQMat cus_socket_struct;
    cus_socket_struct.reset();

    //init结构体或是测试
    const SQMat *there;
    if (!qmatpath.isEmpty() && !qmatname.isEmpty())
        there=cus_socket_struct.init(matpath, matname);
    else {
        qDebug()<<"未输入路径或是名称 已返回";
        return ;
    }
    //发送数据 阻塞式发送 等待这次传输完成
    if (there) {
        //bool status;
        //status=this->blocksend(there, this->cus_socket);
        //if (!status) 
        //    return ;   

        //传递到计算推理队列
        this->inference->que.push(cus_socket_struct);

        std::cout<<"成功发送SQMat数据 \n";
        std::cout<<"路径 "<<_ctexts(there->matpath)<<"\n";
        std::cout<<"名称 "<<_ctexts(there->matname)<<"\n";
        cus_socket_struct.reset();
    }
    else {
        qDebug()<<"打开文件失败 不是有效文件路径";
        qDebug()<<"已返回";
    }

    return ;
}

void talkscreen::on_downbutton_clicked()
{
    ////////////////////////////////////////////////    
    ////////////////////////////////////////////////
}

