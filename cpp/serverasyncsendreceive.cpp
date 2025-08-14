#include "serverpacked.h"
//
//! QServer对象
//! \brief 接收结构体 用于缓冲
SQMat sqmat_recv;
//! \brief 发送结构体 用于缓冲
SQMat sqmat_send;
//! \brief 通过标志位表明接收的是枚举还是拆分数据
SQMatStatus sqmat_enum_recv=SQMatStatus::TOTALL_LEN;
bool isrecv_enum=true;

bool server_recv_single(int& _recvsocket, char *buffer, size_t allsize)
{
    //已接收的数据字节大小
    size_t sendsize=0;

    while (sendsize < allsize)
    {
        ssize_t n=::recv(
            _recvsocket,
            buffer + sendsize,
            allsize - sendsize,
            0
        );

        //对于已经发送的数据必须保证传输正常
        if (n <= 0) {
            if (errno == ACCEPTNONE)
                break;
            else
                return false;
        }

        sendsize+=n;
    }
    
    return true;
}

bool QServer::server_socket_recv()
{
    //始终接收列入队列 cmd命令线程通过_isrun指示是否开始
    //接收拆分数据的枚举标识
    if (isrecv_enum) {
        //当前接收枚举状态
        int em;
        char *em_p=reinterpret_cast<char*>(&em);
        _usize allsize=sizeof(em);
        _size recvsize=0;
        while (recvsize < allsize) {
            _size n=::recv(
                this->_ser_lissocket_com,
                em_p + recvsize,
                allsize - recvsize,
                0
            );
            
            //对于已经发送的数据必须保证传输正常
            if (n <= 0) {
                if (errno == ACCEPTNONE)
                    break; //意外情况 已经无接收数据
                else {
                    std::cerr<<"server listen接收期间socket断开或其他错误 ! 终止服务端";
                    exit(EXIT_FAILURE);
                }
            }

            recvsize+=n;
        }

        //解析接收的枚举值
        sqmat_enum_recv=static_cast<SQMatStatus>(em);
        //标识下一次接收实际数据
        isrecv_enum=false;
        return true;
    }

    //下一次接收实际拆分数据
    char *buffer=nullptr;
    _usize allsize=0;
    //根据缓存的枚举匹配拆分数据
    switch (sqmat_enum_recv)
    {
    case SQMatStatus::TOTALL_LEN:
        //先重置缓冲对象
        sqmat_recv.reset();
        buffer=reinterpret_cast<char*>(&sqmat_recv.totall_len);
        allsize=sizeof(sqmat_recv.totall_len);
        assert(server_recv_single(this->_ser_lissocket_com, buffer, allsize));
        //标识下一次接收枚举变量
        isrecv_enum=true;
        return true;

    case SQMatStatus::MATPATH_LEN:
        buffer=reinterpret_cast<char*>(&sqmat_recv.matpath_len);
        allsize=sizeof(sqmat_recv.matpath_len);
        assert(server_recv_single(this->_ser_lissocket_com, buffer, allsize));
        //标识下一次接收枚举变量
        isrecv_enum=true;
        return true;

    case SQMatStatus::MATNAME_LEN:
        buffer=reinterpret_cast<char*>(&sqmat_recv.matname_len);
        allsize=sizeof(sqmat_recv.matname_len);
        assert(server_recv_single(this->_ser_lissocket_com, buffer, allsize));
        //标识下一次接收枚举变量
        isrecv_enum=true;
        return true;

    case SQMatStatus::MAT_LEN:
        buffer=reinterpret_cast<char*>(&sqmat_recv.mat_len);
        allsize=sizeof(sqmat_recv.mat_len);
        assert(server_recv_single(this->_ser_lissocket_com, buffer, allsize));
        //标识下一次接收枚举变量
        isrecv_enum=true;
        return true;

    case SQMatStatus::MATPATH:
        //只接收有效的路径数据
        //buffer=sqmat_recv.matpath;
        allsize=sqmat_recv.matpath_len;
        assert(server_recv_single(this->_ser_lissocket_com, buffer, allsize));
        //标识下一次接收枚举变量
        isrecv_enum=true;
        return true;
    
    case SQMatStatus::MATNAME:
        //只接收有效的名称数据
        //buffer=sqmat_recv.matname;
        allsize=sqmat_recv.matname_len;
        assert(server_recv_single(this->_ser_lissocket_com, buffer, allsize));
        //标识下一次接收枚举变量
        isrecv_enum=true;
        return true;

    case SQMatStatus::MATDATAS:
        //动态分配大小
        sqmat_recv.matdatas=new char[sqmat_recv.mat_len];
        buffer=sqmat_recv.matdatas;
        allsize=sqmat_recv.mat_len;
        assert(server_recv_single(this->_ser_lissocket_com, buffer, allsize));
        //将拼接完成的缓冲结构体传递给计算进程
        ////////////////////////////////////////////////    
        ////////////////////////////////////////////////
        std::cout<<"成功接收SQMat数据并传递给计算进程 \n";
        std::cout<<"路径 "<<_ctexts(sqmat_recv.matpath)<<'\n';
        std::cout<<"名称 "<<_ctexts(sqmat_recv.matname)<<'\n';

        //对于最后接收数据完成后重置
        sqmat_recv.reset();
        //标识下一次接收枚举变量
        isrecv_enum=true;
        return true;
    
    default:
        //意外分支
        break;
    }

    //意外情况
    return false;
}

bool server_send_single(SQMatStatus sqtus, int& _sendsocket, char *buffer, size_t allsize)
{
    //已发送的总字节数
    size_t sendsize=0;
    
    //先发送拆分数据枚举标识
    int isqtus=static_cast<int>(sqtus);
    char *isqtus_p=reinterpret_cast<char*>(&isqtus);
    while (sendsize < sizeof(isqtus)) {
        ssize_t n=::send(
            _sendsocket,
            isqtus_p + sendsize,
            sizeof(isqtus_p) - sendsize,
            0
        );

        //严格确保数据回传
        if (n <= 0) {
            if (errno == ACCEPTNONE) {
                qDebug()<<"server发送状态指示错误 发送缓冲区满 发送失败 !";        
                return false;
            }
            std::cerr<<"server发送状态指示期间客户端断开连接 发送失败 !\n";
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
 
        if (n <= 0) {
            if (errno == ACCEPTNONE) {
                qDebug()<<"server发送拆分数据错误 发送缓冲区满 发送失败 !";        
                return false;
            }
            std::cerr<<"server发送拆分数据错误 连接状态有误 发送失败 !";
            return false;
        }
        sendsize+=n;
    }
    return true;
}

bool QServer::server_socket_sned()
{
    //现在_issend已经通知存在有效SQMat发送缓冲对象 直接发送准备好的推理结果
    //无需sqmat_send.reset() 计算线程已填充
    //发送总字节长度
    assert(server_send_single(
        SQMatStatus::TOTALL_LEN, this->_ser_lissocket_com,
        reinterpret_cast<char*>(&sqmat_send.totall_len),
        sizeof(sqmat_send.totall_len)
    ));

    //发送路径长度
    assert(server_send_single(
        SQMatStatus::MATPATH_LEN, this->_ser_lissocket_com,
        reinterpret_cast<char*>(&sqmat_send.matpath_len),
        sizeof(sqmat_send.matpath_len)
    ));

    //发送名称长度
    assert(server_send_single(
        SQMatStatus::MATNAME_LEN, this->_ser_lissocket_com,
        reinterpret_cast<char*>(&sqmat_send.matname_len),
        sizeof(sqmat_send.matname_len)
    ));

    //发送图像数据长度
    assert(server_send_single(
        SQMatStatus::MAT_LEN, this->_ser_lissocket_com,
        reinterpret_cast<char*>(&sqmat_send.mat_len),
        sizeof(sqmat_send.mat_len)
    ));

    //发送路径 只发送有效长度
    //assert(server_send_single(
    //    SQMatStatus::MATPATH, this->_ser_lissocket_com,
    //    sqmat_send.matpath,
    //    sqmat_send.matpath_len
    //));

    //发送名称 只发送有效长度
    //assert(server_send_single(
    //    SQMatStatus::MATNAME, this->_ser_lissocket_com,
    //    sqmat_send.matpath,
    //    sqmat_send.matpath_len
    //));

    //发送图像实际数据 动态分配内存
    sqmat_send.matdatas=new char[sqmat_send.mat_len];
    assert(server_send_single(
        SQMatStatus::MATDATAS, this->_ser_lissocket_com,
        sqmat_send.matdatas,
        sqmat_send.mat_len
    ));

    //发送完成后重置
    std::cout<<"成功发送SQMat数据 \n";
    std::cout<<"路径 "<<_ctexts(sqmat_send.matpath)<<"\n";
    std::cout<<"名称 "<<_ctexts(sqmat_send.matname)<<"\n";
    sqmat_send.reset();
    return true;
}
