# RMBG-2.0-RUN
TensorRT 高性能图像分割 将输入图像分割为前景与背景 多客户端并发请求处理

多 Context 共享引擎实现高效并发推理  

epoll 连接处理客户端请求  

OpenCV 3.4  

CUDA 加速的前后处理  

![image](show0.png)  

fp16精度测试

![image](test2.jpg)  

![image](out.jpg)  

平均推理延迟情况

![image](show1.png) 
