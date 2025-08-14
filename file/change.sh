ONNX_MODEL_PATH="./RMBG_model.onnx"
TRT_ENGINE_PATH="./engine.trt"
 
# 设置工作空间 单位为 MB
MEM_POOL_SIZE=6400 
 
MIN_SHAPES="input:1x3x1024x1024"  
OPT_SHAPES="input:1x3x1024x1024"  
MAX_SHAPES="input:1x3x1024x1024"

# 使用 trtexec 转换 ONNX 模型为 TensorRT 引擎
trtexec --onnx=$ONNX_MODEL_PATH \
        --saveEngine=$TRT_ENGINE_PATH \
        --workspace=$MEM_POOL_SIZE \
        --minShapes=$MIN_SHAPES \
        --optShapes=$OPT_SHAPES \
        --maxShapes=$MAX_SHAPES \
	--verbose 
	#--fp16

