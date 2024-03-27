import tensorrt as trt
import numpy as np
import cv2

import pycuda.driver as cuda
import pycuda.autoinit

import time

v_onnx_path = "/home/clip2onnx/deploy/v.onnx"
t_onnx_path = "/home/clip2onnx/deploy/t.onnx"

# 创建Logger和Builder
trt_logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(trt_logger)

# 创建网络
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# 解析ONNX模型
with open(v_onnx_path, 'rb') as model_file:
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse(model_file.read()):
        print ('Failed to parse the ONNX file')
        for error in range(parser.num_errors):
            print (parser.get_error(error))
            
# 获取输入和输出名
input_name = 'input'
output_name = 'output'

# 获取输入层
input_layer = network.get_input(0)

# 创建优化配置文件
profile = builder.create_optimization_profile()

# 设置动态轴的最小、最优和最大尺寸
batch_size_min = 1  # 可根据实际情况调整
batch_size_opt = 1  # 可根据实际情况调整
batch_size_max = 32  # 可根据实际情况调整

profile.set_shape(input_name, (batch_size_min,) + input_layer.shape[1:], (batch_size_opt,) + input_layer.shape[1:], (batch_size_max,) + input_layer.shape[1:])
        
# 创建Builder配置并设置最大工作区大小
builder_config = builder.create_builder_config()
builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
builder_config.add_optimization_profile(profile)

# 使用配置构建引擎
serialized_engine = builder.build_serialized_network(network, builder_config)

# 保存
# with open("v_clip.engine", "wb") as f:
#     f.write(serialized_engine)

# 读取
# with open("your_engine.engine", "rb") as f:
#     serialized_engine = f.read()

runtime = trt.Runtime(trt_logger)
engine = runtime.deserialize_cuda_engine(serialized_engine)
context = engine.create_execution_context()
context.set_input_shape(input_name, [1, 3, 224, 224])

# 预处理时间 100轮测试， 平均 0.01591s fp32
start_time_1 = time.time()
for i in range(0,100):
    # 读取图片
    img_path = "/home/CLIP.png"
    bgr_img = cv2.imread(img_path)

    # 调整图像尺寸为224x224
    resized_img = cv2.resize(bgr_img, (224, 224),interpolation=cv2.INTER_LINEAR)

    # 将BGR图像转换为RGB格式
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    # 归一化到0-1范围
    normalized_img = rgb_img.astype(np.float32) / 255.0

    # 减去均值除以方差（假设mean和stddev是预先计算好的）
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    stddev = np.array([0.26862954, 0.26130258, 0.27577711])

    # 减去均值除以方差
    normalized_img = (normalized_img - mean)/ stddev

    # 转换为CHW布局
    chw_img = normalized_img.transpose((2, 0, 1))

    nchw_image = chw_img[np.newaxis, :, :, :]

end_time_1 = time.time()
execution_time_1 = end_time_1 - start_time_1
print(f"Execution time for Process: {execution_time_1} seconds")

# input = np.random.uniform(low=0.0, high=1.0, size=[1, 3, 224, 224]).astype(np.float32)
input = nchw_image.astype(np.float32)
output = np.empty([1, 512], dtype = np.float32) 

# allocate device memory
d_input = cuda.mem_alloc(input.nbytes)
d_output = cuda.mem_alloc(output.nbytes)
bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()

# 推理时间100轮计时 0.00278194s fp32
start_time_2 = time.time()
for i in range(0,100):
    # 确保图像数据是连续存储的
    input_contiguous = input.copy(order='C')
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, input_contiguous, stream)# execute model
    context.execute_async_v2(bindings, stream.handle, None)# transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)# syncronize threads
    stream.synchronize()

d_input.free()
d_output.free()

end_time_2 = time.time()
execution_time_2 = end_time_2 - start_time_2
print(f"Execution time for infer: {execution_time_2} seconds")

print(output)

# 与pytorch对比
# import torch
# import clip
# model, _ = clip.load("/home/ViT-B-16.pt", "cpu")
# default_visual_output = model.visual(torch.from_numpy(input))
# print("Visual %s ", {np.abs(default_visual_output.detach().numpy() - output).max()})
# print(np.abs(default_visual_output.detach().numpy() - output))