# tensorrt-lib

import os
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
from calibrator import Calibrator
# from torch.autograd import Variable
# import torch
import numpy as np
import time

# add verbose
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)  # ** engine可视化 **


# create tensorrt-engine
# fixed and dynamic
def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="", \
               fp16_mode=False, int8_mode=False, calibration_stream=None, calibration_table_path="", save_engine=False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_builder_config() as config, \
                builder.create_network(1) as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:
                    
            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
                assert network.num_layers > 0, 'Failed to parse ONNX model. \
                            Please check if the ONNX model is compatible '
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
                    
            # 获取输入和输出名, 输出的batchsize一般直接follow输入的batchsize
            input_name = 'input'

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
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
            config.add_optimization_profile(profile)

            # build trt engine
            # builder.max_batch_size = max_batch_size
            # builder.max_workspace_size = 1 << 30  # 1GB
            # config.max_workspace_size = 1 << 30
            # builder.fp16_mode = fp16_mode
            config.set_flag(trt.BuilderFlag.FP16)
            if int8_mode:
                # builder.int8_mode = int8_mode
                config.set_flag(trt.BuilderFlag.INT8)
                assert calibration_stream, 'Error: a calibration_stream should be provided for int8 mode'
                #  builder.int8_calibrator = Calibrator(calibration_stream, calibration_table_path)
                config.int8_calibrator = Calibrator(calibration_stream, calibration_table_path)
                print('Int8 mode enabled')
            engine = builder.build_engine(network, config)
            if engine is None:
                print('Failed to create the engine')
                return None
            print("Completed creating the engine")
            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine)