import sys
import os
import argparse
import tensorrt as trt

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument("onnx_path", type=str)
    parser.add_argument("trt_path", type=str)
    args = parser.parse_args()
    onnx_file_path = args.onnx_path
    engine_file_path = args.trt_path
    print('get start')
    TRT_LOGGER = trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config = builder.create_builder_config()
        config.max_workspace_size =( 1 << 30 ) * 2 # 2 GB
        builder.max_batch_size = 16
        config.set_flag(trt.BuilderFlag.FP16)
        # builder.fp16_mode = True
        # Parse model file
        if not os.path.exists(onnx_file_path):
            print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
            exit(0)
        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
        print(f"raw shape of {network.get_input(0).name} is: ", network.get_input(0).shape)
        # network.get_input(0).shape = [-1, 3, 32, -1] #dynamic model example
        for i in range(1):
            profile = builder.create_optimization_profile()
            # min usual max
            profile.set_shape(network.get_input(0).name, (1, 512, 229), (12, 512, 229), (16, 512, 229))
            config.add_optimization_profile(profile)
        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        engine = builder.build_engine(network,config)
        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())