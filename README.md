# Onsets and Frames TensorRT inference

This repository is a tensorrt deployment of the onsets and frames model, which is implemented using pytorch (https://github.com/jongwook/onsets-and-frames).

## Requirements
- c++ dependencies
  1. TensorRT-8.0.1.6
  2. OpenCV-4.2.0
  3. CUDA Toolkit-10.2
  4. cuDNN-8.2.2.26
  5. Eigen-3.3.7
  6. libsamplerate
  7. protobuf-3.11.4
  8. cnpy
  9. midifile
- python dependencies
  1. tensorrt-8.0.1.6 (c++ TensorRT/python)
  2. toch-1.7.1
  3. opencv-python-4.2
  4. onnx-1.8.0
  4. onnxruntime-1.8.0
  5. cudatoolkit-10.2
  6. torchvision-0.8.2
  7. torchaudio-0.7.2
  8. torch2trt (github)
  9. onnx-simplifier (github)
  10. librosa-0.8.1
  11. tensorboard-2.7.0
  12. mido
  13. mir_eval
  13. tqdm

detailed information can be found in torch1.7.yaml

## Train yourself
see the [READE](python/onsets-and-frames/README.md) file under python/onsets-and-frames/

## Get the TensorRT Engine file
1.train the model yourself or use the pretrained model in Baidu Netdisk https://pan.baidu.com/s/1SiW8A6DHa9du9RQyzouZSQ?pwd=8ir9 Extract code: 8ir9. Put the model file under models.

2.convert the pt file to onnx
```bash
cd python
python tools/convert_pt2onnx.py ../models/model-500000.pt ../models/model.onnx
```
3.convert the onnx to TensorRT engine.
```bash
python tools/convert_onnx2trt.py ../models/model.onnx ../models/model.trt
```

## Make
1.configure cpp/CMakeLists.txt

2.make 
```bash
cd cpp && mkdir build && cd build
cmake ..
make
```
3.inference
```bash
./amt ../../models/model.trt ../../sample/MAPS_MUS-chpn-p19_ENSTDkCl.wav
```
