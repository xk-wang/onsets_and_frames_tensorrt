import torch
import os
import sys
sys.path.append('./onsets-and-frames/')
import argparse
import onnxruntime
import numpy as np
import librosa

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('pt_path', type=str)
    parser.add_argument('onnx_path', type=str)
    return parser.parse_args()

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == '__main__':
    args = parse_args()
    model = torch.load(args.pt_path).cuda().eval()

    with torch.no_grad():
        dummy_input = torch.randn(1, 512, 229).cuda()
        output = model(dummy_input)
        input_names = ['melspec']
        output_names = ['onset_pred', 'offset_pred', 'activation_pred', 'frame_pred', 'velocity_pred']
        dynamic_axes= {'melspec': [0], 'onset_pred': [0], 'offset_pred': [0],
                        'activation_pred': [0], 'frame_pred': [0], 'velocity_pred': [0]}

        # default export
        torch.onnx.export(model, dummy_input, args.onnx_path,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=dynamic_axes,
                        opset_version=11)

    # onnx inference
    ort_session = onnxruntime.InferenceSession(args.onnx_path)
    input_name = ort_session.get_inputs()
    with torch.no_grad():
        dummy_input = torch.ones(2, 512, 229).cuda()
        output = model(dummy_input)
    ort_inputs = {'melspec': to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)
    np.testing.assert_allclose(to_numpy(output[0]), ort_outs[0], rtol=1e-3, atol=1e-5)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")