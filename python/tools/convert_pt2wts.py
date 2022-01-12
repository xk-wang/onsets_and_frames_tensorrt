import torch
import argparse
import os
import struct
import sys
sys.path.append('./onsets-and-frames/') 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pth', type=str, help='the pth model path')
    parser.add_argument('-w', '--wts', type=str, help='the path to save wts model')
    return parser.parse_args()

def main():
    args=parse_args()
    net = torch.load(args.pth).cuda().eval()
    
    with open(args.wts, 'w') as f:
        f.write("{}\n".format(len(net.state_dict().keys())))
        for k,v in net.state_dict().items():
            vr = v.reshape(-1).cpu().numpy()
            f.write("{} {}".format(k, len(vr)))
            for vv in vr:
                f.write(" ")
                f.write(struct.pack(">f", float(vv)).hex())
            f.write("\n")

if __name__ == '__main__':
    main()