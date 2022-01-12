import numpy as np
import os
import torch
import sys
sys.path.append('../onsets-and-frames/')
from onsets_and_frames import *

path = '../../cpp/build/cpp_prob.npy'
data = np.load(path)

p_est, i_est, v_est = extract_notes(torch.from_numpy(data[0]), torch.from_numpy(data[3]), torch.from_numpy(data[4]), 0.5, 0.5)

scaling = HOP_LENGTH / SAMPLE_RATE

i_est = (i_est * scaling).reshape(-1, 2)

for i in range(len(p_est)):
    print('%.3f\t%.3f\t%d\t%.3f\n'%(i_est[i][0], i_est[i][1], p_est[i]+MIN_MIDI, v_est[i]))