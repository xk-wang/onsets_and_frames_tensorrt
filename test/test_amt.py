import argparse
import os
import sys
sys.path.append('./onsets-and-frames/')
import librosa
import torch.nn.functional as F

import numpy as np
import soundfile
from mir_eval.util import midi_to_hz
import librosa
from onsets_and_frames import *

def load_and_process_audio(flac_path, sequence_length, device):

    random = np.random.RandomState(seed=42)
    audio, sr = librosa.load(flac_path, sr=SAMPLE_RATE, mono=True)
    audio = (audio*32767+0.5).astype('int16')
    assert sr == SAMPLE_RATE

    audio = torch.ShortTensor(audio)

    if sequence_length is not None:
        audio_length = len(audio)
        step_begin = random.randint(audio_length - sequence_length) // HOP_LENGTH
        n_steps = sequence_length // HOP_LENGTH

        begin = step_begin * HOP_LENGTH
        end = begin + sequence_length

        audio = audio[begin:end].to(device)
    else:
        audio = audio.to(device)

    audio = audio.float().div_(32768.0)

    return audio


def transcribe(model, audio, flac_path):
    audio, sr = librosa.load(flac_path, sr=16000, mono=True)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, window='hann', pad_mode='reflect', 
        power=1.0, n_mels=229, fmin=30, fmax=8000, htk=True).T
    mel = np.log( np.clip(mel, 1e-5, mel.max()) )
    length = mel.shape[0]
    segement_length = 512
    batches = int(np.ceil(length / segement_length))
    padding = batches * segement_length - length
    mel = np.pad(mel, ((0, padding), (0, 0)) )

    batch_size = 6
    maxvals = [0, 0, 0, 0, 0]
    summax = [0, 0, 0, 0, 0]
    total_outputs = [[] for i in range(5)]
    for i in range(int(np.ceil(batches/batch_size))):
        left_batches = min(batch_size, batches-i*batch_size)
        data = mel[i*batch_size:i*batch_size+left_batches*segement_length].reshape([left_batches, segement_length, -1])
        outputs = model(torch.from_numpy(data).cuda())
        
        # print(len(outputs))
        for i in range(len(outputs)):
            maxvals[i] = max(maxvals[i], outputs[i].cpu().numpy().max())
            total_outputs[i].append(outputs[i])
            summax[i]+=outputs[i].cpu().numpy().max(axis=-1).sum()

    print(len(total_outputs[0]))
    predictions = dict()
    predictions['onset'] = torch.cat([x.view(-1, 88) for x in total_outputs[0]], dim=0) 
    predictions['frame'] = torch.cat([x.view(-1, 88) for x in total_outputs[3]], dim=0) 
    predictions['velocity'] = torch.cat([x.view(-1, 88) for x in total_outputs[4]], dim=0) 
    return predictions

def transcribe_file(model_file, flac_paths, save_path, sequence_length,
                  onset_threshold, frame_threshold, device):

    model = torch.load(model_file, map_location=device).eval()
    summary(model)

    for flac_path in flac_paths:
        print(f'Processing {flac_path}...', file=sys.stderr)
        audio = load_and_process_audio(flac_path, sequence_length, device)
        predictions = transcribe(model, audio, flac_path)

        p_est, i_est, v_est = extract_notes(predictions['onset'], predictions['frame'], predictions['velocity'], onset_threshold, frame_threshold)
        scaling = HOP_LENGTH / SAMPLE_RATE
        i_est = (i_est * scaling).reshape(-1, 2)

        for i in range(len(p_est)):
            print('%.3f\t%.3f\t%d\t%.3f\n'%(i_est[i][0], i_est[i][1], p_est[i]+MIN_MIDI, v_est[i]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('flac_paths', type=str, nargs='+')
    parser.add_argument('--save-path', type=str, default='.')
    parser.add_argument('--sequence-length', default=None, type=int)
    parser.add_argument('--onset-threshold', default=0.5, type=float)
    parser.add_argument('--frame-threshold', default=0.5, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        transcribe_file(**vars(parser.parse_args()))
