import librosa
import time

audiopath = '../../sample/MAPS_MUS-chpn-p19_ENSTDkCl.wav'
audio, sr = librosa.load(audiopath, sr=32000, mono=True)
print(sr)

start = time.time()
mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, window='hann', pad_mode='reflect', 
        power=2.0, n_mels=229, fmin=30, fmax=8000, htk=True)
end = time.time()
print('the mel cal time: ', (end-start)*1000, ' ms')
print(mels.shape)
# res = mels.sum(axis=0)
# for i in range(len(res)):
#     print('%.6f'%res[i], end='\t')
#     if i%10==0: print()