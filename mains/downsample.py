import wave
import audioop
import os
import sys

def downsampleWav(src, dst, inrate=44100, outrate=16000, inchannels=2, outchannels=1):
    if not os.path.exists(src):
        print ('Source not found!')
        return False

    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))

    try:
        s_read = wave.open(src, 'r')
        s_write = wave.open(dst, 'w')
    except:
        print ('Failed to open files!')
        return False

    n_frames = s_read.getnframes()
    data = s_read.readframes(n_frames)

    try:
        converted = audioop.ratecv(data, 2, inchannels, inrate, outrate, None)
        if outchannels == 1:
            converted = audioop.tomono(converted[0], 2, 1, 0)
    except:
        print ('Failed to downsample wav')
        return False

    try:
        s_write.setparams((outchannels, 2, outrate, 0, 'NONE', 'Uncompressed'))
        s_write.writeframes(converted)
    except:
        print ('Failed to write wav')
        return False

    try:
        s_read.close()
        s_write.close()
    except:
        print ('Failed to close wav files')
        return False

    return True

if __name__ == "__main__":
  src_folder = sys.argv[1]
  dst_folder = sys.argv[2]
  sample_from = sys.argv[3]
  sample_to = sys.argv[4]
  for fname in os.listdir(src_folder):
    src_path = os.path.join(src_folder, fname)
    dst_path = os.path.join(dst_folder, fname)
    downsampleWav(src_path, dst_path, inrate=sample_from, outrate=sample_to)
  print("Done!")

"""
Usage:
python ./mains/downsample.py "./data/PMEmo/PMEmo2019/PMEmo2019/chorus_wav" "./data/PMEmo/PMEmo2019/PMEmo2019/chorus_wav_4410" 44100 4410
"""