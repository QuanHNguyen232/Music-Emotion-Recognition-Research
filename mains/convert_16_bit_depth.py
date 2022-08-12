import os
import subprocess
import sys

def main():
  # ffmpeg -i 1.mp3_vocals.wav -c:a pcm_s16le -ar 44100 1.mp3_vocals_1.wav
  src = sys.argv[1]
  dst = sys.argv[2]

  if not os.path.exists(dst):
    os.mkdir(dst)

  for song_name in os.listdir(src):
    dst_song_path = os.path.join(dst, song_name)
    src_song_path = os.path.join(src, song_name)
    os.mkdir(dst_song_path)

    for part_name in os.listdir(src_song_path):
      dst_part_path = os.path.join(dst_song_path, part_name)
      src_part_path = os.path.join(src_song_path, part_name)

      cmd = f"ffmpeg -i {src_part_path} -c:a pcm_s16le -ar 44100 {dst_part_path}"
      returned_value = subprocess.call(cmd, shell=True)


if __name__ == "__main__":
  main()


"""
python ./mains/convert_16_bit_depth.py ./data/PMEmo/PMEmo2019/PMEmo2019/separation ./data/PMEmo/PMEmo2019/PMEmo2019/separation_16

"""