import os
import numpy as np
from pydub import AudioSegment
from utils import is_silent

def split_wav_files(input_dir, output_dir, segment_length=1000, window_length=10):
    """
    将input_dir目录下所有wav文件按segment_length毫秒切割, 并进行静音检测和归一化，保存到output_dir
    :param input_dir: 输入wav文件目录
    :param output_dir: 输出切割后wav文件目录
    :param segment_length: 每段音频长度（毫秒）
    :param window_length: 滑动窗口的分割比例
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, filename))  # 清空输出目录
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.wav'):
            audio = AudioSegment.from_wav(os.path.join(input_dir, filename))
            duration = len(audio)
            base_name = os.path.splitext(filename)[0]
            i = 0
            start = 0
            while start < duration:
                start = i * segment_length / window_length
                end = min(start + segment_length, duration)
                segment = audio[start:end]

                # 转换为 numpy 数组以进行静音检测和归一化
                samples = np.array(segment.get_array_of_samples())
                if is_silent(samples):
                    print(f"跳过静音片段: {base_name}_part{i+1}.wav")
                    i += 1
                    continue

                # 归一化音频
                samples = samples / np.max(np.abs(samples))*100
                segment = segment._spawn(samples.astype(np.int16).tobytes())

                out_name = f"{base_name}_part{i+1}.wav"
                segment.export(os.path.join(output_dir, out_name), format="wav")
                i += 1
            print(f"{filename} 切割完成，共{(duration + segment_length - 1) // segment_length}段")

if __name__ == "__main__":
#    split_wav_files("raw_data/Tomori", "data/Tomori", segment_length=1000, window_length = 1)
#    split_wav_files("raw_data/Anon", "data/Anon", segment_length=1000, window_length = 1)
#    split_wav_files("raw_data/Taki", "data/Taki", segment_length=1000, window_length = 1)
#    split_wav_files("raw_data/Soyo", "data/Soyo", segment_length=1000, window_length = 1)
    split_wav_files("raw_data/Rana", "data/Rana", segment_length=1000, window_length = 1)
