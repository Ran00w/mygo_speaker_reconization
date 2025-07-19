import os
from flask import Flask, request, jsonify, send_from_directory
from moviepy import VideoFileClip
from werkzeug.utils import secure_filename
import imageio_ffmpeg
import torch
import scipy.io.wavfile as wav
from utils import split_audio, get_mfcc, is_silent
from model import Classifier
from flask import send_from_directory
import subprocess
import imageio_ffmpeg


UPLOAD_FOLDER = 'uploads'
SEGMENT_FOLDER = 'segments'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'avi', 'mkv'}
MODEL_PATH = 'model.ckpt'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEGMENT_FOLDER'] = SEGMENT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SEGMENT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_audio(audio_path, base_name):
    # 清空分割目录
    for filename in os.listdir(SEGMENT_FOLDER):
        os.remove(os.path.join(SEGMENT_FOLDER, filename))
    silent_list = []
    # 分割音频
    silent_list = split_audio(os.path.dirname(audio_path), os.path.basename(audio_path), SEGMENT_FOLDER, segment_length=1000)
    # 预测
    model = Classifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
#    segment_files = sorted([f for f in os.listdir(SEGMENT_FOLDER) if f.startswith(base_name) and f.endswith('.wav')])
    len_result = len(os.listdir(SEGMENT_FOLDER))-1
    result = [3,3,3]
    for i in range(len_result-3):
        now_tag = 0
        filename = base_name + f"_part{i+1}.wav"
        if (i+1) in silent_list:
#            print(f"Skipping silent segment: {filename}")
            now_tag = 5
            result.append(5)
        else:
            fs, signal = wav.read(os.path.join(SEGMENT_FOLDER, filename))
            feature = get_mfcc(signal, fs)
            feature = torch.FloatTensor(feature).unsqueeze(0)
            with torch.no_grad():
                output = model(feature)
#                print(f"Processing {filename}, output: {output}")
            result.append(torch.argmax(output, dim=1).item())
            now_tag = torch.argmax(output, dim=1).item()
        print(f"window: {i}, Prediction: {now_tag}")
    # 后处理（平滑）
    window_length = 10
    len_result = len(result)
    start = 0
    end = 0
    while start < len(result):
        while end < (len(result)-1) and result[end+1] == result[start]:
            end += 1
        if (end - start) < 4:
            for i in range(start, end+1):
                result[i] = result[start-1] if start > 0 else result[i]
        start = end+1
    ans = []
    for i in range(len_result):
        left = max(0, i-window_length)
        ans.append(result[i])
    return ans

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        ext = filename.rsplit('.', 1)[1].lower()
        base_name = os.path.basename(file_path).rsplit('.', 1)[0]
        audio_path = os.path.join(UPLOAD_FOLDER, base_name + '.wav')
        video_path = os.path.join(UPLOAD_FOLDER, base_name + '.mp4')
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        if ext in ['mp3']:
            cmd = [
                ffmpeg_path,
                '-i', file_path,
                '-vn',
                '-ar', '44100',
                '-ac', '2',
                '-acodec', 'pcm_s16le',
                '-y',
                audio_path
            ]
            subprocess.run(cmd, check=True)
            play_path = video_path  # 默认关联视频文件
        elif ext in ['mp4', 'avi', 'mkv']:
            mp3_path = os.path.join(UPLOAD_FOLDER, base_name + '.mp3')
            cmd_mp3 = [
                ffmpeg_path,
                '-i', file_path,
                '-vn',
                '-ar', '44100',
                '-ac', '2',
                '-ab', '192k',
                '-f', 'mp3',
                '-y',
                mp3_path
            ]
            subprocess.run(cmd_mp3, check=True)
            cmd_wav = [
                ffmpeg_path,
                '-i', mp3_path,
                '-vn',
                '-ar', '44100',
                '-ac', '2',
                '-acodec', 'pcm_s16le',
                '-y',
                audio_path
            ]
            subprocess.run(cmd_wav, check=True)
            play_path = file_path  # 视频文件路径
        else:
            audio_path = file_path
            play_path = video_path  # 默认关联视频文件

        audio_path = audio_path.replace('\\', '/')
        play_path = play_path.replace('\\', '/')

        base_name = os.path.basename(audio_path).rsplit('.', 1)[0]
        result = predict_audio(audio_path, base_name)
        return jsonify({'audio_path': audio_path, 'video_path': play_path, 'result': result})
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/play')
def play():
    file_path = request.args.get('file_path')
    if not file_path:
        return 'No file_path specified', 400
    print(f"Requested file_path: {file_path}")
    filename = os.path.basename(file_path)
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/<path:filename>')
def static_files(filename):
    # 只允许访问静态资源
    if filename.endswith('.css') or filename.endswith('.png'):
        return send_from_directory('.', filename)
    # 其他文件交给 index.html
    return send_from_directory('.', 'index.html')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)