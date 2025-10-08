import os
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
import imageio_ffmpeg
import torch
import scipy.io.wavfile as wav
from utils.audio_utils import split_audio, is_silent
from utils.feature_extraction import get_mfcc
from models.classifier import Classifier
import subprocess

UPLOAD_FOLDER = 'uploads'
SEGMENT_FOLDER = 'segments'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'avi', 'mkv'}
MODEL_PATH = 'checkpoints/model.ckpt'

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEGMENT_FOLDER'] = SEGMENT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SEGMENT_FOLDER, exist_ok=True)

# 确保模型检查点目录存在
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def allowed_file(filename):
    """
    检查文件类型是否允许
    
    Args:
        filename: 文件名
        
    Returns:
        bool: 是否允许
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_audio(audio_path, base_name):
    """
    预测音频中的说话人
    
    Args:
        audio_path: 音频路径
        base_name: 基础名称
        
    Returns:
        list: 预测结果
    """
    # 清空分割目录
    for filename in os.listdir(SEGMENT_FOLDER):
        os.remove(os.path.join(SEGMENT_FOLDER, filename))
    
    silent_list = []
    # 分割音频
    silent_list = split_audio(os.path.dirname(audio_path), os.path.basename(audio_path), SEGMENT_FOLDER, segment_length=1000)
    
    # 预测
    model = Classifier(n_spks=5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    
    # 获取分割后的音频文件数量
    segment_files = [f for f in os.listdir(SEGMENT_FOLDER) if f.startswith(base_name) and f.endswith('.wav')]
    segment_files.sort(key=lambda x: int(x.split('_part')[1].split('.wav')[0]))
    len_result = len(segment_files)
    result = []
    
    for i in range(len_result):
        now_tag = 0
        filename = base_name + f"_part{i+1}.wav"
        if (i+1) in silent_list:
            now_tag = 5  # 静音标记
            result.append(5)
        else:
            fs, signal = wav.read(os.path.join(SEGMENT_FOLDER, filename))
            feature = get_mfcc(signal, fs)
            feature = torch.FloatTensor(feature).unsqueeze(0)
            with torch.no_grad():
                output = model(feature)
            result.append(torch.argmax(output, dim=1).item())
            now_tag = torch.argmax(output, dim=1).item()
        print(f"window: {i}, Prediction: {now_tag}")
    
    # 后处理（平滑）
    window_length = 10
    start = 0
    end = 0
    while start < len(result):
        while end < (len(result)-1) and result[end+1] == result[start]:
            end += 1
        if (end - start) < 4:
            for i in range(start, end+1):
                result[i] = result[start-1] if start > 0 else result[i]
        start = end+1
    
    # 获取原始音频时长，确保预测结果与视频时长匹配
    fs, signal = wav.read(audio_path)
    audio_duration = len(signal) / fs  # 音频时长（秒）
    # 每0.1秒一个预测结果，总结果数 = 音频时长 * 10
    total_frames = int(audio_duration * 10)
    
    # 如果预测结果少于总帧数，用最后一个结果填充
    if len(result) < total_frames:
        last_result = result[-1] if result else 3
        result.extend([last_result] * (total_frames - len(result)))
    # 如果预测结果多于总帧数，截断
    elif len(result) > total_frames:
        result = result[:total_frames]
    
    # 添加0.2秒延迟（2帧）
    delay_frames = 2
    delayed_result = [5] * delay_frames  # 前0.2秒设为静音
    delayed_result.extend(result[:-delay_frames])  # 去掉最后2帧，保持总长度不变
    
    return delayed_result

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    上传文件并进行预测
    """
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
        ffmpeg_path = "ffmpeg"  # 使用系统ffmpeg

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
            # 如果是wav文件，直接使用，不进行转换
            audio_path = file_path
            play_path = video_path  # 默认关联视频文件

        audio_path = audio_path.replace('\\', '/')
        play_path = play_path.replace('\\', '/')

        base_name = os.path.basename(audio_path).rsplit('.', 1)[0]
        result = predict_audio(audio_path, base_name)
        
        return jsonify({
            'audio_path': audio_path, 
            'video_path': play_path, 
            'result': result
        })
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/play')
def play():
    """
    播放文件
    """
    file_path = request.args.get('file_path')
    if not file_path:
        return 'No file_path specified', 400
    
    print(f"Requested file_path: {file_path}")
    filename = os.path.basename(file_path)
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/')
def index():
    """
    主页
    """
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)