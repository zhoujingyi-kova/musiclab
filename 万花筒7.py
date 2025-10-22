import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft
import matplotlib.colors as mcolors
import threading
import queue
import time
import sys
import math
from matplotlib.patches import Circle
import random
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')

class AudioControlPanel:
    def __init__(self, vortex_app):
        self.vortex_app = vortex_app
        self.root = tk.Tk()
        self.root.title("音频控制面板")
        self.root.geometry("400x600")
        self.root.configure(bg='#2b2b2b')
        
        # 使窗口置顶
        self.root.attributes('-topmost', True)
        
        self.setup_ui()
        
    def setup_ui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = tk.Label(main_frame, text="粒子同心圆音频万花筒", 
                              font=("Arial", 16, "bold"), fg="white", bg="#2b2b2b")
        title_label.pack(pady=10)
        
        # 音频源选择
        source_frame = ttk.LabelFrame(main_frame, text="音频源选择", padding="10")
        source_frame.pack(fill=tk.X, pady=5)
        
        self.source_var = tk.StringVar(value="microphone")
        tk.Radiobutton(source_frame, text="麦克风输入", variable=self.source_var, 
                      value="microphone", command=self.on_source_change,
                      fg="white", bg="#2b2b2b", selectcolor="#2b2b2b").pack(anchor=tk.W)
        tk.Radiobutton(source_frame, text="系统音频", variable=self.source_var, 
                      value="system", command=self.on_source_change,
                      fg="white", bg="#2b2b2b", selectcolor="#2b2b2b").pack(anchor=tk.W)
        tk.Radiobutton(source_frame, text="模拟音频", variable=self.source_var, 
                      value="simulated", command=self.on_source_change,
                      fg="white", bg="#2b2b2b", selectcolor="#2b2b2b").pack(anchor=tk.W)
        
        # 设备选择
        device_frame = ttk.LabelFrame(main_frame, text="音频设备", padding="10")
        device_frame.pack(fill=tk.X, pady=5)
        
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(device_frame, textvariable=self.device_var, state="readonly")
        self.device_combo.pack(fill=tk.X)
        self.device_combo.bind('<<ComboboxSelected>>', self.on_device_change)
        
        # 刷新设备按钮
        ttk.Button(device_frame, text="刷新设备列表", command=self.refresh_devices).pack(pady=5)
        
        # 音频参数控制
        param_frame = ttk.LabelFrame(main_frame, text="音频参数", padding="10")
        param_frame.pack(fill=tk.X, pady=5)
        
        # 音量阈值
        tk.Label(param_frame, text="音量低阈值:", fg="white", bg="#2b2b2b").pack(anchor=tk.W)
        self.low_thresh_var = tk.DoubleVar(value=0.1)
        low_scale = tk.Scale(param_frame, from_=0.01, to=0.5, resolution=0.01,
                           variable=self.low_thresh_var, orient=tk.HORIZONTAL,
                           command=self.on_threshold_change, length=200)
        low_scale.pack(fill=tk.X)
        
        tk.Label(param_frame, text="音量高阈值:", fg="white", bg="#2b2b2b").pack(anchor=tk.W)
        self.high_thresh_var = tk.DoubleVar(value=0.5)
        high_scale = tk.Scale(param_frame, from_=0.1, to=1.0, resolution=0.01,
                            variable=self.high_thresh_var, orient=tk.HORIZONTAL,
                            command=self.on_threshold_change, length=200)
        high_scale.pack(fill=tk.X)
        
        # 饱和度控制
        sat_frame = ttk.LabelFrame(main_frame, text="饱和度控制", padding="10")
        sat_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(sat_frame, text="最小饱和度:", fg="white", bg="#2b2b2b").pack(anchor=tk.W)
        self.min_sat_var = tk.DoubleVar(value=0.2)
        min_sat_scale = tk.Scale(sat_frame, from_=0.0, to=1.0, resolution=0.05,
                               variable=self.min_sat_var, orient=tk.HORIZONTAL,
                               command=self.on_saturation_change, length=200)
        min_sat_scale.pack(fill=tk.X)
        
        tk.Label(sat_frame, text="最大饱和度:", fg="white", bg="#2b2b2b").pack(anchor=tk.W)
        self.max_sat_var = tk.DoubleVar(value=1.0)
        max_sat_scale = tk.Scale(sat_frame, from_=0.0, to=1.0, resolution=0.05,
                               variable=self.max_sat_var, orient=tk.HORIZONTAL,
                               command=self.on_saturation_change, length=200)
        max_sat_scale.pack(fill=tk.X)
        
        # 视觉效果控制
        vis_frame = ttk.LabelFrame(main_frame, text="视觉效果", padding="10")
        vis_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(vis_frame, text="对称性:", fg="white", bg="#2b2b2b").pack(anchor=tk.W)
        self.symmetry_var = tk.IntVar(value=16)
        sym_scale = tk.Scale(vis_frame, from_=4, to=32, resolution=2,
                           variable=self.symmetry_var, orient=tk.HORIZONTAL,
                           command=self.on_symmetry_change, length=200)
        sym_scale.pack(fill=tk.X)
        
        # 粒子数量控制
        tk.Label(vis_frame, text="粒子数量:", fg="white", bg="#2b2b2b").pack(anchor=tk.W)
        self.particle_var = tk.IntVar(value=150)
        particle_scale = tk.Scale(vis_frame, from_=50, to=500, resolution=50,
                                variable=self.particle_var, orient=tk.HORIZONTAL,
                                command=self.on_particle_change, length=200)
        particle_scale.pack(fill=tk.X)
        
        # 状态显示
        status_frame = ttk.LabelFrame(main_frame, text="状态信息", padding="10")
        status_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.status_text = tk.Text(status_frame, height=8, width=40, bg='#1a1a1a', fg='white')
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        # 控制按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="开始/停止", command=self.toggle_animation).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="重置", command=self.reset_visualization).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="退出", command=self.quit_app).pack(side=tk.RIGHT, padx=5)
        
        # 初始化设备列表
        self.refresh_devices()
        
    def refresh_devices(self):
        """刷新音频设备列表"""
        devices = []
        if self.vortex_app.p:
            for i in range(self.vortex_app.p.get_device_count()):
                try:
                    dev_info = self.vortex_app.p.get_device_info_by_index(i)
                    if dev_info['maxInputChannels'] > 0:
                        devices.append(f"{i}: {dev_info['name']}")
                except:
                    pass
        
        self.device_combo['values'] = devices
        if devices:
            self.device_combo.set(devices[0])
            
    def on_source_change(self):
        """音频源改变事件"""
        source = self.source_var.get()
        self.vortex_app.switch_audio_source(source)
        self.update_status(f"切换到音频源: {source}")
        
    def on_device_change(self, event):
        """设备选择改变事件"""
        device_str = self.device_var.get()
        if device_str:
            device_index = int(device_str.split(':')[0])
            self.vortex_app.set_audio_device(device_index)
            self.update_status(f"选择设备: {device_str}")
            
    def on_threshold_change(self, value):
        """音量阈值改变事件"""
        self.vortex_app.saturation_threshold_low = self.low_thresh_var.get()
        self.vortex_app.saturation_threshold_high = self.high_thresh_var.get()
        
    def on_saturation_change(self, value):
        """饱和度参数改变事件"""
        self.vortex_app.min_saturation = self.min_sat_var.get()
        self.vortex_app.max_saturation = self.max_sat_var.get()
        
    def on_symmetry_change(self, value):
        """对称性改变事件"""
        self.vortex_app.symmetry = self.symmetry_var.get()
        
    def on_particle_change(self, value):
        """粒子数量改变事件"""
        self.vortex_app.max_particles_per_ring = self.particle_var.get()
        self.vortex_app.max_center_particles = self.particle_var.get() * 2
        
    def toggle_animation(self):
        """切换动画状态"""
        self.vortex_app.running = not self.vortex_app.running
        status = "运行中" if self.vortex_app.running else "已暂停"
        self.update_status(f"动画状态: {status}")
        
    def reset_visualization(self):
        """重置可视化"""
        # 清空所有粒子
        for i in range(len(self.vortex_app.ring_particles)):
            self.vortex_app.ring_particles[i] = []
        self.vortex_app.center_particles = []
        self.update_status("可视化已重置")
        
    def quit_app(self):
        """退出应用程序"""
        self.vortex_app.running = False
        self.root.quit()
        self.root.destroy()
        
    def update_status(self, message):
        """更新状态信息"""
        self.status_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.status_text.see(tk.END)
        
    def run(self):
        """运行控制面板"""
        self.root.mainloop()

class ParticleVortexKaleidoscope:
    def __init__(self):
        # 音频参数
        self.CHUNK = 2048
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.SILENCE_THRESHOLD = 0.005
        
        # 音频数据队列
        self.audio_queue = queue.Queue()
        
        # 初始化PyAudio
        self.p = None
        self.stream = None
        self.init_pyaudio()
        
        # 创建matplotlib图形
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(14, 14), dpi=120)
        self.fig.canvas.manager.set_window_title('粒子同心圆音频万花筒 - 音量控制饱和度')
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.axis('off')
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        
        # 设置背景
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        
        # 初始化变量
        self.symmetry = 16
        self.rotation_angles = [0] * 12
        self.rotation_speeds = [0.01 + 0.02 * i for i in range(12)]
        self.zoom_factor = 1.0
        self.pulse_phase = 0
        self.time_counter = 0
        
        # 音量控制饱和度的参数
        self.saturation_threshold_low = 0.1   # 低音量阈值
        self.saturation_threshold_high = 0.5  # 高音量阈值
        self.min_saturation = 0.2             # 最小饱和度
        self.max_saturation = 1.0             # 最大饱和度
        
        # 音频特征
        self.audio_features = {
            'volume': 0,
            'centroid': 0,
            'rolloff': 0,
            'bass': 0,
            'mid': 0,
            'treble': 0,
            'beat': False
        }
        
        # 历史数据用于平滑和检测节拍
        self.volume_history = []
        self.max_history = 20
        self.last_beat_time = 0
        
        # 粒子系统
        self.ring_particles = [[] for _ in range(12)]
        self.max_particles_per_ring = 150
        
        # 中心粒子系统
        self.center_particles = []
        self.max_center_particles = 300
        
        # 线程控制
        self.running = True
        self.audio_thread = None
        
        # 音频源设置
        self.audio_source = "microphone"  # 默认麦克风
        self.current_device_index = None
        
        # 启动控制面板
        self.control_panel = AudioControlPanel(self)
        
        # 启动音频捕获
        self.start_audio_capture()
        
        # 设置关闭事件处理
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        print("粒子同心圆万花筒已启动！")
        print(f"饱和度控制: 音量<{self.saturation_threshold_low}时饱和度={self.min_saturation}, 音量>{self.saturation_threshold_high}时饱和度={self.max_saturation}")
        
    def init_pyaudio(self):
        """安全初始化PyAudio"""
        try:
            self.p = pyaudio.PyAudio()
            print("PyAudio初始化成功")
        except Exception as e:
            print(f"PyAudio初始化失败: {e}")
            self.p = None
    
    def switch_audio_source(self, source):
        """切换音频源"""
        self.audio_source = source
        
        # 停止当前音频流
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            except:
                pass
        
        # 停止音频线程
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        
        # 清空音频队列
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
        
        # 启动新的音频源
        self.start_audio_capture()
        
    def set_audio_device(self, device_index):
        """设置音频设备"""
        self.current_device_index = device_index
        self.switch_audio_source("microphone")  # 切换设备时重新启动麦克风
        
    def start_audio_capture(self):
        """启动音频捕获"""
        if self.audio_source == "simulated":
            self.start_simulated_audio()
            return
            
        if not self.p:
            print("PyAudio不可用，使用模拟音频")
            self.start_simulated_audio()
            return
            
        try:
            device_index = self.current_device_index
            
            # 如果没有指定设备，自动选择
            if device_index is None:
                for i in range(self.p.get_device_count()):
                    dev_info = self.p.get_device_info_by_index(i)
                    if dev_info['maxInputChannels'] > 0:
                        # 根据音频源选择设备
                        if self.audio_source == "microphone":
                            if "microphone" in dev_info['name'].lower() or "mic" in dev_info['name'].lower():
                                device_index = i
                                break
                        elif self.audio_source == "system":
                            if "stereo" in dev_info['name'].lower() or "mix" in dev_info['name'].lower() or "what you hear" in dev_info['name'].lower():
                                device_index = i
                                break
                
                # 如果没找到特定设备，使用第一个可用设备
                if device_index is None:
                    for i in range(self.p.get_device_count()):
                        dev_info = self.p.get_device_info_by_index(i)
                        if dev_info['maxInputChannels'] > 0:
                            device_index = i
                            break
            
            if device_index is None:
                print("未找到可用的音频输入设备，切换到模拟音频")
                self.start_simulated_audio()
                return
            
            # 根据音频源调整参数
            if self.audio_source == "system":
                # 系统音频可能需要更高的采样率或不同的格式
                channels = 2  # 立体声
            else:
                channels = self.CHANNELS
            
            self.audio_thread = threading.Thread(target=self.capture_audio, args=(device_index, channels))
            self.audio_thread.daemon = True
            self.audio_thread.start()
            print(f"音频捕获线程已启动 - 源: {self.audio_source}, 设备: {device_index}")
            
        except Exception as e:
            print(f"启动音频捕获失败: {e}")
            self.start_simulated_audio()
    
    def start_simulated_audio(self):
        """启动模拟音频"""
        def simulate():
            import math
            start_time = time.time()
            beat_counter = 0
            
            while self.running:
                current_time = time.time() - start_time
                beat_counter += 1
                
                # 生成更丰富的模拟音频特征
                base_freq = 0.5 + 0.3 * math.sin(current_time * 0.3)
                
                # 模拟节拍
                beat = (beat_counter % 30 == 0)
                
                volume = 0.4 + 0.3 * math.sin(current_time * 2) + 0.1 * math.sin(current_time * 8)
                if beat:
                    volume = min(volume + 0.3, 1.0)
                
                centroid = 0.5 + 0.4 * math.sin(current_time * 1.5)
                rolloff = 0.6 + 0.3 * math.sin(current_time * 1.2)
                bass = 0.3 + 0.2 * math.sin(current_time * 1)
                mid = 0.4 + 0.3 * math.sin(current_time * 2)
                treble = 0.5 + 0.3 * math.sin(current_time * 3)
                
                try:
                    self.audio_queue.put({
                        'volume': volume,
                        'centroid': centroid,
                        'rolloff': rolloff,
                        'bass': bass,
                        'mid': mid,
                        'treble': treble,
                        'beat': beat
                    }, block=False)
                except:
                    pass
                
                time.sleep(0.03)
        
        self.audio_thread = threading.Thread(target=simulate)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        print("模拟音频线程已启动")
        
    def capture_audio(self, device_index, channels=1):
        """捕获音频数据"""
        stream = None
        try:
            stream = self.p.open(
                format=self.FORMAT,
                channels=channels,
                rate=self.RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.CHUNK
            )
            
            print(f"音频流创建成功，开始捕获... 设备: {device_index}, 声道: {channels}")
            
            while self.running:
                try:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    
                    # 如果是立体声，转换为单声道
                    if channels == 2:
                        audio_data = audio_data.reshape(-1, 2)
                        audio_data = np.mean(audio_data, axis=1)
                    
                    audio_data = audio_data / 32768.0
                    
                    self.process_audio_features(audio_data)
                    
                except Exception as e:
                    print(f"音频处理错误: {e}")
                    break
                    
        except Exception as e:
            print(f"音频捕获错误: {e}")
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass
    
    def process_audio_features(self, audio_data):
        """处理音频数据并提取特征"""
        volume = np.sqrt(np.mean(audio_data**2))
        
        # 计算频谱特征
        fft_data = fft(audio_data)
        freqs = np.abs(fft_data[:self.CHUNK//2])
        
        if len(freqs) > 0 and np.sum(freqs) > 0:
            frequencies = np.linspace(0, self.RATE/2, len(freqs))
            
            # 频谱质心
            centroid = np.sum(frequencies * freqs) / np.sum(freqs)
            centroid_norm = centroid / (self.RATE/2)
            
            # 频谱滚降
            cumulative_sum = np.cumsum(freqs)
            threshold = cumulative_sum[-1] * 0.85
            rolloff_idx = np.where(cumulative_sum >= threshold)[0]
            rolloff = frequencies[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
            rolloff_norm = rolloff / (self.RATE/2)
            
            # 频带能量
            bass_band = freqs[:len(freqs)//6]
            bass = np.mean(bass_band) if len(bass_band) > 0 else 0
            bass_norm = bass / np.max(freqs) if np.max(freqs) > 0 else 0
            
            mid_band = freqs[len(freqs)//6:len(freqs)//2]
            mid = np.mean(mid_band) if len(mid_band) > 0 else 0
            mid_norm = mid / np.max(freqs) if np.max(freqs) > 0 else 0
            
            treble_band = freqs[len(freqs)//2:]
            treble = np.mean(treble_band) if len(treble_band) > 0 else 0
            treble_norm = treble / np.max(freqs) if np.max(freqs) > 0 else 0
            
            # 节拍检测
            beat = self.detect_beat(volume)
            
        else:
            centroid_norm = 0.5
            rolloff_norm = 0.5
            bass_norm = 0.3
            mid_norm = 0.4
            treble_norm = 0.5
            beat = False
        
        # 将音频特征放入队列
        try:
            self.audio_queue.put({
                'volume': volume,
                'centroid': centroid_norm,
                'rolloff': rolloff_norm,
                'bass': bass_norm,
                'mid': mid_norm,
                'treble': treble_norm,
                'beat': beat
            }, block=False)
        except:
            pass
    
    def detect_beat(self, current_volume):
        """简单的节拍检测"""
        self.volume_history.append(current_volume)
        if len(self.volume_history) > self.max_history:
            self.volume_history.pop(0)
        
        if len(self.volume_history) < 5:
            return False
        
        # 计算平均音量
        avg_volume = np.mean(self.volume_history)
        
        # 如果当前音量显著高于平均值，检测为节拍
        if current_volume > avg_volume * 1.5 and current_volume > 0.1:
            current_time = time.time()
            # 防止过于频繁的节拍检测
            if current_time - self.last_beat_time > 0.2:
                self.last_beat_time = current_time
                return True
        
        return False
    
    def get_audio_features(self):
        """从队列获取最新的音频特征"""
        try:
            while not self.audio_queue.empty():
                self.audio_features = self.audio_queue.get_nowait()
        except:
            pass
        
        return self.audio_features
    
    def get_volume_based_saturation(self, volume):
        """根据音量计算饱和度"""
        # 使用阈值控制饱和度
        if volume < self.saturation_threshold_low:
            return self.min_saturation
        elif volume > self.saturation_threshold_high:
            return self.max_saturation
        else:
            # 线性插值
            normalized_volume = (volume - self.saturation_threshold_low) / (self.saturation_threshold_high - self.saturation_threshold_low)
            return self.min_saturation + normalized_volume * (self.max_saturation - self.min_saturation)
    
    def generate_rainbow_colors(self, base_hue, volume, bass, mid, treble):
        """生成彩虹色系，饱和度受音量控制"""
        # 基础色调
        hues = []
        for i in range(12):
            hue = (base_hue + i * 0.08) % 1.0
            hues.append(hue)
        
        # 根据音量计算饱和度
        base_saturation = self.get_volume_based_saturation(volume)
        
        colors = []
        glow_colors = []
        neon_colors = []
        
        for hue in hues:
            # 主颜色 - 饱和度受音量控制
            saturation = base_saturation + 0.2 * mid  # 中频增加一些变化
            brightness = 0.7 + 0.3 * volume
            
            rgb_main = mcolors.hsv_to_rgb([hue, saturation, brightness])
            colors.append(rgb_main)
            
            # 发光颜色 - 稍微降低饱和度但增加亮度
            rgb_glow = mcolors.hsv_to_rgb([hue, saturation * 0.7, min(brightness + 0.3, 1.0)])
            glow_colors.append(rgb_glow)
            
            # 霓虹颜色 - 全饱和度和亮度
            rgb_neon = mcolors.hsv_to_rgb([hue, 1.0, 1.0])
            neon_colors.append(rgb_neon)
        
        return colors, glow_colors, neon_colors
    
    def add_ring_particles(self, ring_index, volume, centroid, bass, mid, treble, beat):
        """为特定圆环添加粒子"""
        ring_particles = self.ring_particles[ring_index]
        
        if len(ring_particles) < self.max_particles_per_ring:
            # 基础粒子生成率
            base_spawn_rate = 0.3 + 0.7 * volume
            
            # 节拍时爆发粒子
            if beat:
                base_spawn_rate = 1.0
            
            # 不同圆环有不同的粒子生成特性
            if ring_index < 4:  # 内圈
                spawn_rate = base_spawn_rate * (0.5 + 0.5 * bass)
            elif ring_index < 8:  # 中圈
                spawn_rate = base_spawn_rate * (0.5 + 0.5 * mid)
            else:  # 外圈
                spawn_rate = base_spawn_rate * (0.5 + 0.5 * treble)
            
            if random.random() < spawn_rate * 0.1:
                # 圆环半径
                base_radius = 0.1 + ring_index * 0.15
                radius_variation = 0.05 + 0.1 * volume
                
                # 粒子位置（极坐标）
                angle = random.random() * 2 * math.pi
                radius = base_radius + random.random() * radius_variation
                
                # 粒子速度（切向和径向）
                if ring_index < 4:  # 内圈 - 快速旋转
                    tangential_speed = 0.02 + 0.03 * volume
                    radial_speed = (random.random() - 0.5) * 0.01
                elif ring_index < 8:  # 中圈 - 中等速度
                    tangential_speed = 0.01 + 0.02 * volume
                    radial_speed = (random.random() - 0.5) * 0.005
                else:  # 外圈 - 慢速
                    tangential_speed = 0.005 + 0.01 * volume
                    radial_speed = (random.random() - 0.5) * 0.002
                
                # 粒子大小和生命周期
                size = 3 + random.random() * 10 + 10 * volume
                life = 30 + random.randint(0, 70) + 50 * volume
                
                # 粒子颜色基于频谱质心和圆环位置，饱和度受音量控制
                hue = (centroid + ring_index * 0.08) % 1.0
                saturation = self.get_volume_based_saturation(volume) + 0.2 * mid
                brightness = 0.7 + 0.3 * volume
                color = mcolors.hsv_to_rgb([hue, saturation, brightness])
                
                ring_particles.append({
                    'angle': angle,
                    'radius': radius,
                    'tangential_speed': tangential_speed,
                    'radial_speed': radial_speed,
                    'size': size,
                    'life': life,
                    'max_life': life,
                    'color': color,
                    'pulse_phase': random.random() * 2 * math.pi
                })
    
    def add_center_particles(self, volume, centroid, beat):
        """添加中心粒子"""
        if len(self.center_particles) < self.max_center_particles:
            spawn_rate = 0.2 + 0.8 * volume
            
            # 节拍时爆发
            if beat:
                spawn_rate = 2.0
            
            if random.random() < spawn_rate * 0.05:
                # 中心粒子从中心向外发射
                angle = random.random() * 2 * math.pi
                speed = 0.01 + random.random() * 0.03 + 0.02 * volume
                
                size = 2 + random.random() * 8 + 5 * volume
                life = 20 + random.randint(0, 50) + 30 * volume
                
                # 中心粒子颜色，饱和度受音量控制
                hue = (centroid + 0.5) % 1.0
                saturation = self.get_volume_based_saturation(volume)
                brightness = 0.8 + 0.2 * volume
                color = mcolors.hsv_to_rgb([hue, saturation, brightness])
                
                self.center_particles.append({
                    'x': 0,
                    'y': 0,
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed,
                    'size': size,
                    'life': life,
                    'max_life': life,
                    'color': color,
                    'trail': []  # 粒子轨迹
                })
    
    def update_ring_particles(self, ring_index, volume, bass, mid, treble):
        """更新圆环粒子"""
        ring_particles = self.ring_particles[ring_index]
        
        for particle in ring_particles[:]:
            # 更新角度（旋转）
            particle['angle'] += particle['tangential_speed'] * (1 + volume * 0.5)
            
            # 更新半径（脉动）
            pulse_speed = 0.1 + 0.1 * bass if ring_index < 4 else 0.05 + 0.05 * mid
            particle['pulse_phase'] += pulse_speed
            pulse = math.sin(particle['pulse_phase']) * 0.02 * volume
            
            particle['radius'] += particle['radial_speed'] + pulse
            
            # 限制半径范围
            base_radius = 0.1 + ring_index * 0.15
            min_radius = base_radius * 0.7
            max_radius = base_radius * 1.3
            
            if particle['radius'] < min_radius:
                particle['radius'] = min_radius
                particle['radial_speed'] = abs(particle['radial_speed'])
            elif particle['radius'] > max_radius:
                particle['radius'] = max_radius
                particle['radial_speed'] = -abs(particle['radial_speed'])
            
            # 更新生命周期
            particle['life'] -= 1
            
            # 粒子逐渐缩小
            particle['size'] *= 0.98
            
            if particle['life'] <= 0:
                ring_particles.remove(particle)
    
    def update_center_particles(self, volume):
        """更新中心粒子"""
        for particle in self.center_particles[:]:
            # 更新位置
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            
            # 添加轨迹点
            if len(particle['trail']) < 5:
                particle['trail'].append((particle['x'], particle['y']))
            else:
                particle['trail'].pop(0)
                particle['trail'].append((particle['x'], particle['y']))
            
            # 更新生命周期
            particle['life'] -= 1
            
            # 粒子逐渐缩小
            particle['size'] *= 0.97
            
            # 如果粒子飞出边界或生命周期结束，移除
            distance = math.sqrt(particle['x']**2 + particle['y']**2)
            if particle['life'] <= 0 or distance > 2.0:
                self.center_particles.remove(particle)
    
    def draw_ring_particles(self, ring_index):
        """绘制圆环粒子"""
        ring_particles = self.ring_particles[ring_index]
        
        if not ring_particles:
            return
        
        # 准备绘制数据
        x_coords = []
        y_coords = []
        sizes = []
        colors = []
        alphas = []
        
        for particle in ring_particles:
            # 转换为直角坐标
            x = particle['radius'] * math.cos(particle['angle'])
            y = particle['radius'] * math.sin(particle['angle'])
            
            # 应用圆环旋转
            angle = self.rotation_angles[ring_index]
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            x_rot = x * cos_a - y * sin_a
            y_rot = x * sin_a + y * cos_a
            
            x_coords.append(x_rot)
            y_coords.append(y_rot)
            sizes.append(particle['size'])
            colors.append(particle['color'])
            
            # 透明度基于生命周期
            alpha = particle['life'] / particle['max_life']
            alphas.append(alpha)
        
        # 绘制粒子
        if x_coords:
            self.ax.scatter(x_coords, y_coords, s=sizes, c=colors, alpha=alphas, 
                          edgecolors='white', linewidths=0.5)
    
    def draw_center_particles(self):
        """绘制中心粒子"""
        for particle in self.center_particles:
            alpha = particle['life'] / particle['max_life']
            
            # 绘制粒子
            self.ax.scatter([particle['x']], [particle['y']], 
                          s=particle['size'], color=particle['color'], 
                          alpha=alpha, edgecolors='white', linewidth=0.5)
            
            # 绘制粒子轨迹
            if len(particle['trail']) > 1:
                trail_x = [point[0] for point in particle['trail']]
                trail_y = [point[1] for point in particle['trail']]
                
                # 轨迹渐变透明度
                trail_alphas = [alpha * (i/len(particle['trail'])) for i in range(len(particle['trail']))]
                trail_sizes = [particle['size'] * (i/len(particle['trail'])) for i in range(len(particle['trail']))]
                
                self.ax.scatter(trail_x, trail_y, s=trail_sizes, color=particle['color'], 
                              alpha=trail_alphas, edgecolors='none')
    
    def draw_ring_structures(self, ring_index, volume, centroid, bass, mid, treble):
        """绘制圆环结构（线条和形状）"""
        base_radius = 0.1 + ring_index * 0.15
        radius_variation = 0.02 + 0.08 * volume
        
        # 生成圆环上的点
        num_points = 100 + int(50 * volume)
        theta = np.linspace(0, 2 * np.pi, num_points)
        
        # 圆环形状受音频影响
        if ring_index < 4:  # 内圈受低频影响
            shape_mod = bass * 0.1 * np.sin(8 * theta + self.time_counter * 2)
        elif ring_index < 8:  # 中圈受中频影响
            shape_mod = mid * 0.1 * (np.sin(12 * theta + self.time_counter * 3) + 
                                   0.5 * np.cos(6 * theta - self.time_counter))
        else:  # 外圈受高频影响
            shape_mod = treble * 0.1 * (np.sin(16 * theta + self.time_counter * 4) + 
                                      0.7 * np.cos(10 * theta + self.time_counter * 1.5))
        
        r = base_radius * (1 + shape_mod + radius_variation * np.sin(5 * theta))
        
        # 转换为直角坐标
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # 应用圆环旋转
        angle = self.rotation_angles[ring_index]
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a
        
        # 生成颜色，饱和度受音量控制
        hue = (centroid + ring_index * 0.08) % 1.0
        saturation = self.get_volume_based_saturation(volume) + 0.3 * mid
        brightness = 0.6 + 0.4 * volume
        color = mcolors.hsv_to_rgb([hue, saturation, brightness])
        
        # 绘制圆环线
        line_width = 1 + 3 * volume
        alpha = 0.3 + 0.7 * volume
        
        self.ax.plot(x_rot, y_rot, color=color, linewidth=line_width, alpha=alpha)
        
        # 绘制圆环填充（仅内圈）
        if ring_index < 3:
            fill_alpha = 0.1 + 0.2 * volume
            self.ax.fill(x_rot, y_rot, color=color, alpha=fill_alpha)
    
    def update_frame(self):
        """更新动画帧"""
        if not self.running:
            return False
            
        try:
            features = self.get_audio_features()
            volume = features['volume']
            centroid = features['centroid']
            rolloff = features['rolloff']
            bass = features['bass']
            mid = features['mid']
            treble = features['treble']
            beat = features['beat']
            
            is_silent = volume < self.SILENCE_THRESHOLD
            
            # 清除之前的图形
            self.ax.clear()
            self.ax.axis('off')
            self.ax.set_xlim(-2, 2)
            self.ax.set_ylim(-2, 2)
            
            # 动态背景
            bg_intensity = 0.02 + 0.08 * volume
            self.ax.set_facecolor((bg_intensity, bg_intensity * 0.7, bg_intensity * 1.3))
            
            if is_silent:
                self.draw_fading_vortex()
            else:
                # 更新旋转角度
                for i in range(len(self.rotation_angles)):
                    speed_multiplier = 1 + volume * 2
                    self.rotation_angles[i] += self.rotation_speeds[i] * speed_multiplier
                
                self.time_counter += 0.1
                
                self.draw_particle_vortex(volume, centroid, bass, mid, treble, beat)
            
            # 更新和绘制所有粒子
            for i in range(len(self.ring_particles)):
                self.add_ring_particles(i, volume, centroid, bass, mid, treble, beat)
                self.update_ring_particles(i, volume, bass, mid, treble)
                self.draw_ring_particles(i)
            
            self.add_center_particles(volume, centroid, beat)
            self.update_center_particles(volume)
            self.draw_center_particles()
            
            # 刷新图形
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            return True
            
        except Exception as e:
            print(f"更新错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def draw_fading_vortex(self):
        """绘制淡出的漩涡"""
        # 绘制简单的淡出圆环
        for i in range(12):
            base_radius = 0.1 + i * 0.15
            theta = np.linspace(0, 2 * np.pi, 50)
            r = base_radius * (1 + 0.05 * np.sin(5 * theta))
            
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # 应用缓慢旋转
            angle = self.rotation_angles[i] * 0.1
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            x_rot = x * cos_a - y * sin_a
            y_rot = x * sin_a + y * cos_a
            
            color = (0.3, 0.3, 0.5, 0.1)
            self.ax.plot(x_rot, y_rot, color=color, linewidth=1, alpha=0.05)
    
    def draw_particle_vortex(self, volume, centroid, bass, mid, treble, beat):
        """绘制粒子漩涡"""
        # 绘制所有圆环结构
        for i in range(12):
            self.draw_ring_structures(i, volume, centroid, bass, mid, treble)
        
        # 绘制中心效果
        if volume > 0.1:
            # 中心脉冲圆
            pulse_size = 0.05 + 0.1 * volume + 0.1 * bass
            center_circle = plt.Circle((0, 0), pulse_size, 
                                     color=(1, 1, 1, 0.3 + 0.7 * volume),
                                     fill=True)
            self.ax.add_artist(center_circle)
            
            # 节拍时的中心爆发
            if beat:
                burst_circle = plt.Circle((0, 0), 0.2 + 0.3 * volume,
                                        color=(1, 1, 1, 0.5),
                                        fill=True)
                self.ax.add_artist(burst_circle)
    
    def on_close(self, event):
        """处理窗口关闭事件"""
        print("正在关闭应用程序...")
        self.running = False
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        
        if self.p:
            try:
                self.p.terminate()
            except:
                pass
        
        # 关闭控制面板
        try:
            self.control_panel.quit_app()
        except:
            pass
        
        print("应用程序已关闭")
    
    def run(self):
        """运行应用程序"""
        print("启动粒子同心圆音频万花筒...")
        print("请确保麦克风已连接并授权使用")
        print("控制面板已弹出，可在其中选择音频源和设备")
        print("按Ctrl+C或关闭窗口退出")
        
        # 在单独线程中运行控制面板
        control_thread = threading.Thread(target=self.control_panel.run)
        control_thread.daemon = True
        control_thread.start()
        
        try:
            while self.running:
                if not self.update_frame():
                    break
                plt.pause(0.025)  # 最高帧率
                
        except KeyboardInterrupt:
            print("正在关闭...")
        except Exception as e:
            print(f"应用程序错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.on_close(None)

# 主程序
if __name__ == "__main__":
    app = ParticleVortexKaleidoscope()
    app.run()