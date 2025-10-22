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

class RealTimeKaleidoscope:
    def __init__(self):
        # 音频参数
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.SILENCE_THRESHOLD = 0.01
        
        # 音频数据队列
        self.audio_queue = queue.Queue()
        
        # 初始化PyAudio
        self.p = None
        self.stream = None
        self.init_pyaudio()
        
        # 创建matplotlib图形
        plt.ion()  # 开启交互模式
        self.fig, self.ax = plt.subplots(figsize=(10, 10), dpi=100)
        self.fig.canvas.manager.set_window_title('实时音频万花筒')
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.axis('off')
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        
        # 设置背景
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        
        # 初始化变量
        self.symmetry = 8
        self.rotation_angle = 0
        self.is_rotating = False
        self.audio_features = {
            'volume': 0,
            'centroid': 0,
            'rolloff': 0
        }
        
        # 线程控制
        self.running = True
        self.audio_thread = None
        
        # 初始化图形
        self.draw_static_kaleidoscope()
        plt.draw()
        plt.pause(0.1)
        
        # 启动音频捕获
        self.start_audio_capture()
        
        # 设置关闭事件处理
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
    def init_pyaudio(self):
        """安全初始化PyAudio"""
        try:
            self.p = pyaudio.PyAudio()
            print("PyAudio初始化成功")
        except Exception as e:
            print(f"PyAudio初始化失败: {e}")
            self.p = None
    
    def start_audio_capture(self):
        """启动音频捕获"""
        if not self.p:
            print("PyAudio不可用，使用模拟音频")
            self.start_simulated_audio()
            return
            
        try:
            # 查找可用的输入设备
            device_index = None
            for i in range(self.p.get_device_count()):
                dev_info = self.p.get_device_info_by_index(i)
                if dev_info['maxInputChannels'] > 0:
                    print(f"找到输入设备: {dev_info['name']} (索引: {i})")
                    device_index = i
                    break
            
            if device_index is None:
                print("未找到可用的音频输入设备，切换到模拟音频")
                self.start_simulated_audio()
                return
            
            # 创建音频线程
            self.audio_thread = threading.Thread(target=self.capture_audio, args=(device_index,))
            self.audio_thread.daemon = True
            self.audio_thread.start()
            print("音频捕获线程已启动")
            
        except Exception as e:
            print(f"启动音频捕获失败: {e}")
            self.start_simulated_audio()
    
    def start_simulated_audio(self):
        """启动模拟音频"""
        def simulate():
            import math
            start_time = time.time()
            
            while self.running:
                current_time = time.time() - start_time
                
                # 生成模拟的音频特征
                volume = 0.3 + 0.2 * math.sin(current_time * 2)
                centroid = 0.4 + 0.3 * math.sin(current_time * 1.5)
                rolloff = 0.5 + 0.2 * math.sin(current_time * 1.2)
                
                try:
                    self.audio_queue.put({
                        'volume': volume,
                        'centroid': centroid,
                        'rolloff': rolloff
                    }, block=False)
                except:
                    pass
                
                time.sleep(0.05)
        
        self.audio_thread = threading.Thread(target=simulate)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        print("模拟音频线程已启动")
        
    def capture_audio(self, device_index):
        """捕获音频数据"""
        stream = None
        try:
            stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.CHUNK
            )
            
            print("音频流创建成功，开始捕获...")
            
            while self.running:
                try:
                    # 读取音频数据
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    audio_data = audio_data / 32768.0
                    
                    # 处理音频特征
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
            centroid = np.sum(frequencies * freqs) / np.sum(freqs)
            centroid_norm = centroid / (self.RATE/2)
            
            cumulative_sum = np.cumsum(freqs)
            threshold = cumulative_sum[-1] * 0.85
            rolloff_idx = np.where(cumulative_sum >= threshold)[0]
            rolloff = frequencies[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
            rolloff_norm = rolloff / (self.RATE/2)
        else:
            centroid_norm = 0.5
            rolloff_norm = 0.5
        
        # 将音频特征放入队列
        try:
            self.audio_queue.put({
                'volume': volume,
                'centroid': centroid_norm,
                'rolloff': rolloff_norm
            }, block=False)
        except:
            pass
    
    def get_audio_features(self):
        """从队列获取最新的音频特征"""
        try:
            # 获取队列中的所有数据，只保留最新的
            while not self.audio_queue.empty():
                self.audio_features = self.audio_queue.get_nowait()
        except:
            pass
        
        return self.audio_features
    
    def draw_static_kaleidoscope(self):
        """绘制静态万花筒图案"""
        theta = np.linspace(0, 2*np.pi, 100)
        radius = 0.7
        
        r = radius * (1 + 0.1 * np.sin(5 * theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        for i in range(self.symmetry):
            angle = 2 * np.pi * i / self.symmetry
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            x_rot = x * cos_a - y * sin_a
            y_rot = x * sin_a + y * cos_a
            
            color = colors[i % len(colors)]
            self.ax.fill(x_rot, y_rot, color=color, alpha=0.7)
            self.ax.plot(x_rot, y_rot, color='white', linewidth=1, alpha=0.5)
    
    def generate_color_palette(self, centroid, volume):
        """根据音频特征生成颜色调色板"""
        if centroid > 0.7:
            base_hues = [0.7, 0.75, 0.8, 0.85, 0.9]
        elif centroid < 0.3:
            base_hues = [0.0, 0.05, 0.1, 0.15, 0.2]
        else:
            base_hues = [0.3, 0.35, 0.4, 0.45, 0.5]
        
        saturation = min(0.7 + 0.3 * volume, 1.0)
        brightness = min(0.5 + 0.5 * volume, 1.0)
        
        colors = []
        for hue in base_hues:
            rgb = mcolors.hsv_to_rgb([hue, saturation, brightness])
            colors.append(rgb)
        
        return colors
    
    def update_frame(self):
        """更新动画帧"""
        if not self.running:
            return False
            
        try:
            features = self.get_audio_features()
            volume = features['volume']
            centroid = features['centroid']
            rolloff = features['rolloff']
            
            is_silent = volume < self.SILENCE_THRESHOLD
            
            # 清除之前的图形
            self.ax.clear()
            self.ax.axis('off')
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
            
            # 设置背景颜色
            bg_brightness = 0.1 + 0.1 * volume
            self.ax.set_facecolor((bg_brightness, bg_brightness, bg_brightness))
            
            if is_silent:
                self.is_rotating = False
                self.draw_fading_kaleidoscope()
            else:
                self.is_rotating = True
                rotation_speed = 0.02 + 0.1 * volume
                self.rotation_angle += rotation_speed
                self.draw_dynamic_kaleidoscope(volume, centroid, rolloff)
            
            # 刷新图形
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            return True
            
        except Exception as e:
            print(f"更新错误: {e}")
            return False
    
    def draw_fading_kaleidoscope(self):
        """绘制淡出的静态万花筒"""
        theta = np.linspace(0, 2*np.pi, 100)
        radius = 0.7
        
        r = radius * (1 + 0.1 * np.sin(5 * theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        colors = [(0.5, 0.5, 0.5, 0.3), (0.6, 0.6, 0.6, 0.3), (0.7, 0.7, 0.7, 0.3)]
        
        for i in range(self.symmetry):
            angle = 2 * np.pi * i / self.symmetry + self.rotation_angle * 0.1
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            x_rot = x * cos_a - y * sin_a
            y_rot = x * sin_a + y * cos_a
            
            color = colors[i % len(colors)]
            self.ax.fill(x_rot, y_rot, color=color, alpha=0.2)
            self.ax.plot(x_rot, y_rot, color='white', linewidth=0.5, alpha=0.1)
    
    def draw_dynamic_kaleidoscope(self, volume, centroid, rolloff):
        """绘制动态万花筒"""
        theta = np.linspace(0, 2*np.pi, 100)
        base_radius = 0.5 + 0.4 * volume
        colors = self.generate_color_palette(centroid, volume)
        
        for i in range(self.symmetry):
            angle = 2 * np.pi * i / self.symmetry + self.rotation_angle
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            for j in range(3):
                r_factor = 0.3 + 0.7 * (j / 3)
                radius = base_radius * r_factor
                
                deformation_1 = 0.1 * centroid + 0.05 * np.sin(self.rotation_angle * 2 + j)
                deformation_2 = 0.1 * rolloff + 0.05 * np.cos(self.rotation_angle * 3 + j * 0.7)
                
                r = radius * (1 + deformation_1 * np.cos(5 * theta) + deformation_2 * np.sin(7 * theta))
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                
                x_rot = x * cos_a - y * sin_a
                y_rot = x * sin_a + y * cos_a
                
                color_idx = (i + j) % len(colors)
                color = colors[color_idx]
                
                alpha = 0.3 + 0.7 * volume
                self.ax.fill(x_rot, y_rot, color=color, alpha=alpha)
                
                border_alpha = 0.3 + 0.7 * volume
                self.ax.plot(x_rot, y_rot, color='white', linewidth=1, alpha=border_alpha)
    
    def on_close(self, event):
        """处理窗口关闭事件"""
        print("正在关闭应用程序...")
        self.running = False
        
        # 关闭音频流
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        
        # 终止PyAudio
        if self.p:
            try:
                self.p.terminate()
            except:
                pass
        
        print("应用程序已关闭")
    
    def run(self):
        """运行应用程序"""
        print("启动实时音频万花筒...")
        print("请确保麦克风已连接并授权使用")
        print("按Ctrl+C或关闭窗口退出")
        
        try:
            # 主循环
            while self.running:
                if not self.update_frame():
                    break
                plt.pause(0.05)  # 控制帧率
                
        except KeyboardInterrupt:
            print("正在关闭...")
        except Exception as e:
            print(f"应用程序错误: {e}")
        finally:
            self.on_close(None)

# 主程序
if __name__ == "__main__":
    app = RealTimeKaleidoscope()
    app.run()