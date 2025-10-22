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
        self.fig.canvas.manager.set_window_title('实时音频万花筒 - 增强版')
        self.ax.set_xlim(-1.2, 1.2)
        self.ax.set_ylim(-1.2, 1.2)
        self.ax.axis('off')
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        
        # 设置背景
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        
        # 初始化变量 - 增加对称性和复杂度
        self.symmetry = 12  # 增加对称性
        self.rotation_angle = 0
        self.is_rotating = False
        self.audio_features = {
            'volume': 0,
            'centroid': 0,
            'rolloff': 0,
            'bass': 0,
            'mid': 0,
            'treble': 0
        }
        
        # 增加更多动画参数
        self.pulse_phase = 0
        self.morph_phase = 0
        self.color_shift = 0
        
        # 线程控制
        self.running = True
        self.audio_thread = None
        
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
                
                # 生成更丰富的模拟音频特征
                volume = 0.4 + 0.3 * math.sin(current_time * 2)
                centroid = 0.3 + 0.4 * math.sin(current_time * 1.5)
                rolloff = 0.4 + 0.3 * math.sin(current_time * 1.2)
                bass = 0.5 + 0.3 * math.sin(current_time * 0.8)
                mid = 0.4 + 0.4 * math.sin(current_time * 1.1)
                treble = 0.3 + 0.5 * math.sin(current_time * 1.7)
                
                try:
                    self.audio_queue.put({
                        'volume': volume,
                        'centroid': centroid,
                        'rolloff': rolloff,
                        'bass': bass,
                        'mid': mid,
                        'treble': treble
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
            
            # 计算低频、中频、高频能量
            bass_range = int(len(freqs) * 0.1)
            mid_range = int(len(freqs) * 0.5)
            treble_range = int(len(freqs) * 0.8)
            
            bass = np.mean(freqs[:bass_range]) if bass_range > 0 else 0
            mid = np.mean(freqs[bass_range:mid_range]) if mid_range > bass_range else 0
            treble = np.mean(freqs[mid_range:treble_range]) if treble_range > mid_range else 0
            
            # 归一化
            max_val = max(bass, mid, treble, 1e-10)
            bass_norm = bass / max_val
            mid_norm = mid / max_val
            treble_norm = treble / max_val
        else:
            centroid_norm = 0.5
            rolloff_norm = 0.5
            bass_norm = 0.5
            mid_norm = 0.5
            treble_norm = 0.5
        
        # 将音频特征放入队列
        try:
            self.audio_queue.put({
                'volume': volume,
                'centroid': centroid_norm,
                'rolloff': rolloff_norm,
                'bass': bass_norm,
                'mid': mid_norm,
                'treble': treble_norm
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
    
    def generate_color_palette(self, centroid, volume, bass, mid, treble):
        """根据音频特征生成更丰富的颜色调色板"""
        # 使用所有音频特征来影响颜色
        self.color_shift += 0.01
        
        if centroid > 0.7:
            base_hues = [(0.7 + self.color_shift) % 1.0, 
                         (0.75 + self.color_shift * 0.7) % 1.0, 
                         (0.8 + self.color_shift * 0.5) % 1.0,
                         (0.85 + self.color_shift * 0.3) % 1.0,
                         (0.9 + self.color_shift * 0.9) % 1.0]
        elif centroid < 0.3:
            base_hues = [(0.0 + self.color_shift) % 1.0, 
                         (0.05 + self.color_shift * 0.7) % 1.0, 
                         (0.1 + self.color_shift * 0.5) % 1.0,
                         (0.15 + self.color_shift * 0.3) % 1.0,
                         (0.2 + self.color_shift * 0.9) % 1.0]
        else:
            base_hues = [(0.3 + self.color_shift) % 1.0, 
                         (0.35 + self.color_shift * 0.7) % 1.0, 
                         (0.4 + self.color_shift * 0.5) % 1.0,
                         (0.45 + self.color_shift * 0.3) % 1.0,
                         (0.5 + self.color_shift * 0.9) % 1.0]
        
        # 使用低频、中频、高频来影响饱和度和亮度
        saturation = min(0.6 + 0.4 * volume + 0.2 * mid, 1.0)
        brightness = min(0.4 + 0.6 * volume + 0.3 * treble, 1.0)
        
        colors = []
        for i, hue in enumerate(base_hues):
            # 每个颜色略有不同的饱和度和亮度
            sat_var = saturation * (0.8 + 0.2 * math.sin(self.color_shift * 3 + i))
            bright_var = brightness * (0.8 + 0.2 * math.cos(self.color_shift * 2 + i))
            rgb = mcolors.hsv_to_rgb([hue, sat_var, bright_var])
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
            bass = features['bass']
            mid = features['mid']
            treble = features['treble']
            
            is_silent = volume < self.SILENCE_THRESHOLD
            
            # 更新动画相位
            self.pulse_phase += 0.1 + 0.3 * volume
            self.morph_phase += 0.05 + 0.2 * centroid
            
            # 清除之前的图形
            self.ax.clear()
            self.ax.axis('off')
            self.ax.set_xlim(-1.2, 1.2)
            self.ax.set_ylim(-1.2, 1.2)
            
            # 设置背景颜色 - 更动态
            bg_brightness = 0.05 + 0.1 * volume
            bg_color = (bg_brightness, bg_brightness * 0.7, bg_brightness * 0.9)
            self.ax.set_facecolor(bg_color)
            
            if is_silent:
                self.is_rotating = False
                self.draw_fading_kaleidoscope()
            else:
                self.is_rotating = True
                rotation_speed = 0.03 + 0.15 * volume + 0.1 * treble
                self.rotation_angle += rotation_speed
                self.draw_dynamic_kaleidoscope(volume, centroid, rolloff, bass, mid, treble)
            
            # 刷新图形
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            return True
            
        except Exception as e:
            print(f"更新错误: {e}")
            return False
    
    def draw_fading_kaleidoscope(self):
        """绘制淡出的静态万花筒"""
        # 创建更复杂的几何图案
        num_points = 200
        theta = np.linspace(0, 2*np.pi, num_points)
        
        # 创建多个半径的圆环
        for ring in range(3):
            radius = 0.3 + 0.3 * ring
            
            # 添加波形变形
            deformation = 0.1 * np.sin(8 * theta + self.rotation_angle)
            r = radius * (1 + deformation)
            
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            colors = [(0.5, 0.5, 0.7, 0.2), (0.6, 0.6, 0.8, 0.15), (0.7, 0.7, 0.9, 0.1)]
            
            for i in range(self.symmetry):
                angle = 2 * np.pi * i / self.symmetry + self.rotation_angle * 0.05
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                
                x_rot = x * cos_a - y * sin_a
                y_rot = x * sin_a + y * cos_a
                
                color = colors[ring % len(colors)]
                self.ax.fill(x_rot, y_rot, color=color, alpha=0.15)
                self.ax.plot(x_rot, y_rot, color='white', linewidth=0.5, alpha=0.08)
    
    def draw_dynamic_kaleidoscope(self, volume, centroid, rolloff, bass, mid, treble):
        """绘制动态万花筒 - 大幅增强变化"""
        num_points = 300  # 增加点数以获得更平滑的曲线
        theta = np.linspace(0, 2*np.pi, num_points)
        
        # 基础半径受音量影响更大
        base_radius = 0.3 + 0.7 * volume
        
        # 生成颜色调色板
        colors = self.generate_color_palette(centroid, volume, bass, mid, treble)
        
        # 创建多层结构
        layers = 5
        
        for layer in range(layers):
            layer_factor = layer / layers
            
            # 每层的半径和变形参数不同
            radius_factor = 0.2 + 0.8 * layer_factor
            radius = base_radius * radius_factor
            
            # 大幅增加变形幅度
            deformation_1 = (0.2 * bass + 0.1 * np.sin(self.morph_phase * 2 + layer)) * (1 + 0.5 * volume)
            deformation_2 = (0.2 * mid + 0.1 * np.cos(self.morph_phase * 3 + layer * 0.7)) * (1 + 0.5 * volume)
            deformation_3 = (0.2 * treble + 0.1 * np.sin(self.morph_phase * 5 + layer * 1.2)) * (1 + 0.5 * volume)
            
            # 脉冲效果
            pulse = 0.1 * np.sin(self.pulse_phase + layer)
            
            # 创建复杂形状
            r = radius * (1 + 
                         deformation_1 * np.sin(6 * theta) + 
                         deformation_2 * np.cos(8 * theta) + 
                         deformation_3 * np.sin(12 * theta) + 
                         pulse)
            
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # 为每个对称片段绘制
            for i in range(self.symmetry):
                angle = 2 * np.pi * i / self.symmetry + self.rotation_angle * (1 + 0.5 * layer_factor)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                
                x_rot = x * cos_a - y * sin_a
                y_rot = x * sin_a + y * cos_a
                
                # 颜色选择 - 更复杂的模式
                color_idx = (i + layer * 2) % len(colors)
                color = colors[color_idx]
                
                # 透明度 - 更动态
                alpha = (0.2 + 0.8 * volume) * (0.5 + 0.5 * layer_factor)
                
                # 绘制填充形状
                self.ax.fill(x_rot, y_rot, color=color, alpha=alpha)
                
                # 绘制边框 - 更明显
                border_width = 1 + 2 * volume
                border_alpha = 0.4 + 0.6 * volume
                self.ax.plot(x_rot, y_rot, color='white', linewidth=border_width, alpha=border_alpha)
        
        # 添加中心元素
        self.draw_center_element(volume, centroid, bass, mid, treble)
    
    def draw_center_element(self, volume, centroid, bass, mid, treble):
        """绘制中心元素 - 类似传统万花筒的中心结构"""
        num_points = 100
        theta = np.linspace(0, 2*np.pi, num_points)
        
        # 中心半径
        center_radius = 0.1 + 0.2 * volume
        
        # 创建星形或花朵形状
        points = 6 + int(6 * bass)  # 点数受低频影响
        shape_factor = 0.3 + 0.4 * mid  # 形状受中频影响
        
        r = center_radius * (1 + shape_factor * np.sin(points * theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # 中心颜色 - 受高频影响
        center_hue = (0.1 + 0.8 * treble) % 1.0
        center_color = mcolors.hsv_to_rgb([center_hue, 0.8 + 0.2 * volume, 0.9 + 0.1 * volume])
        
        # 绘制中心形状
        self.ax.fill(x, y, color=center_color, alpha=0.7 + 0.3 * volume)
        self.ax.plot(x, y, color='white', linewidth=2, alpha=0.9)
        
        # 添加旋转的小元素
        small_elements = 8
        for i in range(small_elements):
            angle = 2 * np.pi * i / small_elements + self.rotation_angle * 2
            distance = center_radius * 1.5
            
            elem_x = distance * np.cos(angle)
            elem_y = distance * np.sin(angle)
            
            # 小元素的大小和颜色
            elem_size = 0.05 + 0.05 * volume
            elem_color = mcolors.hsv_to_rgb([(center_hue + 0.3 + 0.2 * i/small_elements) % 1.0, 
                                           0.7 + 0.3 * volume, 0.8 + 0.2 * volume])
            
            # 绘制小元素
            circle = plt.Circle((elem_x, elem_y), elem_size, color=elem_color, alpha=0.8)
            self.ax.add_patch(circle)
    
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
        print("启动增强版实时音频万花筒...")
        print("请确保麦克风已连接并授权使用")
        print("按Ctrl+C或关闭窗口退出")
        
        try:
            # 主循环
            while self.running:
                if not self.update_frame():
                    break
                plt.pause(0.03)  # 提高帧率
                
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