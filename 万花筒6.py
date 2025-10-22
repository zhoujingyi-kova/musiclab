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
from matplotlib.patches import Circle
import math

class RealTime3DKaleidoscope:
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
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 12), dpi=100)
        self.fig.canvas.manager.set_window_title('3D发光音频万花筒')
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.axis('off')
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        
        # 设置背景
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        
        # 初始化变量
        self.symmetry = 12  # 增加对称性
        self.rotation_angle = 0
        self.zoom_factor = 1.0
        self.pulse_phase = 0
        self.is_rotating = False
        self.audio_features = {
            'volume': 0,
            'centroid': 0,
            'rolloff': 0,
            'bass': 0,
            'treble': 0
        }
        
        # 历史数据用于平滑
        self.volume_history = []
        self.max_history = 10
        
        # 粒子系统
        self.particles = []
        self.max_particles = 200
        
        # 线程控制
        self.running = True
        self.audio_thread = None
        
        # 启动音频捕获
        self.start_audio_capture()
        
        # 设置关闭事件处理
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        print("3D发光万花筒已启动！")
        
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
                base_freq = 0.5 + 0.3 * math.sin(current_time * 0.3)
                volume = 0.4 + 0.3 * math.sin(current_time * 2) + 0.1 * math.sin(current_time * 8)
                centroid = 0.5 + 0.4 * math.sin(current_time * 1.5)
                rolloff = 0.6 + 0.3 * math.sin(current_time * 1.2)
                bass = 0.3 + 0.2 * math.sin(current_time * 1)
                treble = 0.4 + 0.3 * math.sin(current_time * 3)
                
                try:
                    self.audio_queue.put({
                        'volume': volume,
                        'centroid': centroid,
                        'rolloff': rolloff,
                        'bass': bass,
                        'treble': treble
                    }, block=False)
                except:
                    pass
                
                time.sleep(0.03)  # 更快的更新
        
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
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)
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
            
            # 低频能量
            bass_band = freqs[:len(freqs)//8]
            bass = np.mean(bass_band) if len(bass_band) > 0 else 0
            bass_norm = bass / np.max(freqs) if np.max(freqs) > 0 else 0
            
            # 高频能量
            treble_band = freqs[len(freqs)*3//4:]
            treble = np.mean(treble_band) if len(treble_band) > 0 else 0
            treble_norm = treble / np.max(freqs) if np.max(freqs) > 0 else 0
            
        else:
            centroid_norm = 0.5
            rolloff_norm = 0.5
            bass_norm = 0.3
            treble_norm = 0.4
        
        # 将音频特征放入队列
        try:
            self.audio_queue.put({
                'volume': volume,
                'centroid': centroid_norm,
                'rolloff': rolloff_norm,
                'bass': bass_norm,
                'treble': treble_norm
            }, block=False)
        except:
            pass
    
    def get_audio_features(self):
        """从队列获取最新的音频特征"""
        try:
            while not self.audio_queue.empty():
                self.audio_features = self.audio_queue.get_nowait()
        except:
            pass
        
        # 平滑音量数据
        self.volume_history.append(self.audio_features['volume'])
        if len(self.volume_history) > self.max_history:
            self.volume_history.pop(0)
        
        smoothed_volume = np.mean(self.volume_history)
        self.audio_features['smoothed_volume'] = smoothed_volume
        
        return self.audio_features
    
    def generate_glow_colors(self, centroid, volume, bass, treble):
        """生成发光颜色调色板"""
        # 根据音频特征选择基础色调
        if centroid > 0.7:
            base_hues = [0.7, 0.75, 0.8, 0.85, 0.9]  # 蓝紫色调
        elif centroid < 0.3:
            base_hues = [0.0, 0.05, 0.1, 0.15, 0.2]  # 红色调
        else:
            base_hues = [0.3, 0.35, 0.4, 0.45, 0.5]  # 绿色调
        
        # 增强饱和度和亮度
        saturation = min(0.8 + 0.4 * volume, 1.0)
        brightness = min(0.6 + 0.6 * volume, 1.0)
        
        colors = []
        glow_colors = []
        
        for i, hue in enumerate(base_hues):
            # 主颜色
            v_main = min(brightness + 0.3 * bass + 0.2 * treble, 1.0)
            rgb_main = mcolors.hsv_to_rgb([hue, saturation, v_main])
            colors.append(rgb_main)
            
            # 发光颜色（更亮）
            v_glow = min(v_main + 0.3, 1.0)
            rgb_glow = mcolors.hsv_to_rgb([hue, saturation * 0.8, v_glow])
            glow_colors.append(rgb_glow)
        
        return colors, glow_colors
    
    def create_3d_shape(self, theta, base_radius, volume, centroid, bass, time_offset=0):
        """创建3D形状"""
        # 基础形状
        r_base = base_radius
        
        # 低频影响的脉动
        pulse = 0.2 * bass * np.sin(8 * theta + self.pulse_phase + time_offset)
        
        # 高频影响的细节
        detail = 0.15 * volume * (np.sin(13 * theta + time_offset) + 
                                0.7 * np.cos(7 * theta - time_offset))
        
        # 频谱质心影响形状复杂度
        complexity = 5 + int(centroid * 8)
        shape_mod = 0.1 * volume * np.sin(complexity * theta + time_offset)
        
        # 组合所有效果
        r = r_base * (1 + pulse + detail + shape_mod)
        
        return r
    
    def add_particles(self, volume, centroid):
        """添加粒子效果"""
        if volume > 0.1 and len(self.particles) < self.max_particles:
            # 根据音量添加粒子
            if np.random.random() < volume * 0.3:
                angle = np.random.random() * 2 * np.pi
                distance = 0.3 + np.random.random() * 0.7
                speed = 0.01 + np.random.random() * 0.02
                size = 2 + np.random.random() * 8
                life = 20 + np.random.randint(0, 60)
                
                # 粒子颜色基于频谱质心
                hue = centroid + (np.random.random() - 0.5) * 0.3
                color = mcolors.hsv_to_rgb([hue % 1.0, 0.8, 0.9])
                
                self.particles.append({
                    'x': np.cos(angle) * distance,
                    'y': np.sin(angle) * distance,
                    'vx': np.cos(angle) * speed,
                    'vy': np.sin(angle) * speed,
                    'size': size,
                    'life': life,
                    'max_life': life,
                    'color': color
                })
    
    def update_particles(self):
        """更新粒子状态"""
        for particle in self.particles[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['life'] -= 1
            
            # 粒子逐渐缩小
            particle['size'] *= 0.97
            
            if particle['life'] <= 0:
                self.particles.remove(particle)
    
    def draw_glow_effect(self, x, y, color, alpha_multiplier=1.0):
        """绘制发光效果"""
        # 多层发光效果
        glow_layers = [
            (1.15, 0.3 * alpha_multiplier, 15),
            (1.3, 0.15 * alpha_multiplier, 25),
            (1.5, 0.08 * alpha_multiplier, 35)
        ]
        
        for scale, alpha, linewidth in glow_layers:
            x_glow = x * scale
            y_glow = y * scale
            self.ax.plot(x_glow, y_glow, color=color, linewidth=linewidth, 
                        alpha=alpha, solid_capstyle='round')
    
    def update_frame(self):
        """更新动画帧"""
        if not self.running:
            return False
            
        try:
            features = self.get_audio_features()
            volume = features['volume']
            smoothed_volume = features['smoothed_volume']
            centroid = features['centroid']
            rolloff = features['rolloff']
            bass = features['bass']
            treble = features['treble']
            
            is_silent = volume < self.SILENCE_THRESHOLD
            
            # 清除之前的图形
            self.ax.clear()
            self.ax.axis('off')
            self.ax.set_xlim(-1.5, 1.5)
            self.ax.set_ylim(-1.5, 1.5)
            
            # 动态背景
            bg_intensity = 0.05 + 0.1 * smoothed_volume
            self.ax.set_facecolor((bg_intensity, bg_intensity, bg_intensity * 1.2))
            
            if is_silent:
                self.is_rotating = False
                self.draw_fading_kaleidoscope()
            else:
                self.is_rotating = True
                # 动态旋转速度
                rotation_speed = 0.01 + 0.15 * volume + 0.1 * bass
                self.rotation_angle += rotation_speed
                self.pulse_phase += 0.1 + 0.2 * treble
                
                # 动态缩放
                self.zoom_factor = 0.8 + 0.4 * smoothed_volume
                
                self.draw_3d_kaleidoscope(volume, centroid, bass, treble)
            
            # 更新和绘制粒子
            self.add_particles(volume, centroid)
            self.update_particles()
            self.draw_particles()
            
            # 刷新图形
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            return True
            
        except Exception as e:
            print(f"更新错误: {e}")
            return False
    
    def draw_fading_kaleidoscope(self):
        """绘制淡出的静态万花筒"""
        theta = np.linspace(0, 2*np.pi, 200)
        base_radius = 0.6
        
        r = base_radius * (1 + 0.1 * np.sin(8 * theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        colors = [(0.3, 0.3, 0.5, 0.2), (0.4, 0.4, 0.6, 0.15), (0.5, 0.5, 0.7, 0.1)]
        
        for i in range(self.symmetry):
            angle = 2 * np.pi * i / self.symmetry + self.rotation_angle * 0.05
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            x_rot = x * cos_a - y * sin_a
            y_rot = x * sin_a + y * cos_a
            
            color = colors[i % len(colors)]
            self.ax.fill(x_rot, y_rot, color=color, alpha=0.1)
            self.ax.plot(x_rot, y_rot, color=(0.8, 0.8, 1.0), linewidth=1, alpha=0.05)
    
    def draw_3d_kaleidoscope(self, volume, centroid, bass, treble):
        """绘制3D发光万花筒"""
        theta = np.linspace(0, 2*np.pi, 300)  # 更多的点以获得更平滑的曲线
        base_radius = 0.5 * self.zoom_factor + 0.3 * volume
        
        colors, glow_colors = self.generate_glow_colors(centroid, volume, bass, treble)
        
        for i in range(self.symmetry):
            angle = 2 * np.pi * i / self.symmetry + self.rotation_angle
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # 创建3D形状
            time_offset = i * 0.5
            r = self.create_3d_shape(theta, base_radius, volume, centroid, bass, time_offset)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # 旋转到对称位置
            x_rot = x * cos_a - y * sin_a
            y_rot = x * sin_a + y * cos_a
            
            color_idx = i % len(colors)
            main_color = colors[color_idx]
            glow_color = glow_colors[color_idx]
            
            # 动态透明度
            alpha = 0.4 + 0.6 * volume
            
            # 绘制发光效果
            self.draw_glow_effect(x_rot, y_rot, glow_color, alpha * treble)
            
            # 绘制主形状
            self.ax.fill(x_rot, y_rot, color=main_color, alpha=alpha * 0.8)
            
            # 绘制边框
            border_width = 2 + 4 * volume
            self.ax.plot(x_rot, y_rot, color=(1, 1, 1, 0.8), 
                        linewidth=border_width, alpha=0.6)
            
            # 绘制内层高光形状（增强3D效果）
            if volume > 0.2:
                r_inner = r * (0.7 + 0.2 * bass)
                x_inner = r_inner * np.cos(theta)
                y_inner = r_inner * np.sin(theta)
                x_inner_rot = x_inner * cos_a - y_inner * sin_a
                y_inner_rot = x_inner * sin_a + y_inner * cos_a
                
                highlight_alpha = 0.3 + 0.4 * treble
                self.ax.fill(x_inner_rot, y_inner_rot, color=(1, 1, 1, 0.5), 
                           alpha=highlight_alpha)
    
    def draw_particles(self):
        """绘制粒子"""
        for particle in self.particles:
            alpha = particle['life'] / particle['max_life']
            size = particle['size'] * alpha
            
            self.ax.scatter(particle['x'], particle['y'], 
                          s=size, color=particle['color'], 
                          alpha=alpha, edgecolors='none')
    
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
        
        print("应用程序已关闭")
    
    def run(self):
        """运行应用程序"""
        print("启动3D发光音频万花筒...")
        print("请确保麦克风已连接并授权使用")
        print("按Ctrl+C或关闭窗口退出")
        
        try:
            while self.running:
                if not self.update_frame():
                    break
                plt.pause(0.03)  # 更高的帧率
                
        except KeyboardInterrupt:
            print("正在关闭...")
        except Exception as e:
            print(f"应用程序错误: {e}")
        finally:
            self.on_close(None)

# 主程序
if __name__ == "__main__":
    app = RealTime3DKaleidoscope()
    app.run()