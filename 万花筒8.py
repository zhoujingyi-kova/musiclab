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

class ParticleVortexKaleidoscope:
    def __init__(self):
        # 音频参数
        self.CHUNK = 2048
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.SILENCE_THRESHOLD = 0.005
        
        # 音频数据队列
        self.audio_queue = queue.Queue(maxsize=10)
        
        # 初始化PyAudio
        self.p = None
        self.stream = None
        self.stream_lock = threading.Lock()
        self.init_pyaudio()
        
        # 创建matplotlib图形
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(14, 14), dpi=120)
        self.fig.canvas.manager.set_window_title('粒子同心圆音频万花筒 - 颜色范围限制版')
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
        self.saturation_threshold_low = 0.1
        self.saturation_threshold_high = 0.5
        self.min_saturation = 0.2
        self.max_saturation = 1.0
        
        # 颜色范围限制参数
        self.color_range_min = 0.0    # 最小颜色值
        self.color_range_max = 1.0    # 最大颜色值
        self.min_color = (0.0, 0.0, 0.0)      # 最小值对应的颜色 (黑色)
        self.max_color = (1.0, 1.0, 1.0)      # 最大值对应的颜色 (白色)
        
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
        self.audio_thread_running = False
        
        # 启动音频捕获
        self.start_audio_capture()
        
        # 设置关闭事件处理
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        print("颜色范围限制版粒子同心圆万花筒已启动！")
        print(f"颜色范围: {self.color_range_min} - {self.color_range_max}")
        print(f"最小值颜色: {self.min_color}")
        print(f"最大值颜色: {self.max_color}")
        
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
            
            self.audio_thread_running = True
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
            beat_counter = 0
            
            while self.running and self.audio_thread_running:
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
                    }, block=False, timeout=0.01)
                except:
                    pass
                
                time.sleep(0.03)
        
        self.audio_thread_running = True
        self.audio_thread = threading.Thread(target=simulate)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        print("模拟音频线程已启动")
        
    def capture_audio(self, device_index):
        """捕获音频数据"""
        stream = None
        try:
            with self.stream_lock:
                stream = self.p.open(
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=self.CHUNK,
                    stream_callback=None
                )
                self.stream = stream
            
            print("音频流创建成功，开始捕获...")
            
            while self.running and self.audio_thread_running:
                try:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    audio_data = audio_data.astype(np.float32) / 32768.0
                    
                    self.process_audio_features(audio_data)
                    
                except IOError as e:
                    if "Input overflowed" in str(e):
                        print("音频输入溢出，跳过此帧")
                        continue
                    else:
                        print(f"音频IO错误: {e}")
                        break
                except Exception as e:
                    print(f"音频处理错误: {e}")
                    break
                    
        except Exception as e:
            print(f"音频捕获错误: {e}")
        finally:
            self.safe_close_stream()
    
    def safe_close_stream(self):
        """安全关闭音频流"""
        with self.stream_lock:
            if hasattr(self, 'stream') and self.stream:
                try:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                    self.stream.close()
                    self.stream = None
                except Exception as e:
                    print(f"关闭音频流时出错: {e}")
    
    def process_audio_features(self, audio_data):
        """处理音频数据并提取特征"""
        volume = np.sqrt(np.mean(audio_data**2))
        
        if len(self.volume_history) > 0 and volume == self.volume_history[-1]:
            return
            
        try:
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
                
                bass_band = freqs[:len(freqs)//6]
                bass = np.mean(bass_band) if len(bass_band) > 0 else 0
                bass_norm = bass / np.max(freqs) if np.max(freqs) > 0 else 0
                
                mid_band = freqs[len(freqs)//6:len(freqs)//2]
                mid = np.mean(mid_band) if len(mid_band) > 0 else 0
                mid_norm = mid / np.max(freqs) if np.max(freqs) > 0 else 0
                
                treble_band = freqs[len(freqs)//2:]
                treble = np.mean(treble_band) if len(treble_band) > 0 else 0
                treble_norm = treble / np.max(freqs) if np.max(freqs) > 0 else 0
                
                beat = self.detect_beat(volume)
                
            else:
                centroid_norm = 0.5
                rolloff_norm = 0.5
                bass_norm = 0.3
                mid_norm = 0.4
                treble_norm = 0.5
                beat = False
            
            try:
                self.audio_queue.put({
                    'volume': volume,
                    'centroid': centroid_norm,
                    'rolloff': rolloff_norm,
                    'bass': bass_norm,
                    'mid': mid_norm,
                    'treble': treble_norm,
                    'beat': beat
                }, block=False, timeout=0.01)
            except:
                pass
                
        except Exception as e:
            print(f"频谱处理错误: {e}")
    
    def detect_beat(self, current_volume):
        """简单的节拍检测"""
        self.volume_history.append(current_volume)
        if len(self.volume_history) > self.max_history:
            self.volume_history.pop(0)
        
        if len(self.volume_history) < 5:
            return False
        
        avg_volume = np.mean(self.volume_history)
        
        if current_volume > avg_volume * 1.5 and current_volume > 0.1:
            current_time = time.time()
            if current_time - self.last_beat_time > 0.2:
                self.last_beat_time = current_time
                return True
        
        return False
    
    def get_audio_features(self):
        """从队列获取最新的音频特征"""
        try:
            latest_features = None
            while not self.audio_queue.empty():
                latest_features = self.audio_queue.get_nowait()
            
            if latest_features:
                self.audio_features = latest_features
        except:
            pass
        
        return self.audio_features
    
    def get_volume_based_saturation(self, volume):
        """根据音量计算饱和度"""
        if volume < self.saturation_threshold_low:
            return self.min_saturation
        elif volume > self.saturation_threshold_high:
            return self.max_saturation
        else:
            normalized_volume = (volume - self.saturation_threshold_low) / (self.saturation_threshold_high - self.saturation_threshold_low)
            return self.min_saturation + normalized_volume * (self.max_saturation - self.min_saturation)
    
    def map_to_color_range(self, value):
        """将值映射到颜色范围，超出范围的按极值处理"""
        if value <= self.color_range_min:
            return self.min_color
        elif value >= self.color_range_max:
            return self.max_color
        else:
            # 线性插值
            normalized_value = (value - self.color_range_min) / (self.color_range_max - self.color_range_min)
            r = self.min_color[0] + normalized_value * (self.max_color[0] - self.min_color[0])
            g = self.min_color[1] + normalized_value * (self.max_color[1] - self.min_color[1])
            b = self.min_color[2] + normalized_value * (self.max_color[2] - self.min_color[2])
            return (r, g, b)
    
    def safe_hsv_to_rgb(self, hsv):
        """安全的HSV到RGB转换，使用颜色范围映射"""
        try:
            rgb = mcolors.hsv_to_rgb(hsv)
            # 对每个通道应用颜色范围映射
            r = self.map_to_color_range(rgb[0])[0]
            g = self.map_to_color_range(rgb[1])[1]
            b = self.map_to_color_range(rgb[2])[2]
            return (r, g, b)
        except Exception as e:
            print(f"颜色转换错误: {e}, HSV: {hsv}")
            return self.min_color  # 返回最小值颜色作为默认
    
    def generate_rainbow_colors(self, base_hue, volume, bass, mid, treble):
        """生成彩虹色系，饱和度受音量控制"""
        hues = []
        for i in range(12):
            hue = (base_hue + i * 0.08) % 1.0
            hues.append(hue)
        
        base_saturation = self.get_volume_based_saturation(volume)
        
        colors = []
        glow_colors = []
        neon_colors = []
        
        for hue in hues:
            # 确保参数在有效范围内
            hue = max(0.0, min(1.0, hue))
            saturation = max(0.0, min(1.0, base_saturation + 0.2 * mid))
            brightness = max(0.0, min(1.0, 0.7 + 0.3 * volume))
            
            rgb_main = self.safe_hsv_to_rgb([hue, saturation, brightness])
            colors.append(rgb_main)
            
            # 发光颜色
            glow_saturation = max(0.0, min(1.0, saturation * 0.7))
            glow_brightness = max(0.0, min(1.0, brightness + 0.3))
            rgb_glow = self.safe_hsv_to_rgb([hue, glow_saturation, glow_brightness])
            glow_colors.append(rgb_glow)
            
            # 霓虹颜色
            rgb_neon = self.safe_hsv_to_rgb([hue, 1.0, 1.0])
            neon_colors.append(rgb_neon)
        
        return colors, glow_colors, neon_colors
    
    def add_ring_particles(self, ring_index, volume, centroid, bass, mid, treble, beat):
        """为特定圆环添加粒子"""
        ring_particles = self.ring_particles[ring_index]
        
        if len(ring_particles) < self.max_particles_per_ring:
            base_spawn_rate = 0.3 + 0.7 * volume
            
            if beat:
                base_spawn_rate = 1.0
            
            if ring_index < 4:
                spawn_rate = base_spawn_rate * (0.5 + 0.5 * bass)
            elif ring_index < 8:
                spawn_rate = base_spawn_rate * (0.5 + 0.5 * mid)
            else:
                spawn_rate = base_spawn_rate * (0.5 + 0.5 * treble)
            
            if random.random() < spawn_rate * 0.1:
                base_radius = 0.1 + ring_index * 0.15
                radius_variation = 0.05 + 0.1 * volume
                
                angle = random.random() * 2 * math.pi
                radius = base_radius + random.random() * radius_variation
                
                if ring_index < 4:
                    tangential_speed = 0.02 + 0.03 * volume
                    radial_speed = (random.random() - 0.5) * 0.01
                elif ring_index < 8:
                    tangential_speed = 0.01 + 0.02 * volume
                    radial_speed = (random.random() - 0.5) * 0.005
                else:
                    tangential_speed = 0.005 + 0.01 * volume
                    radial_speed = (random.random() - 0.5) * 0.002
                
                size = 3 + random.random() * 10 + 10 * volume
                life = 30 + random.randint(0, 70) + 50 * volume
                
                hue = (centroid + ring_index * 0.08) % 1.0
                saturation = max(0.0, min(1.0, self.get_volume_based_saturation(volume) + 0.2 * mid))
                brightness = max(0.0, min(1.0, 0.7 + 0.3 * volume))
                color = self.safe_hsv_to_rgb([hue, saturation, brightness])
                
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
            
            if beat:
                spawn_rate = 2.0
            
            if random.random() < spawn_rate * 0.05:
                angle = random.random() * 2 * math.pi
                speed = 0.01 + random.random() * 0.03 + 0.02 * volume
                
                size = 2 + random.random() * 8 + 5 * volume
                life = 20 + random.randint(0, 50) + 30 * volume
                
                hue = (centroid + 0.5) % 1.0
                saturation = max(0.0, min(1.0, self.get_volume_based_saturation(volume)))
                brightness = max(0.0, min(1.0, 0.8 + 0.2 * volume))
                color = self.safe_hsv_to_rgb([hue, saturation, brightness])
                
                self.center_particles.append({
                    'x': 0,
                    'y': 0,
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed,
                    'size': size,
                    'life': life,
                    'max_life': life,
                    'color': color,
                    'trail': []
                })
    
    def update_ring_particles(self, ring_index, volume, bass, mid, treble):
        """更新圆环粒子"""
        ring_particles = self.ring_particles[ring_index]
        
        for particle in ring_particles[:]:
            particle['angle'] += particle['tangential_speed'] * (1 + volume * 0.5)
            
            pulse_speed = 0.1 + 0.1 * bass if ring_index < 4 else 0.05 + 0.05 * mid
            particle['pulse_phase'] += pulse_speed
            pulse = math.sin(particle['pulse_phase']) * 0.02 * volume
            
            particle['radius'] += particle['radial_speed'] + pulse
            
            base_radius = 0.1 + ring_index * 0.15
            min_radius = base_radius * 0.7
            max_radius = base_radius * 1.3
            
            if particle['radius'] < min_radius:
                particle['radius'] = min_radius
                particle['radial_speed'] = abs(particle['radial_speed'])
            elif particle['radius'] > max_radius:
                particle['radius'] = max_radius
                particle['radial_speed'] = -abs(particle['radial_speed'])
            
            particle['life'] -= 1
            particle['size'] *= 0.98
            
            if particle['life'] <= 0:
                ring_particles.remove(particle)
    
    def update_center_particles(self, volume):
        """更新中心粒子"""
        for particle in self.center_particles[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            
            if len(particle['trail']) < 5:
                particle['trail'].append((particle['x'], particle['y']))
            else:
                particle['trail'].pop(0)
                particle['trail'].append((particle['x'], particle['y']))
            
            particle['life'] -= 1
            particle['size'] *= 0.97
            
            distance = math.sqrt(particle['x']**2 + particle['y']**2)
            if particle['life'] <= 0 or distance > 2.0:
                self.center_particles.remove(particle)
    
    def draw_ring_particles(self, ring_index):
        """绘制圆环粒子"""
        ring_particles = self.ring_particles[ring_index]
        
        if not ring_particles:
            return
        
        x_coords = []
        y_coords = []
        sizes = []
        colors = []
        alphas = []
        
        for particle in ring_particles:
            x = particle['radius'] * math.cos(particle['angle'])
            y = particle['radius'] * math.sin(particle['angle'])
            
            angle = self.rotation_angles[ring_index]
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            x_rot = x * cos_a - y * sin_a
            y_rot = x * sin_a + y * cos_a
            
            x_coords.append(x_rot)
            y_coords.append(y_rot)
            sizes.append(particle['size'])
            
            # 确保颜色是有效的RGB元组
            if isinstance(particle['color'], (list, np.ndarray)):
                # 对每个通道应用颜色范围映射
                r = self.map_to_color_range(particle['color'][0])[0]
                g = self.map_to_color_range(particle['color'][1])[1]
                b = self.map_to_color_range(particle['color'][2])[2]
                color = (r, g, b)
            else:
                color = particle['color']
            colors.append(color)
            
            alpha = max(0.0, min(1.0, particle['life'] / particle['max_life']))
            alphas.append(alpha)
        
        if x_coords:
            try:
                self.ax.scatter(x_coords, y_coords, s=sizes, c=colors, alpha=alphas, 
                              edgecolors='white', linewidths=0.5)
            except ValueError as e:
                print(f"绘制圆环粒子时颜色错误: {e}")
                # 使用默认颜色
                self.ax.scatter(x_coords, y_coords, s=sizes, c='white', alpha=alphas, 
                              edgecolors='white', linewidths=0.5)
    
    def draw_center_particles(self):
        """绘制中心粒子"""
        for particle in self.center_particles:
            alpha = max(0.0, min(1.0, particle['life'] / particle['max_life']))
            
            # 确保颜色是有效的RGB元组
            if isinstance(particle['color'], (list, np.ndarray)):
                # 对每个通道应用颜色范围映射
                r = self.map_to_color_range(particle['color'][0])[0]
                g = self.map_to_color_range(particle['color'][1])[1]
                b = self.map_to_color_range(particle['color'][2])[2]
                color = (r, g, b)
            else:
                color = particle['color']
            
            try:
                self.ax.scatter([particle['x']], [particle['y']], 
                              s=particle['size'], color=color, 
                              alpha=alpha, edgecolors='white', linewidth=0.5)
            except ValueError as e:
                print(f"绘制中心粒子时颜色错误: {e}")
                # 使用默认颜色
                self.ax.scatter([particle['x']], [particle['y']], 
                              s=particle['size'], color='white', 
                              alpha=alpha, edgecolors='white', linewidth=0.5)
            
            if len(particle['trail']) > 1:
                trail_x = [point[0] for point in particle['trail']]
                trail_y = [point[1] for point in particle['trail']]
                
                trail_alphas = [alpha * (i/len(particle['trail'])) for i in range(len(particle['trail']))]
                trail_sizes = [particle['size'] * (i/len(particle['trail'])) for i in range(len(particle['trail']))]
                
                try:
                    self.ax.scatter(trail_x, trail_y, s=trail_sizes, color=color, 
                                  alpha=trail_alphas, edgecolors='none')
                except ValueError as e:
                    print(f"绘制粒子轨迹时颜色错误: {e}")
                    # 使用默认颜色
                    self.ax.scatter(trail_x, trail_y, s=trail_sizes, color='white', 
                                  alpha=trail_alphas, edgecolors='none')
    
    def draw_ring_structures(self, ring_index, volume, centroid, bass, mid, treble):
        """绘制圆环结构"""
        base_radius = 0.1 + ring_index * 0.15
        radius_variation = 0.02 + 0.08 * volume
        
        num_points = 100 + int(50 * volume)
        theta = np.linspace(0, 2 * np.pi, num_points)
        
        if ring_index < 4:
            shape_mod = bass * 0.1 * np.sin(8 * theta + self.time_counter * 2)
        elif ring_index < 8:
            shape_mod = mid * 0.1 * (np.sin(12 * theta + self.time_counter * 3) + 
                                   0.5 * np.cos(6 * theta - self.time_counter))
        else:
            shape_mod = treble * 0.1 * (np.sin(16 * theta + self.time_counter * 4) + 
                                      0.7 * np.cos(10 * theta + self.time_counter * 1.5))
        
        r = base_radius * (1 + shape_mod + radius_variation * np.sin(5 * theta))
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        angle = self.rotation_angles[ring_index]
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a
        
        hue = (centroid + ring_index * 0.08) % 1.0
        saturation = max(0.0, min(1.0, self.get_volume_based_saturation(volume) + 0.3 * mid))
        brightness = max(0.0, min(1.0, 0.6 + 0.4 * volume))
        color = self.safe_hsv_to_rgb([hue, saturation, brightness])
        
        line_width = 1 + 3 * volume
        alpha = max(0.0, min(1.0, 0.3 + 0.7 * volume))
        
        try:
            self.ax.plot(x_rot, y_rot, color=color, linewidth=line_width, alpha=alpha)
        except ValueError as e:
            print(f"绘制圆环结构时颜色错误: {e}")
            # 使用默认颜色
            self.ax.plot(x_rot, y_rot, color='white', linewidth=line_width, alpha=alpha)
        
        if ring_index < 3:
            fill_alpha = max(0.0, min(1.0, 0.1 + 0.2 * volume))
            try:
                self.ax.fill(x_rot, y_rot, color=color, alpha=fill_alpha)
            except ValueError as e:
                print(f"填充圆环时颜色错误: {e}")
                # 使用默认颜色
                self.ax.fill(x_rot, y_rot, color='white', alpha=fill_alpha)
    
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
            
            # 动态背景 - 应用颜色范围映射
            bg_r_raw = 0.02 + 0.08 * volume
            bg_g_raw = bg_r_raw * 0.7
            bg_b_raw = bg_r_raw * 1.3
            
            bg_r = self.map_to_color_range(bg_r_raw)[0]
            bg_g = self.map_to_color_range(bg_g_raw)[1]
            bg_b = self.map_to_color_range(bg_b_raw)[2]
            
            self.ax.set_facecolor((bg_r, bg_g, bg_b))
            
            if is_silent:
                self.draw_fading_vortex()
            else:
                max_speed_multiplier = 3.0
                speed_multiplier = min(1 + volume * 2, max_speed_multiplier)
                
                for i in range(len(self.rotation_angles)):
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
        for i in range(12):
            base_radius = 0.1 + i * 0.15
            theta = np.linspace(0, 2 * np.pi, 50)
            r = base_radius * (1 + 0.05 * np.sin(5 * theta))
            
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            angle = self.rotation_angles[i] * 0.1
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            x_rot = x * cos_a - y * sin_a
            y_rot = x * sin_a + y * cos_a
            
            color = (0.3, 0.3, 0.5, 0.1)
            self.ax.plot(x_rot, y_rot, color=color, linewidth=1, alpha=0.05)
    
    def draw_particle_vortex(self, volume, centroid, bass, mid, treble, beat):
        """绘制粒子漩涡"""
        for i in range(12):
            self.draw_ring_structures(i, volume, centroid, bass, mid, treble)
        
        if volume > 0.1:
            pulse_size = max(0.0, 0.05 + 0.1 * volume + 0.1 * bass)
            center_alpha = max(0.0, min(1.0, 0.3 + 0.7 * volume))
            center_circle = plt.Circle((0, 0), pulse_size, 
                                     color=(1, 1, 1, center_alpha),
                                     fill=True)
            self.ax.add_artist(center_circle)
            
            if beat:
                burst_size = max(0.0, 0.2 + 0.3 * volume)
                burst_circle = plt.Circle((0, 0), burst_size,
                                        color=(1, 1, 1, 0.5),
                                        fill=True)
                self.ax.add_artist(burst_circle)
    
    def on_close(self, event):
        """处理窗口关闭事件"""
        print("正在安全关闭应用程序...")
        self.running = False
        self.audio_thread_running = False
        
        if self.audio_thread and self.audio_thread.is_alive():
            print("等待音频线程结束...")
            self.audio_thread.join(timeout=2.0)
        
        self.safe_close_stream()
        
        if self.p:
            try:
                self.p.terminate()
                print("PyAudio已终止")
            except Exception as e:
                print(f"终止PyAudio时出错: {e}")
        
        print("应用程序已安全关闭")
    
    def run(self):
        """运行应用程序"""
        print("启动颜色范围限制版粒子同心圆音频万花筒...")
        print("请确保麦克风已连接并授权使用")
        print("按Ctrl+C或关闭窗口退出")
        
        try:
            frame_count = 0
            while self.running:
                if not self.update_frame():
                    break
                
                frame_count += 1
                if frame_count % 100 == 0:
                    if len(self.center_particles) > self.max_center_particles * 1.5:
                        self.center_particles = self.center_particles[-self.max_center_particles:]
                    for i in range(len(self.ring_particles)):
                        if len(self.ring_particles[i]) > self.max_particles_per_ring * 1.5:
                            self.ring_particles[i] = self.ring_particles[i][-self.max_particles_per_ring:]
                
                plt.pause(0.025)
                
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