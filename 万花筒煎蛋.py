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
        self.CHUNK = 2048  # 增加块大小以提高频谱分辨率
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
        self.fig.canvas.manager.set_window_title('粒子同心圆音频万花筒')
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.axis('off')
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        
        # 设置背景
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        
        # 初始化变量
        self.symmetry = 16  # 增加对称性
        self.rotation_angles = [0] * 12  # 12个同心圆层
        self.rotation_speeds = [0.01 + 0.02 * i for i in range(12)]  # 每层不同的旋转速度
        self.zoom_factor = 1.0
        self.pulse_phase = 0
        self.time_counter = 0
        
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
        
        # 粒子系统 - 每个圆环都有自己的粒子
        self.ring_particles = [[] for _ in range(12)]  # 12个圆环
        self.max_particles_per_ring = 150
        
        # 中心粒子系统
        self.center_particles = []
        self.max_center_particles = 300
        
        # 线程控制
        self.running = True
        self.audio_thread = None
        
        # 启动音频捕获
        self.start_audio_capture()
        
        # 设置关闭事件处理
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        print("粒子同心圆万花筒已启动！")
        
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
    
    def generate_rainbow_colors(self, base_hue, volume, bass, mid, treble):
        """生成彩虹色系"""
        # 基础色调
        hues = []
        for i in range(12):
            hue = (base_hue + i * 0.08) % 1.0
            hues.append(hue)
        
        colors = []
        glow_colors = []
        neon_colors = []
        
        for hue in hues:
            # 主颜色 - 根据频带调整饱和度和亮度
            saturation = 0.8 + 0.2 * mid
            brightness = 0.7 + 0.3 * volume
            
            rgb_main = mcolors.hsv_to_rgb([hue, saturation, brightness])
            colors.append(rgb_main)
            
            # 发光颜色
            rgb_glow = mcolors.hsv_to_rgb([hue, saturation * 0.7, min(brightness + 0.3, 1.0)])
            glow_colors.append(rgb_glow)
            
            # 霓虹颜色
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
                
                # 粒子颜色基于频谱质心和圆环位置
                hue = (centroid + ring_index * 0.08) % 1.0
                saturation = 0.8 + 0.2 * mid
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
                
                # 中心粒子颜色
                hue = (centroid + 0.5) % 1.0
                saturation = 0.9
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
        
        # 生成颜色
        hue = (centroid + ring_index * 0.08) % 1.0
        saturation = 0.7 + 0.3 * mid
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
        
        print("应用程序已关闭")
    
    def run(self):
        """运行应用程序"""
        print("启动粒子同心圆音频万花筒...")
        print("请确保麦克风已连接并授权使用")
        print("按Ctrl+C或关闭窗口退出")
        
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