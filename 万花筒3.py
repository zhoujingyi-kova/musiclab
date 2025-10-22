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
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy.integrate import solve_ivp

class AttractorKaleidoscope:
    def __init__(self):
        # 音频参数
        self.CHUNK = 1024
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
        
        # 创建图形 - 使用2D提高性能
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 12), dpi=100, facecolor='black')
        self.fig.canvas.manager.set_window_title('吸引子万花筒 - 流畅分形版')
        
        # 设置2D视图
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ax.axis('off')
        self.ax.set_facecolor('black')
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        
        # 初始化变量
        self.symmetry = 8
        self.rotation_angle = 0
        self.is_silent = True
        
        # 音频特征
        self.audio_features = {
            'volume': 0,
            'centroid': 0.5,
            'bass': 0.5,
            'mid': 0.5,
            'treble': 0.5
        }
        
        # 吸引子参数
        self.a = 80
        self.c = 0.083
        self.d = 0.5
        self.e = 0.05
        self.f = 30
        
        # 粒子系统
        self.particles = []
        self.max_particles = 500
        self.trail_length = 50
        
        # 霓虹颜色调色板
        self.neon_colors = [
            (0.0, 1.0, 1.0),    # 荧光蓝
            (0.0, 1.0, 0.0),    # 亮绿
            (1.0, 0.0, 0.5),    # 玫粉
            (0.5, 0.0, 1.0),    # 荧光紫
            (1.0, 1.0, 0.0),    # 荧光黄
            (1.0, 0.3, 0.0)     # 荧光橙
        ]
        
        # 动画参数
        self.pulse_phase = 0
        self.morph_phase = 0
        self.color_shift = 0
        
        # 性能优化
        self.last_update_time = time.time()
        self.target_fps = 60  # 提高帧率
        self.frame_count = 0
        
        # 预计算缓冲区
        self.particle_buffer = []
        self.buffer_size = 100
        
        # 线程控制
        self.running = True
        self.audio_thread = None
        
        # 初始化粒子
        self.init_particles()
        
        # 启动音频捕获
        self.start_audio_capture()
        
        # 设置关闭事件处理
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        print("吸引子万花筒初始化完成")
    
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
            start_time = time.time()
            
            while self.running:
                current_time = time.time() - start_time
                
                # 生成有节奏的模拟音频
                base_freq = 0.8
                volume = 0.3 + 0.25 * abs(math.sin(current_time * base_freq))
                volume += 0.15 * math.sin(current_time * base_freq * 3)
                
                centroid = 0.4 + 0.3 * math.sin(current_time * 0.7)
                bass = 0.3 + 0.4 * abs(math.sin(current_time * base_freq))
                mid = 0.4 + 0.3 * math.sin(current_time * base_freq * 2)
                treble = 0.5 + 0.3 * math.sin(current_time * base_freq * 4)
                
                # 随机脉冲
                if random.random() < 0.08:
                    volume = min(1.0, volume + 0.5)
                
                try:
                    self.audio_queue.put({
                        'volume': volume,
                        'centroid': centroid,
                        'bass': bass,
                        'mid': mid,
                        'treble': treble
                    }, block=False)
                except:
                    pass
                
                time.sleep(0.016)  # 约60Hz更新频率
        
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
                    
                    gain = 3.0  # 提高增益
                    audio_data = np.clip(audio_data * gain, -1.0, 1.0)
                    
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
        compressed_volume = 1.0 - np.exp(-10.0 * volume)  # 更强的压缩
        
        fft_data = fft(audio_data)
        freqs = np.abs(fft_data[:self.CHUNK//2])
        
        if len(freqs) > 0 and np.sum(freqs) > 0:
            frequencies = np.linspace(0, self.RATE/2, len(freqs))
            
            centroid = np.sum(frequencies * freqs) / np.sum(freqs)
            centroid_norm = min(centroid / (self.RATE/4), 1.0)
            
            bass_range = int(len(freqs) * 0.1)
            mid_range = int(len(freqs) * 0.5)
            treble_range = int(len(freqs) * 0.8)
            
            bass = np.mean(freqs[:bass_range]) if bass_range > 0 else 0
            mid = np.mean(freqs[bass_range:mid_range]) if mid_range > bass_range else 0
            treble = np.mean(freqs[mid_range:treble_range]) if treble_range > mid_range else 0
            
            max_val = max(bass, mid, treble, 1e-10)
            bass_norm = (bass / max_val) ** 0.5  # 更强的非线性
            mid_norm = (mid / max_val) ** 0.5
            treble_norm = (treble / max_val) ** 0.5
        else:
            centroid_norm = 0.5
            bass_norm = 0.5
            mid_norm = 0.5
            treble_norm = 0.5
        
        try:
            self.audio_queue.put({
                'volume': compressed_volume,
                'centroid': centroid_norm,
                'bass': bass_norm,
                'mid': mid_norm,
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
        
        return self.audio_features
    
    def attractor_equations(self, t, state):
        """吸引子微分方程"""
        x, y, z = state
        
        # 根据你提供的方程，但修正第一个方程（dx/dt = α(x-x) 应该是笔误）
        # 我假设第一个方程应该是类似 dx/dt = α(y - x) 的形式
        dxdt = self.a * (y - x)  # 修正后的方程
        dydt = self.d * x - self.a * x + self.f * y
        dzdt = -self.c * z**2 + x * y + self.c * z
        
        return [dxdt, dydt, dzdt]
    
    def init_particles(self):
        """初始化粒子系统"""
        self.particles = []
        for _ in range(self.max_particles):
            self.create_particle()
    
    def create_particle(self):
        """创建新粒子"""
        # 随机初始位置
        x0 = random.uniform(-2, 2)
        y0 = random.uniform(-2, 2)
        z0 = random.uniform(-2, 2)
        
        # 随机颜色
        color_idx = random.randint(0, len(self.neon_colors) - 1)
        base_color = self.neon_colors[color_idx]
        
        # 调整颜色亮度
        brightness = random.uniform(0.7, 1.0)
        color = tuple(c * brightness for c in base_color)
        
        particle = {
            'position': [x0, y0, z0],
            'trail': [],
            'color': color,
            'speed': random.uniform(0.5, 2.0),
            'size': random.uniform(1, 3),
            'alpha': random.uniform(0.3, 0.9),
            'trail_alpha': random.uniform(0.1, 0.3)
        }
        
        # 初始化轨迹
        for _ in range(self.trail_length):
            particle['trail'].append([x0, y0, z0])
        
        self.particles.append(particle)
    
    def update_particles(self, volume, centroid):
        """更新粒子位置 - 使用吸引子动力学"""
        dt = 0.01 + 0.05 * volume  # 时间步长随音量变化
        
        for particle in self.particles:
            x, y, z = particle['position']
            
            # 使用吸引子方程计算导数
            derivatives = self.attractor_equations(0, [x, y, z])
            
            # 欧拉方法更新位置
            speed_factor = particle['speed'] * (0.5 + 0.5 * volume)
            x_new = x + derivatives[0] * dt * speed_factor
            y_new = y + derivatives[1] * dt * speed_factor
            z_new = z + derivatives[2] * dt * speed_factor
            
            # 更新位置
            particle['position'] = [x_new, y_new, z_new]
            
            # 更新轨迹
            particle['trail'].append([x_new, y_new, z_new])
            if len(particle['trail']) > self.trail_length:
                particle['trail'].pop(0)
            
            # 边界检查 - 如果粒子跑太远，重置
            max_bound = 10
            if (abs(x_new) > max_bound or abs(y_new) > max_bound or abs(z_new) > max_bound):
                self.reset_particle(particle)
    
    def reset_particle(self, particle):
        """重置粒子位置"""
        particle['position'] = [
            random.uniform(-2, 2),
            random.uniform(-2, 2), 
            random.uniform(-2, 2)
        ]
        particle['trail'] = [particle['position'].copy() for _ in range(self.trail_length)]
    
    def get_neon_color(self, index, volume, centroid):
        """获取霓虹颜色"""
        base_color = self.neon_colors[index % len(self.neon_colors)]
        
        # 动态颜色调整
        brightness_boost = 0.5 + 0.5 * volume
        hue_shift = math.sin(self.color_shift + index * 0.3) * 0.1
        
        hsv = mcolors.rgb_to_hsv(base_color)
        hsv[0] = (hsv[0] + hue_shift) % 1.0
        hsv[1] = min(1.0, hsv[1] * (0.8 + 0.2 * centroid))
        hsv[2] = min(1.0, hsv[2] * brightness_boost)
        
        return mcolors.hsv_to_rgb(hsv)
    
    def draw_attractor_kaleidoscope(self, volume, centroid, bass, mid, treble):
        """绘制吸引子万花筒"""
        # 清除图形
        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ax.set_facecolor('black')
        
        # 检测是否静音
        self.is_silent = volume < self.SILENCE_THRESHOLD
        
        if not self.is_silent:
            # 有声音时更新动画参数
            self.pulse_phase += 0.1 + 0.4 * volume
            self.morph_phase += 0.05 + 0.3 * centroid
            self.color_shift += 0.02 + 0.1 * volume
            
            # 更新粒子
            self.update_particles(volume, centroid)
        
        # 绘制对称粒子系统
        self.draw_symmetric_particles(volume, centroid)
        
        # 绘制中心吸引子
        self.draw_central_attractor(volume, centroid)
    
    def draw_symmetric_particles(self, volume, centroid):
        """绘制对称粒子系统"""
        for particle in self.particles:
            base_color = particle['color']
            trail_color = self.get_neon_color(
                self.particles.index(particle) % len(self.neon_colors), 
                volume, centroid
            )
            
            # 为每个对称角度绘制
            for i in range(self.symmetry):
                angle = 2 * math.pi * i / self.symmetry
                
                if not self.is_silent:
                    # 动态旋转
                    dynamic_angle = angle + self.rotation_angle * (0.5 + 0.5 * volume)
                else:
                    # 静态
                    dynamic_angle = angle
                
                cos_a, sin_a = math.cos(dynamic_angle), math.sin(dynamic_angle)
                
                # 变换粒子位置
                x, y, z = particle['position']
                x_rot = x * cos_a - y * sin_a
                y_rot = x * sin_a + y * cos_a
                
                # 绘制粒子
                alpha = particle['alpha'] * volume if not self.is_silent else particle['alpha'] * 0.3
                size = particle['size'] * (1 + volume)
                
                self.ax.scatter([x_rot], [y_rot], c=[base_color], s=size, alpha=alpha)
                
                # 绘制轨迹
                if len(particle['trail']) > 1:
                    trail_x = []
                    trail_y = []
                    
                    for point in particle['trail']:
                        tx, ty, tz = point
                        tx_rot = tx * cos_a - ty * sin_a
                        ty_rot = tx * sin_a + ty * cos_a
                        trail_x.append(tx_rot)
                        trail_y.append(ty_rot)
                    
                    # 轨迹透明度渐变
                    for j in range(len(trail_x) - 1):
                        trail_alpha = particle['trail_alpha'] * (j / len(trail_x)) * volume
                        if trail_alpha > 0.01:  # 只绘制可见的轨迹
                            self.ax.plot(
                                trail_x[j:j+2], trail_y[j:j+2], 
                                color=trail_color, alpha=trail_alpha, 
                                linewidth=0.5 + volume
                            )
        
        # 更新旋转角度
        if not self.is_silent:
            self.rotation_angle += 0.01 + 0.05 * volume
    
    def draw_central_attractor(self, volume, centroid):
        """绘制中心吸引子结构"""
        # 创建吸引子轨迹
        num_points = 100
        t_span = (0, 10 + 20 * volume)
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        
        # 初始条件
        y0 = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
        
        try:
            # 数值求解吸引子方程
            sol = solve_ivp(self.attractor_equations, t_span, y0, t_eval=t_eval, method='RK45')
            
            if sol.success:
                x, y, z = sol.y
                
                # 为每个对称角度绘制
                for i in range(self.symmetry):
                    angle = 2 * math.pi * i / self.symmetry
                    cos_a, sin_a = math.cos(angle), math.sin(angle)
                    
                    x_rot = x * cos_a - y * sin_a
                    y_rot = x * sin_a + y * cos_a
                    
                    # 选择颜色
                    color = self.get_neon_color(i, volume, centroid)
                    
                    # 绘制吸引子轨迹
                    alpha = 0.1 + 0.2 * volume
                    linewidth = 0.5 + 1.5 * volume
                    
                    self.ax.plot(x_rot, y_rot, color=color, alpha=alpha, linewidth=linewidth)
                    
                    # 在轨迹上添加亮点
                    if len(x_rot) > 0 and volume > 0.1:
                        highlight_idx = int(len(x_rot) * 0.7)
                        self.ax.scatter(
                            [x_rot[highlight_idx]], [y_rot[highlight_idx]], 
                            color=color, s=50 * volume, alpha=0.8
                        )
        except:
            pass  # 忽略数值积分错误
    
    def update_frame(self):
        """更新动画帧 - 优化性能"""
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        
        # 严格控制帧率
        if elapsed < 1.0 / self.target_fps:
            return True
            
        self.last_update_time = current_time
        self.frame_count += 1
        
        try:
            features = self.get_audio_features()
            volume = features['volume']
            centroid = features['centroid']
            bass = features['bass']
            mid = features['mid']
            treble = features['treble']
            
            # 绘制吸引子万花筒
            self.draw_attractor_kaleidoscope(volume, centroid, bass, mid, treble)
            
            # 使用blit技术提高性能
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            return True
            
        except Exception as e:
            print(f"更新错误: {e}")
            return False
    
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
        print("启动吸引子万花筒 - 流畅分形版")
        print("请确保麦克风已连接并授权使用")
        print("按Ctrl+C或关闭窗口退出")
        
        try:
            while self.running:
                if not self.update_frame():
                    # 精确控制帧率
                    time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("正在关闭...")
        except Exception as e:
            print(f"应用程序错误: {e}")
        finally:
            self.on_close(None)

# 主程序
if __name__ == "__main__":
    app = AttractorKaleidoscope()
    app.run()