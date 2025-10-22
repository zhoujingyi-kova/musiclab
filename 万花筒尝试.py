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

class FractalNeonKaleidoscope:
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
        
        # 创建图形
        plt.ion()
        self.fig = plt.figure(figsize=(12, 12), dpi=100, facecolor='black')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.manager.set_window_title('分形霓虹万花筒 - 荧光科技风')
        
        # 设置3D视图
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_zlim(-2, 2)
        self.ax.axis('off')
        
        # 设置纯黑背景
        self.ax.set_facecolor('black')
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        
        # 初始化变量
        self.symmetry = 12  # 高度对称
        self.rotation_angle = 0
        self.last_volume = 0
        self.is_silent = True
        
        # 音频特征
        self.audio_features = {
            'volume': 0,
            'centroid': 0.5,
            'bass': 0.5,
            'mid': 0.5,
            'treble': 0.5
        }
        
        # 分形参数
        self.fractal_iterations = 5
        self.spiral_density = 8
        
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
        self.target_fps = 30
        
        # 线程控制
        self.running = True
        self.audio_thread = None
        
        # 启动音频捕获
        self.start_audio_capture()
        
        # 设置关闭事件处理
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        print("分形霓虹万花筒初始化完成")
    
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
                base_freq = 0.5
                volume = 0.2 + 0.3 * abs(math.sin(current_time * base_freq))
                volume += 0.1 * math.sin(current_time * base_freq * 3)
                
                centroid = 0.4 + 0.3 * math.sin(current_time * 0.7)
                bass = 0.3 + 0.4 * abs(math.sin(current_time * base_freq))
                mid = 0.4 + 0.3 * math.sin(current_time * base_freq * 2)
                treble = 0.5 + 0.3 * math.sin(current_time * base_freq * 4)
                
                # 随机脉冲
                if random.random() < 0.05:
                    volume = min(1.0, volume + 0.4)
                
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
                
                time.sleep(0.03)
        
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
                    
                    gain = 2.5  # 提高增益增强灵敏度
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
        compressed_volume = 1.0 - np.exp(-8.0 * volume)  # 更强的压缩
        
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
            bass_norm = (bass / max_val) ** 0.6  # 更强的非线性增强
            mid_norm = (mid / max_val) ** 0.6
            treble_norm = (treble / max_val) ** 0.6
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
    
    def get_neon_color(self, index, volume, centroid):
        """获取霓虹颜色，带发光效果"""
        base_color = self.neon_colors[index % len(self.neon_colors)]
        
        # 根据音频特征调整颜色亮度
        brightness_boost = 0.3 + 0.7 * volume
        saturation_boost = 0.7 + 0.3 * centroid
        
        # 应用颜色偏移创造动态效果
        hue_shift = math.sin(self.color_shift + index * 0.5) * 0.1
        
        # 转换为HSV进行调整
        hsv = mcolors.rgb_to_hsv(base_color)
        hsv[0] = (hsv[0] + hue_shift) % 1.0
        hsv[1] = min(1.0, hsv[1] * saturation_boost)
        hsv[2] = min(1.0, hsv[2] * brightness_boost)
        
        return mcolors.hsv_to_rgb(hsv)
    
    def fractal_function(self, r, theta, phi, iteration, volume, centroid):
        """分形函数 - 创建复杂的分形纹理"""
        # 基础分形模式
        fractal = 0
        
        for i in range(1, self.fractal_iterations + 1):
            freq = 2 ** i
            amp = 0.5 ** i
            
            # 添加多个频率的分形细节
            fractal += amp * math.sin(freq * r + self.morph_phase * i * 0.3)
            fractal += amp * math.sin(freq * theta * 3 + self.morph_phase * i * 0.5)
            fractal += amp * math.sin(freq * phi * 2 + self.morph_phase * i * 0.7)
        
        # 音频影响分形强度
        fractal_strength = 0.1 + 0.3 * volume + 0.2 * centroid
        
        return 1.0 + fractal * fractal_strength
    
    def spiral_function(self, theta, r, volume):
        """涡旋曲线函数"""
        spiral_density = self.spiral_density + int(4 * volume)
        return math.sin(spiral_density * theta + 5 * r + self.pulse_phase)
    
    def draw_fractal_mandala(self, volume, centroid, bass, mid, treble):
        """绘制分形曼陀罗"""
        # 清除图形
        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_zlim(-2, 2)
        
        # 设置视角
        elevation = 15 + 10 * math.sin(self.rotation_angle * 0.1)
        self.ax.view_init(elev=elevation, azim=self.rotation_angle * 20)
        
        # 检测是否静音
        self.is_silent = volume < self.SILENCE_THRESHOLD
        
        if not self.is_silent:
            # 有声音时更新动画参数
            self.pulse_phase += 0.2 + 0.8 * volume
            self.morph_phase += 0.1 + 0.4 * centroid
            self.color_shift += 0.03 + 0.1 * volume
            
            # 更新旋转角度
            rotation_speed = 0.3 + 1.5 * volume + 0.5 * treble
            self.rotation_angle += rotation_speed
        
        # 绘制分形球体
        self.draw_fractal_sphere(volume, centroid, bass, mid, treble)
        
        # 绘制涡旋曼陀罗
        self.draw_spiral_mandala(volume, centroid)
        
        # 绘制发光粒子
        self.draw_glowing_particles(volume, centroid)
    
    def draw_fractal_sphere(self, volume, centroid, bass, mid, treble):
        """绘制分形球体"""
        u_resolution = 60
        v_resolution = 30
        
        u = np.linspace(0, 2 * np.pi, u_resolution)
        v = np.linspace(0, np.pi, v_resolution)
        
        U, V = np.meshgrid(u, v)
        
        # 基础球体半径
        base_radius = 0.8 + 0.4 * volume
        
        for layer in range(3):
            layer_factor = layer / 3
            radius = base_radius * (0.7 + 0.3 * layer_factor)
            
            # 应用分形变形
            fractal_deformation = np.zeros_like(U)
            for i in range(U.shape[0]):
                for j in range(U.shape[1]):
                    fractal_deformation[i, j] = self.fractal_function(
                        radius, U[i, j], V[i, j], layer, volume, centroid
                    )
            
            # 应用涡旋变形
            spiral_deformation = 0.1 * volume * np.sin(5 * U + self.pulse_phase)
            
            # 综合变形
            R = radius * (fractal_deformation + spiral_deformation)
            
            # 球坐标转直角坐标
            X = R * np.sin(V) * np.cos(U)
            Y = R * np.sin(V) * np.sin(U)
            Z = R * np.cos(V)
            
            # 应用旋转
            X_rot, Y_rot, Z_rot = self.rotate_3d(X, Y, Z)
            
            # 选择霓虹颜色
            color = self.get_neon_color(layer + 2, volume, centroid)
            
            # 设置透明度 - 多层叠加创造发光效果
            alpha = 0.08 + 0.12 * volume
            
            # 绘制分形表面
            self.ax.plot_surface(X_rot, Y_rot, Z_rot, color=color, alpha=alpha,
                               rstride=1, cstride=1, linewidth=0.5, 
                               edgecolor=color, antialiased=True)
    
    def draw_spiral_mandala(self, volume, centroid):
        """绘制涡旋曼陀罗"""
        radial_points = 100
        angular_symmetry = self.symmetry
        
        r = np.linspace(0.2, 1.5, radial_points)
        
        for sym in range(angular_symmetry):
            angle_offset = 2 * np.pi * sym / angular_symmetry
            
            # 创建涡旋曲线
            spiral_r = []
            spiral_theta = []
            
            for radius in r:
                # 涡旋角度
                theta = angle_offset + 5 * radius + self.pulse_phase * 0.5
                
                # 涡旋变形
                spiral_def = self.spiral_function(theta, radius, volume)
                final_theta = theta + 0.3 * spiral_def * volume
                final_r = radius * (1 + 0.2 * spiral_def * volume)
                
                spiral_theta.append(final_theta)
                spiral_r.append(final_r)
            
            # 转换为3D坐标 (在XY平面)
            x = [r_val * math.cos(theta_val) for r_val, theta_val in zip(spiral_r, spiral_theta)]
            y = [r_val * math.sin(theta_val) for r_val, theta_val in zip(spiral_r, spiral_theta)]
            z = [0.1 * math.sin(3 * theta_val + self.morph_phase) for theta_val in spiral_theta]
            
            # 应用3D旋转
            x_rot, y_rot, z_rot = self.rotate_3d(np.array(x), np.array(y), np.array(z))
            
            # 选择颜色
            color = self.get_neon_color(sym, volume, centroid)
            
            # 绘制涡旋线
            line_width = 1 + 3 * volume
            self.ax.plot(x_rot, y_rot, z_rot, color=color, linewidth=line_width, alpha=0.8)
            
            # 在涡旋末端添加发光点
            if len(x_rot) > 0 and volume > 0.1:
                end_point_size = 20 + 80 * volume
                self.ax.scatter([x_rot[-1]], [y_rot[-1]], [z_rot[-1]], 
                              color=color, s=end_point_size, alpha=0.9)
    
    def draw_glowing_particles(self, volume, centroid):
        """绘制发光粒子"""
        if volume < 0.1:
            return
            
        num_particles = int(20 + 30 * volume)
        
        for i in range(num_particles):
            # 在球面上随机分布
            theta = random.uniform(0, 2 * np.pi)
            phi = math.acos(2 * random.random() - 1)
            r = random.uniform(0.5, 1.8)
            
            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.sin(phi) * math.sin(theta)
            z = r * math.cos(phi)
            
            # 应用旋转
            x_rot, y_rot, z_rot = self.rotate_3d_single(x, y, z)
            
            # 选择颜色
            color = self.get_neon_color(i, volume, centroid)
            
            # 粒子大小和透明度
            size = random.uniform(10, 50) * volume
            alpha = random.uniform(0.3, 0.9) * volume
            
            # 绘制发光粒子
            self.ax.scatter([x_rot], [y_rot], [z_rot], color=color, s=size, alpha=alpha)
    
    def rotate_3d(self, x, y, z):
        """应用3D旋转"""
        if self.is_silent:
            # 静音时不旋转
            return x, y, z
            
        # 绕X轴旋转
        y1 = y * np.cos(self.rotation_angle * 0.7) - z * np.sin(self.rotation_angle * 0.7)
        z1 = y * np.sin(self.rotation_angle * 0.7) + z * np.cos(self.rotation_angle * 0.7)
        
        # 绕Y轴旋转
        x2 = x * np.cos(self.rotation_angle) + z1 * np.sin(self.rotation_angle)
        z2 = -x * np.sin(self.rotation_angle) + z1 * np.cos(self.rotation_angle)
        
        # 绕Z轴旋转
        x3 = x2 * np.cos(self.rotation_angle * 0.5) - y1 * np.sin(self.rotation_angle * 0.5)
        y3 = x2 * np.sin(self.rotation_angle * 0.5) + y1 * np.cos(self.rotation_angle * 0.5)
        
        return x3, y3, z2
    
    def rotate_3d_single(self, x, y, z):
        """应用3D旋转（单点）"""
        if self.is_silent:
            return x, y, z
            
        # 绕X轴旋转
        y1 = y * math.cos(self.rotation_angle * 0.7) - z * math.sin(self.rotation_angle * 0.7)
        z1 = y * math.sin(self.rotation_angle * 0.7) + z * math.cos(self.rotation_angle * 0.7)
        
        # 绕Y轴旋转
        x2 = x * math.cos(self.rotation_angle) + z1 * math.sin(self.rotation_angle)
        z2 = -x * math.sin(self.rotation_angle) + z1 * math.cos(self.rotation_angle)
        
        # 绕Z轴旋转
        x3 = x2 * math.cos(self.rotation_angle * 0.5) - y1 * math.sin(self.rotation_angle * 0.5)
        y3 = x2 * math.sin(self.rotation_angle * 0.5) + y1 * math.cos(self.rotation_angle * 0.5)
        
        return x3, y3, z2
    
    def update_frame(self):
        """更新动画帧"""
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        
        if elapsed < 1.0 / self.target_fps:
            return True
            
        self.last_update_time = current_time
        
        try:
            features = self.get_audio_features()
            volume = features['volume']
            centroid = features['centroid']
            bass = features['bass']
            mid = features['mid']
            treble = features['treble']
            
            # 绘制分形曼陀罗
            self.draw_fractal_mandala(volume, centroid, bass, mid, treble)
            
            # 刷新图形
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
            return True
            
        except Exception as e:
            print(f"更新错误: {e}")
            import traceback
            traceback.print_exc()
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
        print("启动分形霓虹万花筒 - 荧光科技风")
        print("请确保麦克风已连接并授权使用")
        print("按Ctrl+C或关闭窗口退出")
        
        try:
            while self.running:
                if not self.update_frame():
                    time.sleep(0.001)
                
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
    app = FractalNeonKaleidoscope()
    app.run()