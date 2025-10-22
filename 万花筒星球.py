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

class RealTime3DKaleidoscope:
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
        
        # 创建3D图形
        plt.ion()
        self.fig = plt.figure(figsize=(12, 12), dpi=100, facecolor='black')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.manager.set_window_title('3D实时音频万花筒 - 修复版')
        
        # 设置3D视图
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_zlim(-2, 2)
        self.ax.axis('off')
        
        # 设置黑色背景
        self.ax.set_facecolor('black')
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        
        # 初始化变量
        self.symmetry = 8
        self.rotation_angle = 0
        
        # 音频特征
        self.audio_features = {
            'volume': 0,
            'centroid': 0.5,
            'bass': 0.5,
            'mid': 0.5,
            'treble': 0.5
        }
        
        # 粒子系统
        self.particles = []
        self.max_particles = 200
        
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
        
        print("3D实时音频万花筒初始化完成")
    
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
                
                volume = 0.3 + 0.25 * math.sin(current_time * 3)
                centroid = 0.4 + 0.3 * math.sin(current_time * 1.7)
                bass = 0.4 + 0.3 * math.sin(current_time * 0.9)
                mid = 0.5 + 0.3 * math.sin(current_time * 1.3)
                treble = 0.6 + 0.25 * math.sin(current_time * 2.5)
                
                if random.random() < 0.1:
                    volume = min(1.0, volume + 0.3)
                
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
                    
                    gain = 2.0
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
        compressed_volume = 1.0 - np.exp(-5.0 * volume)
        
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
            bass_norm = (bass / max_val) ** 0.7
            mid_norm = (mid / max_val) ** 0.7
            treble_norm = (treble / max_val) ** 0.7
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
    
    def create_particle(self):
        """创建新粒子"""
        theta = random.uniform(0, 2 * math.pi)
        phi = math.acos(2 * random.random() - 1)
        r = random.uniform(0.5, 1.5)
        
        x = r * math.sin(phi) * math.cos(theta)
        y = r * math.sin(phi) * math.sin(theta)
        z = r * math.cos(phi)
        
        vx = random.uniform(-0.01, 0.01)
        vy = random.uniform(-0.01, 0.01)
        vz = random.uniform(-0.01, 0.01)
        
        hue = random.uniform(0, 1)
        color = mcolors.hsv_to_rgb([hue, 0.8, 1.0])
        size = random.uniform(10, 30)
        
        particle = {
            'x': x, 'y': y, 'z': z,
            'vx': vx, 'vy': vy, 'vz': vz,
            'color': color,
            'size': size,
            'glow_size': size * 1.5,
            'alpha': random.uniform(0.3, 0.8)
        }
        
        return particle
    
    def update_particles(self, volume):
        """更新粒子状态"""
        # 添加新粒子
        if len(self.particles) < self.max_particles and random.random() < volume:
            self.particles.append(self.create_particle())
        
        # 更新现有粒子
        particles_to_remove = []
        
        for i, p in enumerate(self.particles):
            p['x'] += p['vx']
            p['y'] += p['vy'] 
            p['z'] += p['vz']
            
            # 音频影响
            p['vx'] += (random.random() - 0.5) * 0.005 * volume
            p['vy'] += (random.random() - 0.5) * 0.005 * volume
            p['vz'] += (random.random() - 0.5) * 0.005 * volume
            
            # 边界检查
            max_bound = 2.0
            if abs(p['x']) > max_bound or abs(p['y']) > max_bound or abs(p['z']) > max_bound:
                particles_to_remove.append(i)
        
        # 移除超出边界的粒子
        for i in sorted(particles_to_remove, reverse=True):
            self.particles.pop(i)
    
    def generate_color_palette(self, centroid, volume):
        """生成颜色调色板"""
        self.color_shift += 0.01
        
        if centroid > 0.6:
            base_hues = [0.7, 0.75, 0.8]
        elif centroid < 0.4:
            base_hues = [0.0, 0.05, 0.1]
        else:
            base_hues = [0.3, 0.35, 0.4]
        
        base_hues = [(hue + self.color_shift) % 1.0 for hue in base_hues]
        
        saturation = 0.7 + 0.3 * volume
        brightness = 0.6 + 0.4 * volume
        
        colors = []
        for hue in base_hues:
            rgb = mcolors.hsv_to_rgb([hue, saturation, brightness])
            colors.append(rgb)
        
        return colors
    
    def draw_kaleidoscope(self, volume, centroid, bass, mid, treble):
        """绘制万花筒"""
        # 清除图形
        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_zlim(-2, 2)
        
        # 设置视角
        self.ax.view_init(elev=20, azim=self.rotation_angle * 30)
        
        # 生成颜色
        colors = self.generate_color_palette(centroid, volume)
        
        # 绘制基础几何结构
        self.draw_geometric_structures(volume, centroid, colors)
        
        # 绘制粒子
        self.draw_particles()
        
        # 更新旋转
        self.rotation_angle += 0.5 + 1.5 * volume
    
    def draw_geometric_structures(self, volume, centroid, colors):
        """绘制几何结构"""
        # 创建球面
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        
        U, V = np.meshgrid(u, v)
        
        for i in range(2):
            radius = 0.8 + 0.4 * i
            
            # 变形参数
            deformation = 0.2 * volume * np.sin(3 * U) + 0.1 * centroid * np.cos(2 * V)
            R = radius * (1 + deformation)
            
            X = R * np.sin(V) * np.cos(U)
            Y = R * np.sin(V) * np.sin(U)
            Z = R * np.cos(V)
            
            # 旋转
            angle = self.rotation_angle * (1 + 0.5 * i)
            X_rot = X * np.cos(angle) - Y * np.sin(angle)
            Y_rot = X * np.sin(angle) + Y * np.cos(angle)
            Z_rot = Z
            
            color = colors[i % len(colors)]
            alpha = 0.1 + 0.15 * volume
            
            self.ax.plot_surface(X_rot, Y_rot, Z_rot, color=color, alpha=alpha, 
                               rstride=2, cstride=2, linewidth=0)
        
        # 绘制环形
        self.draw_torus(volume, colors)
    
    def draw_torus(self, volume, colors):
        """绘制环形"""
        theta = np.linspace(0, 2 * np.pi, 40)
        phi = np.linspace(0, 2 * np.pi, 20)
        
        Theta, Phi = np.meshgrid(theta, phi)
        
        R = 1.2
        r = 0.3 + 0.2 * volume
        
        X = (R + r * np.cos(Theta)) * np.cos(Phi)
        Y = (R + r * np.cos(Theta)) * np.sin(Phi)
        Z = r * np.sin(Theta)
        
        # 旋转
        angle = self.rotation_angle * 0.7
        X_rot = X * np.cos(angle) - Z * np.sin(angle)
        Y_rot = Y
        Z_rot = X * np.sin(angle) + Z * np.cos(angle)
        
        color = colors[2 % len(colors)]
        alpha = 0.15 + 0.1 * volume
        
        self.ax.plot_surface(X_rot, Y_rot, Z_rot, color=color, alpha=alpha,
                           rstride=2, cstride=2, linewidth=0)
    
    def draw_particles(self):
        """绘制粒子 - 修复颜色匹配问题"""
        if not self.particles:
            return
        
        # 准备粒子数据
        x_coords = [p['x'] for p in self.particles]
        y_coords = [p['y'] for p in self.particles]
        z_coords = [p['z'] for p in self.particles]
        colors = [p['color'] for p in self.particles]
        sizes = [p['size'] for p in self.particles]
        glow_sizes = [p['glow_size'] for p in self.particles]
        alphas = [p['alpha'] for p in self.particles]
        
        # 修复：确保颜色和alpha都是标量或相同长度的列表
        # 对于发光层，使用统一的低alpha值
        glow_alpha = 0.2
        
        # 绘制发光层
        self.ax.scatter(x_coords, y_coords, z_coords, 
                       s=glow_sizes, c=colors, alpha=glow_alpha,
                       marker='o', edgecolors='none')
        
        # 绘制核心粒子
        self.ax.scatter(x_coords, y_coords, z_coords,
                       s=sizes, c=colors, alpha=alphas,
                       marker='o', edgecolors='white', linewidths=0.3)
    
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
            
            # 更新动画相位
            self.pulse_phase += 0.1 + 0.3 * volume
            self.morph_phase += 0.05 + 0.2 * centroid
            
            # 更新粒子
            self.update_particles(volume)
            
            # 绘制万花筒
            self.draw_kaleidoscope(volume, centroid, bass, mid, treble)
            
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
        print("启动3D实时音频万花筒 - 修复版")
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
    app = RealTime3DKaleidoscope()
    app.run()