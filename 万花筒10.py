"""
音频驱动的3D万花筒可视化系统 + 粒子效果 + 环形噪波
使用 PyAudio 实现音频输入
依赖：numpy, matplotlib, pyaudio
安装：pip install numpy matplotlib pyaudio
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import pyaudio
import threading
from collections import deque
import time
import random
import math

class Particle:
    """粒子类，基于提供的代码逻辑"""
    def __init__(self, canvas_width=4, canvas_height=4):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.init()
        
    def init(self):
        """初始化粒子参数"""
        self.x = random.uniform(-self.canvas_width/2, self.canvas_width/2)
        self.y = random.uniform(-self.canvas_height/2, -self.canvas_height/4)
        self.z = random.uniform(-2, 2)
        
        self.d_max = random.uniform(0.1, 0.5)
        self.d = 0
        
        self.t = 0
        self.t1 = random.randint(30, 90)
        
        self.y_step = random.uniform(0.01, 0.05)
        self.x_step = random.uniform(-0.02, 0.02)
        self.z_step = random.uniform(-0.01, 0.01)
        
        base_hue = random.uniform(0, 1)
        self.base_color = self.hsv_to_rgb(base_hue, 0.8, 1.0)
        self.col = self.base_color
        
        self.audio_energy_scale = 1.0
        self.audio_freq_hue_shift = 0.0
        
    def hsv_to_rgb(self, h, s, v):
        """HSV转RGB颜色"""
        if s == 0.0:
            return (v, v, v)
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        if i == 0:
            return (v, t, p)
        if i == 1:
            return (q, v, p)
        if i == 2:
            return (p, v, t)
        if i == 3:
            return (p, q, v)
        if i == 4:
            return (t, p, v)
        if i == 5:
            return (v, p, q)
    
    def update_color(self, energy, freq_hue, low_freq_energy, mid_freq_energy, high_freq_energy):
        """根据音频特征更新颜色"""
        brightness = min(1.0, 0.3 + energy * 0.7)
        hue_shift = (self.audio_freq_hue_shift + freq_hue) % 1.0
        
        # 根据频率能量调整饱和度
        # 重低音使用高饱和度，非重低音使用低饱和度
        saturation = 0.3 + low_freq_energy * 0.7  # 重低音饱和度高
        saturation += mid_freq_energy * 0.2  # 中频饱和度中等
        saturation += high_freq_energy * 0.1  # 高频饱和度低
        saturation = min(1.0, saturation)
        
        r, g, b = self.hsv_to_rgb(hue_shift, saturation, brightness)
        self.col = (r, g, b, 0.8)
    
    def move(self, audio_energy=0, dominant_freq=0):
        """更新粒子状态"""
        self.t += 1
        
        self.audio_energy_scale = 1.0 + audio_energy * 3
        self.audio_freq_hue_shift = (dominant_freq / 2000) % 1.0
        
        if 0 < self.t < self.t1:
            n = self.t / (self.t1 - 1)
            self.d = self.d_max * np.sin(n * np.pi) * self.audio_energy_scale
        
        if self.t > self.t1:
            self.init()
            return
        
        self.y += self.y_step * self.audio_energy_scale
        self.x += self.x_step
        self.z += self.z_step
        
        if (abs(self.x) > self.canvas_width or 
            self.y > self.canvas_height or 
            abs(self.z) > 3):
            self.init()

class NoiseRingSystem:
    """环形噪波系统，基于提供的代码逻辑"""
    def __init__(self):
        self.rings = []
        self.time = 0
        self.audio_energy = 0
        self.dominant_freq = 0
        self.low_freq_energy = 0
        self.mid_freq_energy = 0
        self.high_freq_energy = 0
        
    def update(self, audio_energy, dominant_freq, low_freq_energy, mid_freq_energy, high_freq_energy, frame):
        """更新噪波系统"""
        self.audio_energy = audio_energy
        self.dominant_freq = dominant_freq
        self.low_freq_energy = low_freq_energy
        self.mid_freq_energy = mid_freq_energy
        self.high_freq_energy = high_freq_energy
        self.time = frame * 0.01  # 时间累积
        
    def generate_ring_points(self, num_rings=12, points_per_ring=200):
        """生成环形噪波点"""
        points = []
        colors = []
        sizes = []
        
        # 主循环：360° 每 30° 一条环
        for ring_index, d in enumerate(range(360, 0, -30)):
            ring_radius = d / 180.0  # 标准化半径
            
            for i in range(points_per_ring):
                r = (i / points_per_ring) * 2 * math.pi
                
                # 噪波采样（与提供的参数保持一致）
                n = self.simplex_noise(d * 0.01, r * 0.02, -self.time * 0.01)
                
                # tangent 透明度因子
                T = math.tan(n * 9)
                if abs(T) < 0.001:  # 避免除0
                    continue
                    
                alpha = min(1.0, 1.0 / abs(T))
                
                # 色相基于角度和音频频率
                hue = ((r * 57) % 360) / 360.0 + (self.dominant_freq / 2000)
                
                # 根据频率能量调整饱和度
                # 重低音使用高饱和度，非重低音使用低饱和度
                saturation = 0.3 + self.low_freq_energy * 0.7  # 重低音饱和度高
                saturation += self.mid_freq_energy * 0.2  # 中频饱和度中等
                saturation += self.high_freq_energy * 0.1  # 高频饱和度低
                saturation = min(1.0, saturation)
                
                brightness = 0.8 + self.audio_energy * 0.2
                
                # HSV转RGB
                color = self.hsv_to_rgb(hue % 1.0, saturation, brightness)
                color = (*color, alpha)  # 添加透明度
                
                # 旋转角 A 随时间漂移
                A = r - self.time * 0.01 + math.radians(d)
                
                # 3D位置计算
                x = math.cos(A) * ring_radius
                y = math.sin(A) * ring_radius
                z = math.sin(self.time + ring_index) * 0.5  # 添加Z轴波动
                
                # 音频能量影响半径
                radius_scale = 1.0 + self.audio_energy * 2
                x *= radius_scale
                y *= radius_scale
                
                points.append((x, y, z))
                colors.append(color)
                sizes.append(2 + alpha * 8)  # 点的大小基于透明度
                
        return points, colors, sizes
    
    def simplex_noise(self, x, y, z):
        """简化的噪声函数（替代Perlin噪声）"""
        # 使用三角函数组合模拟噪声效果
        return (math.sin(x * 10 + self.time) * 0.3 +
                math.cos(y * 8 + self.time * 1.3) * 0.3 +
                math.sin(z * 12 + self.time * 0.7) * 0.4)
    
    def hsv_to_rgb(self, h, s, v):
        """HSV转RGB颜色"""
        if s == 0.0:
            return (v, v, v)
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        if i == 0:
            return (v, t, p)
        if i == 1:
            return (q, v, p)
        if i == 2:
            return (p, v, t)
        if i == 3:
            return (p, q, v)
        if i == 4:
            return (t, p, v)
        if i == 5:
            return (v, p, q)

class AudioVisualizer:
    def __init__(self):
        # 音频参数
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.audio_buffer = deque(maxlen=5)
        
        # PyAudio 实例
        self.audio = pyaudio.PyAudio()
        
        # 可视化参数 - 使用正方形窗口
        self.fig = plt.figure(figsize=(10, 10))  # 正方形窗口
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 音频特征
        self.energy = 0.0
        self.dominant_freq = 0.0
        self.low_freq_energy = 0.0
        self.mid_freq_energy = 0.0
        self.high_freq_energy = 0.0
        self.spectrum = np.zeros(self.chunk_size // 2)
        
        # 控制参数
        self.is_playing = True
        self.paused = False
        self.rotation_speed = 0.0
        self.scale_factor = 1.0
        self.color_shift = 0.0
        self.light_position = [1, 1, 1]
        
        # 粒子系统
        self.particles = []
        self.num_particles = 50
        self._init_particles()
        
        # 噪波环系统
        self.noise_rings = NoiseRingSystem()
        
        # 当前显示的模式
        self.current_mode = 'hybrid'
        
        self._setup_audio()
        self._setup_visualization()
    
    def _init_particles(self):
        """初始化粒子系统"""
        for _ in range(self.num_particles):
            self.particles.append(Particle())
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio 音频回调函数"""
        if status:
            print(f"Audio status: {status}")
        
        if self.is_playing and not self.paused:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_buffer.append(audio_data.copy())
        
        return (None, pyaudio.paContinue)
    
    def _setup_audio(self):
        """设置PyAudio输入流"""
        try:
            self.audio_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.audio_stream.start_stream()
            print("PyAudio音频输入已启动 - 正在监听麦克风...")
            
        except Exception as e:
            print(f"无法启动音频输入: {e}")
            print("请检查麦克风是否可用，或尝试更改音频设备")
    
    def _setup_visualization(self):
        """设置可视化参数"""
        # 设置坐标轴范围，使图形覆盖整个窗口
        self.ax.set_xlim([-3, 3])
        self.ax.set_ylim([-3, 3])
        self.ax.set_zlim([-3, 3])
        self.ax.set_facecolor('black')
        self.ax.grid(False)
        self.ax.axis('off')
        
        # 调整子图位置，使图形填满整个窗口
        self.ax.set_position([0, 0, 1, 1])
        
        # 连接键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        print("可视化已初始化 - 使用键盘控制:")
        print("空格键: 暂停/恢复")
        print("上下左右: 调整参数")
        print("1/2/3: 切换模式")
        print("+/-: 调整粒子数量")
    
    def _update_mode(self, mode_name):
        """更新当前显示的模式"""
        self.current_mode = mode_name
    
    def _analyze_audio(self):
        """分析音频数据，提取特征"""
        if not self.audio_buffer:
            return
        
        audio_data = np.concatenate(list(self.audio_buffer))
        
        if len(audio_data) < self.chunk_size:
            return
        
        # 应用汉宁窗
        window = np.hanning(len(audio_data))
        windowed_data = audio_data * window
        
        # FFT变换
        fft_data = np.fft.fft(windowed_data)
        freq_magnitudes = np.abs(fft_data[:len(fft_data)//2])
        freqs = np.fft.fftfreq(len(windowed_data), 1/self.sample_rate)[:len(freq_magnitudes)]
        
        # 计算能量（RMS）
        self.energy = np.sqrt(np.mean(audio_data**2))
        
        # 找到主频率
        if len(freq_magnitudes) > 0:
            dominant_idx = np.argmax(freq_magnitudes)
            self.dominant_freq = freqs[dominant_idx]
        
        # 计算不同频段的能量
        # 低频 (0-150Hz) - 重低音
        low_freq_mask = (freqs >= 0) & (freqs <= 150)
        self.low_freq_energy = np.mean(freq_magnitudes[low_freq_mask]) if np.any(low_freq_mask) else 0
        
        # 中频 (150-1000Hz)
        mid_freq_mask = (freqs > 150) & (freqs <= 1000)
        self.mid_freq_energy = np.mean(freq_magnitudes[mid_freq_mask]) if np.any(mid_freq_mask) else 0
        
        # 高频 (1000Hz以上)
        high_freq_mask = freqs > 1000
        self.high_freq_energy = np.mean(freq_magnitudes[high_freq_mask]) if np.any(high_freq_mask) else 0
        
        # 归一化频段能量
        total_freq_energy = self.low_freq_energy + self.mid_freq_energy + self.high_freq_energy
        if total_freq_energy > 0:
            self.low_freq_energy /= total_freq_energy
            self.mid_freq_energy /= total_freq_energy
            self.high_freq_energy /= total_freq_energy
        
        self.spectrum = freq_magnitudes
    
    def _calculate_visual_parameters(self):
        """根据音频特征计算可视化参数"""
        self.rotation_speed = self.energy * 50
        self.scale_factor = 1.0 + self.energy * 3
        self.color_shift = (self.dominant_freq / 1000) % 1.0
    
    def _update_particles(self):
        """更新所有粒子状态"""
        for particle in self.particles:
            particle.move(self.energy, self.dominant_freq)
            particle.update_color(self.energy, self.color_shift, 
                                self.low_freq_energy, self.mid_freq_energy, self.high_freq_energy)
    
    def _draw_particles(self):
        """绘制所有粒子"""
        if not self.particles:
            return []
        
        x_pos = [p.x for p in self.particles]
        y_pos = [p.y for p in self.particles]
        z_pos = [p.z for p in self.particles]
        sizes = [p.d * 100 for p in self.particles]
        colors = [p.col for p in self.particles]
        
        scatter = self.ax.scatter(x_pos, y_pos, z_pos, 
                                 s=sizes, c=colors, 
                                 alpha=0.7, marker='o',
                                 edgecolors='none', depthshade=False)
        return [scatter]
    
    def _draw_noise_rings(self, frame):
        """绘制噪波环"""
        # 更新噪波系统
        self.noise_rings.update(self.energy, self.dominant_freq, 
                               self.low_freq_energy, self.mid_freq_energy, self.high_freq_energy, frame)
        
        # 生成噪波点
        points, colors, sizes = self.noise_rings.generate_ring_points()
        
        if not points:
            return []
        
        # 分离坐标
        x_pos = [p[0] for p in points]
        y_pos = [p[1] for p in points]
        z_pos = [p[2] for p in points]
        
        # 绘制噪波点
        scatter = self.ax.scatter(x_pos, y_pos, z_pos, 
                                 s=sizes, c=colors, 
                                 alpha=0.8, marker='o',
                                 edgecolors='none', depthshade=True)
        return [scatter]
    
    def _on_key_press(self, event):
        """处理键盘事件"""
        if event.key == ' ':
            self.paused = not self.paused
            print(f"{'暂停' if self.paused else '恢复'}音频处理")
        
        elif event.key == 'up':
            self.light_position[1] += 0.2
        
        elif event.key == 'down':
            self.light_position[1] -= 0.2
        
        elif event.key == 'left':
            self.color_shift = (self.color_shift - 0.1) % 1.0
        
        elif event.key == 'right':
            self.color_shift = (self.color_shift + 0.1) % 1.0
        
        elif event.key == '1':
            self._update_mode('particles_only')
            print("切换到仅粒子模式")
        
        elif event.key == '2':
            self._update_mode('noise_rings')
            print("切换到噪波环模式")
        
        elif event.key == '3':
            self._update_mode('hybrid')
            print("切换到混合模式")
        
        elif event.key == '+':
            self.num_particles = min(200, self.num_particles + 10)
            while len(self.particles) < self.num_particles:
                self.particles.append(Particle())
            print(f"粒子数量: {self.num_particles}")
        
        elif event.key == '-':
            self.num_particles = max(10, self.num_particles - 10)
            self.particles = self.particles[:self.num_particles]
            print(f"粒子数量: {self.num_particles}")
    
    def update_visualization(self, frame):
        """更新可视化显示"""
        if self.paused:
            return []
        
        # 清除之前的图形
        self.ax.clear()
        
        # 设置坐标轴范围，使图形覆盖整个窗口
        self.ax.set_xlim([-3, 3])
        self.ax.set_ylim([-3, 3])
        self.ax.set_zlim([-3, 3])
        self.ax.set_facecolor('black')
        self.ax.grid(False)
        self.ax.axis('off')
        
        # 分析音频
        self._analyze_audio()
        self._calculate_visual_parameters()
        
        visual_elements = []
        
        # 根据当前模式绘制相应的可视化元素
        if self.current_mode == 'particles_only':
            self._update_particles()
            visual_elements.extend(self._draw_particles())
        
        elif self.current_mode == 'noise_rings':
            visual_elements.extend(self._draw_noise_rings(frame))
        
        elif self.current_mode == 'hybrid':
            # 混合模式：显示所有元素
            self._update_particles()
            visual_elements.extend(self._draw_particles())
            visual_elements.extend(self._draw_noise_rings(frame))
        
        # 设置标题显示音频特征
        self.ax.text2D(0.02, 0.98, 
                      f'音频可视化 | 能量: {self.energy:.3f} | 主频率: {self.dominant_freq:.1f} Hz\n'
                      f'低频: {self.low_freq_energy:.2f} | 中频: {self.mid_freq_energy:.2f} | 高频: {self.high_freq_energy:.2f}\n'
                      f'粒子数量: {self.num_particles} | 模式: {self.current_mode}',
                      transform=self.ax.transAxes, 
                      color='white', 
                      fontsize=10,
                      verticalalignment='top')
        
        return visual_elements
    
    def start_visualization(self):
        """启动可视化"""
        print("启动3D音频可视化+粒子系统+噪波环...")
        self.animation = FuncAnimation(
            self.fig, 
            self.update_visualization, 
            interval=50,
            blit=False,
            cache_frame_data=False
        )
        plt.show()
    
    def cleanup(self):
        """清理资源"""
        self.is_playing = False
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()
        print("资源已清理")

def main():
    """主函数"""
    try:
        visualizer = AudioVisualizer()
        visualizer.start_visualization()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        if 'visualizer' in locals():
            visualizer.cleanup()

if __name__ == "__main__":
    main()