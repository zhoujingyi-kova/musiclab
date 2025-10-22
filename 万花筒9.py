"""
音频驱动的3D万花筒可视化系统 + 粒子效果
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

class Particle:
    """粒子类，基于提供的代码逻辑"""
    def __init__(self, canvas_width=4, canvas_height=4):
        # 初始化参数
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.init()
        
    def init(self):
        """初始化粒子参数"""
        # 位置 - 在底部随机生成
        self.x = random.uniform(-self.canvas_width/2, self.canvas_width/2)
        self.y = random.uniform(-self.canvas_height/2, -self.canvas_height/4)
        self.z = random.uniform(-2, 2)  # 3D深度
        
        # 尺寸参数
        self.d_max = random.uniform(0.1, 0.5)  # 最大直径
        self.d = 0  # 当前直径
        
        # 时间参数
        self.t = 0  # 当前时间
        self.t1 = random.randint(30, 90)  # 生命周期
        
        # 运动参数
        self.y_step = random.uniform(0.01, 0.05)  # 上升速度
        self.x_step = random.uniform(-0.02, 0.02)  # 水平漂移
        self.z_step = random.uniform(-0.01, 0.01)  # 深度漂移
        
        # 颜色 - 基于音频会动态调整
        base_hue = random.uniform(0, 1)
        self.base_color = self.hsv_to_rgb(base_hue, 0.8, 1.0)
        self.col = self.base_color
        
        # 音频影响参数
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
    
    def update_color(self, energy, freq_hue):
        """根据音频特征更新颜色"""
        # 能量影响亮度
        brightness = min(1.0, 0.3 + energy * 0.7)
        
        # 频率影响色相偏移
        hue_shift = (self.audio_freq_hue_shift + freq_hue) % 1.0
        
        # 更新颜色
        r, g, b = self.hsv_to_rgb(hue_shift, 0.8, brightness)
        self.col = (r, g, b, 0.8)  # 带透明度
    
    def move(self, audio_energy=0, dominant_freq=0):
        """更新粒子状态"""
        self.t += 1  # 时间步进
        
        # 存储音频参数用于颜色更新
        self.audio_energy_scale = 1.0 + audio_energy * 3
        self.audio_freq_hue_shift = (dominant_freq / 2000) % 1.0
        
        # 在生命周期内：直径按正弦曲线变化
        if 0 < self.t < self.t1:
            n = self.t / (self.t1 - 1)  # 归一化到 0~1
            self.d = self.d_max * np.sin(n * np.pi) * self.audio_energy_scale
        
        # 生命周期结束后重新初始化
        if self.t > self.t1:
            self.init()
            return
        
        # 运动：上飘 + 随机漂移
        self.y += self.y_step * self.audio_energy_scale
        self.x += self.x_step
        self.z += self.z_step
        
        # 边界检查 - 如果飘出画布就重新初始化
        if (abs(self.x) > self.canvas_width or 
            self.y > self.canvas_height or 
            abs(self.z) > 3):
            self.init()

class AudioVisualizer:
    def __init__(self):
        # 音频参数
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.audio_buffer = deque(maxlen=5)
        
        # PyAudio 实例
        self.audio = pyaudio.PyAudio()
        
        # 可视化参数
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 音频特征
        self.energy = 0.0
        self.dominant_freq = 0.0
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
        self.num_particles = 50  # 粒子数量
        self._init_particles()
        
        # 几何形状数据
        self.shapes = {
            'sphere': self._create_sphere(1.0, 20),
            'cube': self._create_cube(1.0),
            'torus': self._create_torus(1.0, 0.3, 30),
            'particles_only': None  # 仅显示粒子模式
        }
        
        # 当前显示的形状
        self.current_shape = 'sphere'
        self.shape_vertices = None
        self.shape_faces = None
        
        self._setup_audio()
        self._setup_visualization()
    
    def _init_particles(self):
        """初始化粒子系统"""
        for _ in range(self.num_particles):
            self.particles.append(Particle())
    
    def _create_sphere(self, radius, resolution):
        """创建球体网格"""
        phi = np.linspace(0, np.pi, resolution)
        theta = np.linspace(0, 2*np.pi, resolution)
        phi, theta = np.meshgrid(phi, theta)
        
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        
        return x, y, z
    
    def _create_cube(self, size):
        """创建立方体网格"""
        vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ]) * size / 2
        
        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
            [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]
        ]
        
        return vertices, faces
    
    def _create_torus(self, major_radius, minor_radius, resolution):
        """创建圆环面网格"""
        theta = np.linspace(0, 2*np.pi, resolution)
        phi = np.linspace(0, 2*np.pi, resolution)
        theta, phi = np.meshgrid(theta, phi)
        
        x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
        y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
        z = minor_radius * np.sin(phi)
        
        return x, y, z
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio 音频回调函数"""
        if status:
            print(f"Audio status: {status}")
        
        if self.is_playing and not self.paused:
            # 将字节数据转换为numpy数组
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_buffer.append(audio_data.copy())
        
        # 返回空数据和继续标志
        return (None, pyaudio.paContinue)
    
    def _setup_audio(self):
        """设置PyAudio输入流"""
        try:
            # 创建音频流
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
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])
        self.ax.set_facecolor('black')
        self.ax.grid(False)
        self.ax.axis('off')
        
        self._update_shape('sphere')
        
        # 连接键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        print("可视化已初始化 - 使用键盘控制:")
        print("空格键: 暂停/恢复")
        print("上下左右: 调整参数")
        print("1/2/3/4: 切换形状（4=仅粒子）")
        print("+/-: 调整粒子数量")
    
    def _update_shape(self, shape_name):
        """更新当前显示的形状"""
        self.current_shape = shape_name
        # 实际绘制在update_visualization中处理
    
    def _analyze_audio(self):
        """分析音频数据，提取特征"""
        if not self.audio_buffer:
            return
        
        audio_data = np.concatenate(list(self.audio_buffer))
        
        # 如果音频数据太短，跳过分析
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
        
        self.spectrum = freq_magnitudes
    
    def _calculate_visual_parameters(self):
        """根据音频特征计算可视化参数"""
        # 旋转速度基于能量
        self.rotation_speed = self.energy * 50
        
        # 缩放基于低频能量
        low_freq_energy = np.mean(self.spectrum[:len(self.spectrum)//4]) if len(self.spectrum) > 0 else 0
        self.scale_factor = 1.0 + low_freq_energy * 10
        
        # 颜色偏移基于主频率
        self.color_shift = (self.dominant_freq / 1000) % 1.0
        
        # 根据能量切换形状
        if self.energy > 0.1:
            shape_index = int((self.dominant_freq / 500) % 4)
            shapes = ['sphere', 'cube', 'torus', 'particles_only']
            new_shape = shapes[shape_index]
            if new_shape != self.current_shape:
                self._update_shape(new_shape)
    
    def _update_particles(self):
        """更新所有粒子状态"""
        for particle in self.particles:
            particle.move(self.energy, self.dominant_freq)
            particle.update_color(self.energy, self.color_shift)
    
    def _draw_particles(self):
        """绘制所有粒子"""
        if not self.particles:
            return []
        
        # 收集所有粒子的位置和颜色
        x_pos = [p.x for p in self.particles]
        y_pos = [p.y for p in self.particles]
        z_pos = [p.z for p in self.particles]
        sizes = [p.d * 100 for p in self.particles]  # 缩放尺寸用于散点图
        colors = [p.col for p in self.particles]
        
        # 绘制粒子
        scatter = self.ax.scatter(x_pos, y_pos, z_pos, 
                                 s=sizes, c=colors, 
                                 alpha=0.7, marker='o',
                                 edgecolors='none', depthshade=False)
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
            self._update_shape('sphere')
            print("切换到球体")
        
        elif event.key == '2':
            self._update_shape('cube')
            print("切换到立方体")
        
        elif event.key == '3':
            self._update_shape('torus')
            print("切换到圆环面")
        
        elif event.key == '4':
            self._update_shape('particles_only')
            print("切换到仅粒子模式")
        
        elif event.key == '+':
            self.num_particles = min(200, self.num_particles + 10)
            while len(self.particles) < self.num_particles:
                self.particles.append(Particle())
            print(f"粒子数量: {self.num_particles}")
        
        elif event.key == '-':
            self.num_particles = max(10, self.num_particles - 10)
            self.particles = self.particles[:self.num_particles]
            print(f"粒子数量: {self.num_particles}")
    
    def _get_color_from_audio(self):
        """根据音频特征生成颜色"""
        r = (np.sin(self.color_shift * 2 * np.pi) + 1) / 2
        g = (np.sin(self.color_shift * 2 * np.pi + 2 * np.pi / 3) + 1) / 2
        b = (np.sin(self.color_shift * 2 * np.pi + 4 * np.pi / 3) + 1) / 2
        
        saturation = min(1.0, self.energy * 5)
        r = 0.5 + (r - 0.5) * saturation
        g = 0.5 + (g - 0.5) * saturation
        b = 0.5 + (b - 0.5) * saturation
        
        return (r, g, b, 0.8)
    
    def update_visualization(self, frame):
        """更新可视化显示"""
        if self.paused:
            return []
        
        # 清除之前的图形
        self.ax.clear()
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])
        self.ax.set_facecolor('black')
        self.ax.grid(False)
        self.ax.axis('off')
        
        # 分析音频
        self._analyze_audio()
        self._calculate_visual_parameters()
        
        # 更新粒子系统
        self._update_particles()
        
        visual_elements = []
        
        # 绘制粒子（始终显示）
        particle_elements = self._draw_particles()
        visual_elements.extend(particle_elements)
        
        # 如果不是仅粒子模式，绘制主要几何形状
        if self.current_shape != 'particles_only':
            color = self._get_color_from_audio()
            
            if self.current_shape in ['sphere', 'torus']:
                x, y, z = self.shapes[self.current_shape]
                
                # 应用缩放和旋转
                x_scaled = x * self.scale_factor
                y_scaled = y * self.scale_factor
                z_scaled = z * self.scale_factor
                
                angle = frame * self.rotation_speed * 0.01
                x_rot = x_scaled * np.cos(angle) - z_scaled * np.sin(angle)
                z_rot = x_scaled * np.sin(angle) + z_scaled * np.cos(angle)
                y_rot = y_scaled * np.cos(angle) - x_rot * np.sin(angle)
                x_rot = x_rot * np.cos(angle) + y_rot * np.sin(angle)
                
                # 绘制表面
                mesh = self.ax.plot_surface(
                    x_rot, y_rot, z_rot,
                    color=color,
                    alpha=0.6,
                    linewidth=0,
                    antialiased=True,
                    shade=True
                )
                visual_elements.append(mesh)
            
            else:  # cube
                vertices, faces = self.shapes['cube']
                
                # 应用缩放和旋转
                angle = frame * self.rotation_speed * 0.01
                scale_matrix = np.array([
                    [self.scale_factor, 0, 0],
                    [0, self.scale_factor, 0],
                    [0, 0, self.scale_factor]
                ])
                
                rotation_x = np.array([
                    [1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)]
                ])
                
                rotation_y = np.array([
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]
                ])
                
                vertices_transformed = vertices @ scale_matrix.T @ rotation_x.T @ rotation_y.T
                
                # 绘制立方体
                for face in faces:
                    face_vertices = vertices_transformed[face]
                    line = self.ax.plot3D(
                        face_vertices[:, 0], 
                        face_vertices[:, 1], 
                        face_vertices[:, 2], 
                        color=color,
                        linewidth=2
                    )
                    visual_elements.extend(line)
        
        # 设置标题显示音频特征
        self.ax.set_title(
            f'音频可视化 | 能量: {self.energy:.3f} | 主频率: {self.dominant_freq:.1f} Hz\n'
            f'粒子数量: {self.num_particles} | 模式: {self.current_shape}',
            color='white', 
            fontsize=10
        )
        
        return visual_elements
    
    def start_visualization(self):
        """启动可视化"""
        print("启动3D音频可视化+粒子系统...")
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