import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.colors as mcolors
import threading
import queue
import time
import math
import random

class ElegantAttractor:
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
        self.fig, self.ax = plt.subplots(figsize=(12, 12), dpi=100, facecolor='black')
        self.fig.canvas.manager.set_window_title('优雅吸引子 - 简约美学')
        
        # 设置视图
        self.ax.set_xlim(-2.5, 2.5)
        self.ax.set_ylim(-2.5, 2.5)
        self.ax.axis('off')
        self.ax.set_facecolor('black')
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        
        # 初始化变量
        self.symmetry = 6  # 适度的对称性
        self.rotation_angle = 0
        self.is_silent = True
        
        # 音频特征
        self.audio_features = {
            'volume': 0,
            'centroid': 0.5,
            'bass': 0.5
        }
        
        # 吸引子参数
        self.a = 10.0
        self.b = 28.0
        self.c = 8.0 / 3.0
        
        # 轨迹数据
        self.trajectory = []
        self.max_trajectory_length = 2000
        
        # 优雅的颜色调色板
        self.colors = [
            (0.8, 0.2, 0.8),  # 优雅紫色
            (0.2, 0.8, 0.8),  # 青蓝色
            (0.8, 0.8, 0.2),  # 金色
        ]
        
        # 性能优化
        self.last_update_time = time.time()
        self.target_fps = 60
        self.frame_count = 0
        
        # 线程控制
        self.running = True
        self.audio_thread = None
        
        # 初始化轨迹
        self.reset_trajectory()
        
        # 启动音频捕获
        self.start_audio_capture()
        
        # 设置关闭事件处理
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        print("优雅吸引子初始化完成")
    
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
                
                # 生成简洁的模拟音频
                base_freq = 0.5
                volume = 0.2 + 0.3 * abs(math.sin(current_time * base_freq))
                
                centroid = 0.4 + 0.3 * math.sin(current_time * 0.7)
                bass = 0.3 + 0.4 * abs(math.sin(current_time * base_freq))
                
                # 偶尔的脉冲
                if random.random() < 0.05:
                    volume = min(1.0, volume + 0.4)
                
                try:
                    self.audio_queue.put({
                        'volume': volume,
                        'centroid': centroid,
                        'bass': bass
                    }, block=False)
                except:
                    pass
                
                time.sleep(0.016)
        
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
        compressed_volume = 1.0 - np.exp(-8.0 * volume)
        
        fft_data = np.fft.fft(audio_data)
        freqs = np.abs(fft_data[:self.CHUNK//2])
        
        if len(freqs) > 0 and np.sum(freqs) > 0:
            frequencies = np.linspace(0, self.RATE/2, len(freqs))
            
            centroid = np.sum(frequencies * freqs) / np.sum(freqs)
            centroid_norm = min(centroid / (self.RATE/4), 1.0)
            
            bass_range = int(len(freqs) * 0.2)
            bass = np.mean(freqs[:bass_range]) if bass_range > 0 else 0
            
            max_val = max(bass, 1e-10)
            bass_norm = (bass / max_val) ** 0.7
        else:
            centroid_norm = 0.5
            bass_norm = 0.5
        
        try:
            self.audio_queue.put({
                'volume': compressed_volume,
                'centroid': centroid_norm,
                'bass': bass_norm
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
    
    def lorenz_attractor(self, t, state):
        """洛伦兹吸引子方程 - 经典且优雅"""
        x, y, z = state
        
        # 经典洛伦兹方程
        dxdt = self.a * (y - x)
        dydt = x * (self.b - z) - y
        dzdt = x * y - self.c * z
        
        return [dxdt, dydt, dzdt]
    
    def reset_trajectory(self):
        """重置轨迹"""
        self.trajectory = []
        # 随机初始位置
        self.current_state = [
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(0, 2)
        ]
    
    def update_trajectory(self, volume):
        """更新吸引子轨迹"""
        if self.is_silent and len(self.trajectory) > 100:
            # 静音时保持轨迹不变
            return
        
        # 时间步长随音量变化
        dt = 0.01 + 0.04 * volume
        
        # 使用欧拉方法简单积分
        derivatives = self.lorenz_attractor(0, self.current_state)
        
        self.current_state[0] += derivatives[0] * dt
        self.current_state[1] += derivatives[1] * dt
        self.current_state[2] += derivatives[2] * dt
        
        # 添加到轨迹
        self.trajectory.append(self.current_state.copy())
        
        # 限制轨迹长度
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory.pop(0)
    
    def get_elegant_color(self, volume, centroid):
        """获取优雅的颜色"""
        # 基于音频特征选择基础颜色
        if centroid > 0.6:
            base_color = self.colors[1]  # 青蓝色
        elif centroid < 0.4:
            base_color = self.colors[2]  # 金色
        else:
            base_color = self.colors[0]  # 紫色
        
        # 轻微的颜色变化
        hue_shift = math.sin(time.time() * 0.1) * 0.05
        hsv = mcolors.rgb_to_hsv(base_color)
        hsv[0] = (hsv[0] + hue_shift) % 1.0
        
        # 音量影响亮度
        brightness = 0.6 + 0.4 * volume
        hsv[2] = min(1.0, hsv[2] * brightness)
        
        return mcolors.hsv_to_rgb(hsv)
    
    def draw_elegant_attractor(self, volume, centroid, bass):
        """绘制优雅的吸引子"""
        # 清除图形
        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_xlim(-2.5, 2.5)
        self.ax.set_ylim(-2.5, 2.5)
        self.ax.set_facecolor('black')
        
        # 检测是否静音
        self.is_silent = volume < self.SILENCE_THRESHOLD
        
        if not self.is_silent:
            # 有声音时更新轨迹
            self.update_trajectory(volume)
            
            # 缓慢旋转
            self.rotation_angle += 0.002 + 0.01 * volume
        
        # 如果没有足够的轨迹点，跳过绘制
        if len(self.trajectory) < 10:
            return
        
        # 获取优雅的颜色
        color = self.get_elegant_color(volume, centroid)
        
        # 提取轨迹坐标
        x_coords = [point[0] for point in self.trajectory]
        y_coords = [point[1] for point in self.trajectory]
        z_coords = [point[2] for point in self.trajectory]
        
        # 创建对称图案
        for i in range(self.symmetry):
            angle = 2 * math.pi * i / self.symmetry + self.rotation_angle
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            
            # 旋转坐标
            x_rot = [x * cos_a - y * sin_a for x, y in zip(x_coords, y_coords)]
            y_rot = [x * sin_a + y * cos_a for x, y in zip(x_coords, y_coords)]
            
            # 计算透明度渐变 - 新点更亮
            alphas = []
            for j in range(len(self.trajectory)):
                # 线性渐变，新点alpha高，旧点alpha低
                alpha = 0.1 + 0.9 * (j / len(self.trajectory))
                alphas.append(alpha)
            
            # 绘制轨迹线
            for j in range(len(x_rot) - 1):
                if alphas[j] > 0.05:  # 只绘制可见的线段
                    line_width = 0.5 + 1.5 * volume
                    self.ax.plot(
                        x_rot[j:j+2], y_rot[j:j+2],
                        color=color,
                        alpha=alphas[j] * (0.3 + 0.7 * volume),
                        linewidth=line_width
                    )
        
        # 绘制中心点（当前状态）
        if not self.is_silent:
            center_size = 20 + 80 * volume
            self.ax.scatter(0, 0, color=color, s=center_size, alpha=0.8)
    
    def update_frame(self):
        """更新动画帧"""
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
            
            # 绘制优雅吸引子
            self.draw_elegant_attractor(volume, centroid, bass)
            
            # 优化绘制
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
        print("启动优雅吸引子 - 简约美学")
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
        finally:
            self.on_close(None)

# 主程序
if __name__ == "__main__":
    app = ElegantAttractor()
    app.run()