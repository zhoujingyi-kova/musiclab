"""
万花筒5（PyAudio 兼容版）
依赖：pyaudio, numpy, matplotlib, scipy
说明：移除 sounddevice，确保能在 pyaudio_venv 中运行。
键位：1/2/3/4 切换样式；A 开/关自动轮播
"""

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import matplotlib.colors as mcolors
import threading
import queue
import time
import math


class Kaleidoscope5:
    def __init__(self):
        # 音频参数
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.SILENCE_THRESHOLD = 0.005

        # 音频
        self.audio_queue = queue.Queue()
        self.p = None
        self.stream = None

        # 画布
        plt.ion()
        self.fig = plt.figure(figsize=(10, 10), dpi=100, facecolor='black')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('black')
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        try:
            self.fig.canvas.manager.set_window_title('万花筒5（PyAudio 版）')
        except Exception:
            pass

        # 状态
        self.running = True
        self.rotation = 0.0
        self.color_shift = 0.0
        self.target_fps = 30
        self._last = time.time()
        # 视觉模式：0=原版球+环，1=驻波球(cymatics)，2=霓虹蕾丝(moiré)，3=放射花瓣(mandala)
        self.style_mode = 0
        self.auto_cycle = True
        self.cycle_interval = 10.0
        self._last_switch = time.time()
        self._phase = 0.0

        # 启动音频
        self.init_pyaudio()
        self.start_audio_capture()
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def init_pyaudio(self):
        try:
            self.p = pyaudio.PyAudio()
            print('PyAudio 初始化成功')
        except Exception as e:
            print(f'PyAudio 初始化失败: {e}')
            self.p = None

    def start_audio_capture(self):
        if not self.p:
            print('无 PyAudio，启用模拟音频')
            self.start_simulated_audio()
            return
        try:
            device_index = None
            dev_info = None
            for i in range(self.p.get_device_count()):
                info = self.p.get_device_info_by_index(i)
                if info.get('maxInputChannels', 0) > 0:
                    device_index = i
                    dev_info = info
                    print(f"找到输入设备: {info.get('name', str(i))} (索引: {i})")
                    break
            if device_index is None:
                print('未找到输入设备，使用模拟音频')
                self.start_simulated_audio()
                return
            # 使用设备默认采样率（macOS 可避免 AUHAL -50）
            try:
                default_rate = int(round(dev_info.get('defaultSampleRate', self.RATE)))
                if default_rate > 0:
                    self.RATE = default_rate
                print(f'使用设备默认采样率: {self.RATE} Hz')
            except Exception:
                pass

            t = threading.Thread(target=self.capture_audio, args=(device_index,))
            t.daemon = True
            t.start()
        except Exception as e:
            print(f'启动音频失败: {e}')
            self.start_simulated_audio()

    def start_simulated_audio(self):
        def simulate():
            start = time.time()
            while self.running:
                tnow = time.time() - start
                vol = 0.3 + 0.25 * math.sin(2.5 * tnow)
                centroid = 0.4 + 0.3 * math.sin(1.4 * tnow)
                bass = 0.4 + 0.3 * math.sin(0.9 * tnow)
                mid = 0.5 + 0.3 * math.sin(1.2 * tnow)
                treble = 0.6 + 0.25 * math.sin(2.1 * tnow)
                try:
                    self.audio_queue.put({
                        'volume': vol,
                        'centroid': centroid,
                        'bass': bass,
                        'mid': mid,
                        'treble': treble
                    }, block=False)
                except Exception:
                    pass
                time.sleep(0.03)
        t = threading.Thread(target=simulate)
        t.daemon = True
        t.start()

    def capture_audio(self, device_index):
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
            self.stream = stream
            print('音频流创建成功，开始捕获...')
            while self.running:
                try:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    self.process_features(audio)
                except Exception as e:
                    print(f'音频处理错误: {e}')
                    time.sleep(0.05)
                    break
        except Exception as e:
            print(f'音频捕获错误: {e}')
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            self.stream = None

    def process_features(self, audio):
        vol = float(np.sqrt(np.mean(audio**2)))
        fft_data = np.abs(fft(audio))[: self.CHUNK // 2]
        if fft_data.size == 0 or fft_data.sum() == 0:
            centroid = 0.5
            bass = mid = treble = 0.5
        else:
            freqs = np.linspace(0, self.RATE / 2, len(fft_data))
            centroid_hz = float(np.sum(freqs * fft_data) / np.sum(fft_data))
            centroid = min(centroid_hz / (self.RATE / 4), 1.0)
            b = int(len(fft_data) * 0.1)
            m = int(len(fft_data) * 0.5)
            t = int(len(fft_data) * 0.8)
            bass = float(np.mean(fft_data[:b])) if b > 0 else 0.0
            mid = float(np.mean(fft_data[b:m])) if m > b else 0.0
            treble = float(np.mean(fft_data[m:t])) if t > m else 0.0
            mx = max(bass, mid, treble, 1e-8)
            bass, mid, treble = (bass / mx) ** 0.7, (mid / mx) ** 0.7, (treble / mx) ** 0.7
        try:
            self.audio_queue.put({
                'volume': 1 - math.exp(-5 * vol),
                'centroid': centroid,
                'bass': bass,
                'mid': mid,
                'treble': treble
            }, block=False)
        except Exception:
            pass

    def get_features(self):
        feats = None
        try:
            while not self.audio_queue.empty():
                feats = self.audio_queue.get_nowait()
        except Exception:
            pass
        return feats

    def palette(self, centroid, volume):
        self.color_shift += 0.01
        if centroid > 0.6:
            hues = [0.7, 0.78, 0.85]
        elif centroid < 0.4:
            hues = [0.02, 0.08, 0.14]
        else:
            hues = [0.32, 0.38, 0.44]
        hues = [(h + self.color_shift) % 1.0 for h in hues]
        sat = 0.7 + 0.3 * volume
        val = 0.6 + 0.4 * volume
        return [mcolors.hsv_to_rgb([h, sat, val]) for h in hues]

    def draw(self, feats):
        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_zlim(-2, 2)
        self.ax.view_init(elev=20, azim=self.rotation)

        volume = feats['volume']
        centroid = feats['centroid']
        colors = self.palette(centroid, volume)

        # 自动轮播
        if self.auto_cycle and (time.time() - self._last_switch > self.cycle_interval):
            self.style_mode = (self.style_mode + 1) % 4
            self._last_switch = time.time()

        # 相位推进
        self._phase += 0.05 + 0.3 * centroid

        if self.style_mode == 0:
            self.draw_base_spheres_and_torus(volume, centroid, colors)
        elif self.style_mode == 1:
            self.draw_cymatics_sphere(volume, centroid)
        elif self.style_mode == 2:
            self.draw_moire_lace(volume, centroid)
        else:
            self.draw_radial_petals(volume, centroid)

        self.rotation += 0.5 + 1.5 * volume

    def draw_base_spheres_and_torus(self, volume, centroid, colors):
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 40)
        U, V = np.meshgrid(u, v)
        for i in range(2):
            radius = 0.9 + 0.4 * i
            deform = 0.2 * volume * np.sin(3 * U) + 0.1 * centroid * np.cos(2 * V)
            R = radius * (1 + deform)
            X = R * np.sin(V) * np.cos(U)
            Y = R * np.sin(V) * np.sin(U)
            Z = R * np.cos(V)
            ang = self.rotation * (1 + 0.4 * i)
            Xr = X * np.cos(ang) - Y * np.sin(ang)
            Yr = X * np.sin(ang) + Y * np.cos(ang)
            Zr = Z
            self.ax.plot_surface(
                Xr, Yr, Zr,
                color=self.palette(centroid, volume)[i % 3],
                alpha=0.12 + 0.15 * volume,
                rstride=2, cstride=2, linewidth=0
            )

        theta = np.linspace(0, 2 * np.pi, 50)
        phi = np.linspace(0, 2 * np.pi, 24)
        Th, Ph = np.meshgrid(theta, phi)
        Rm, rm = 1.1, 0.28 + 0.18 * volume
        X = (Rm + rm * np.cos(Th)) * np.cos(Ph)
        Y = (Rm + rm * np.cos(Th)) * np.sin(Ph)
        Z = rm * np.sin(Th)
        ang = self.rotation * 0.8
        Xr = X * np.cos(ang) - Z * np.sin(ang)
        Yr = Y
        Zr = X * np.sin(ang) + Z * np.cos(ang)
        self.ax.plot_surface(
            Xr, Yr, Zr,
            color=self.palette(centroid, volume)[2 % 3],
            alpha=0.18 + 0.1 * volume,
            rstride=2, cstride=2, linewidth=0
        )

    def draw_cymatics_sphere(self, volume, centroid):
        # 驻波样式：叠加两个球面波，形成Chladni样节点
        u = np.linspace(0, 2 * np.pi, 64)
        v = np.linspace(0, np.pi, 64)
        U, V = np.meshgrid(u, v)
        n = 4 + int(6 * volume)
        m = 5 + int(6 * centroid)
        amp = 0.18 + 0.25 * volume
        pattern = np.sin(n * U + self._phase) * np.sin(m * V + 0.7 * self._phase)
        R = 1.2 * (1 + amp * pattern)

        X = R * np.sin(V) * np.cos(U)
        Y = R * np.sin(V) * np.sin(U)
        Z = R * np.cos(V)

        # 颜色：高对比霓虹（根据pattern映射）
        # NumPy 2.0 移除了 ndarray.ptp()，使用 np.ptp(pattern) 以保持兼容
        norm = (pattern - pattern.min()) / (np.ptp(pattern) + 1e-9)
        hues = (0.6 + 0.4 * norm + 0.1 * math.sin(self._phase)) % 1.0
        sat = 0.8
        val = 0.7 + 0.3 * volume
        fc = np.zeros((norm.shape[0], norm.shape[1], 3))
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                fc[i, j] = mcolors.hsv_to_rgb([hues[i, j], sat, val])

        self.ax.plot_surface(X, Y, Z, facecolors=fc, rstride=2, cstride=2, linewidth=0,
                             antialiased=False, alpha=0.9)

    def draw_moire_lace(self, volume, centroid):
        # 霓虹蕾丝：大量经纬线+轻微半径扰动，形成莫尔条纹
        lines_lon = 36
        lines_lat = 18
        rad = 1.3
        k1 = 6 + int(6 * centroid)
        k2 = 4 + int(6 * volume)
        amp = 0.12 + 0.15 * volume
        lw = 0.7 + 1.8 * volume
        hue0 = (self.color_shift + 0.2 * centroid) % 1.0
        for i in range(lines_lon):
            u = np.linspace(0, 2 * np.pi, 150)
            v = (i / lines_lon) * np.pi
            R = rad * (1 + amp * np.sin(k1 * u + self._phase) * np.sin(k2 * v + 0.5 * self._phase))
            X = R * np.sin(v) * np.cos(u)
            Y = R * np.sin(v) * np.sin(u)
            Z = R * np.cos(v)
            c = mcolors.hsv_to_rgb([(hue0 + i / lines_lon) % 1.0, 0.9, 0.9])
            self.ax.plot(X, Y, Z, color=c, alpha=0.25, linewidth=lw)
        for j in range(1, lines_lat):
            v = np.linspace(0, np.pi, 150)
            u = (j / lines_lat) * 2 * np.pi
            R = rad * (1 + amp * np.sin(k1 * u + self._phase) * np.sin(k2 * v + 0.5 * self._phase))
            X = R * np.sin(v) * np.cos(u)
            Y = R * np.sin(v) * np.sin(u)
            Z = R * np.cos(v)
            c = mcolors.hsv_to_rgb([(hue0 + j / lines_lat) % 1.0, 0.9, 0.9])
            self.ax.plot(X, Y, Z, color=c, alpha=0.25, linewidth=lw)

    def draw_radial_petals(self, volume, centroid):
        # 放射花瓣：多条花瓣曲线围绕圆周旋转
        petals = 6 + int(8 * centroid)
        t = np.linspace(0, 2 * np.pi, 600)
        r = 1.0 + 0.35 * np.sin((petals / 2) * t + self._phase) + 0.15 * np.sin(3 * t + 0.7 * self._phase)
        z = 0.3 * np.sin(2 * t + self._phase) * (0.4 + volume)
        hue0 = (self.color_shift + 0.1) % 1.0
        for k in range(petals):
            ang = 2 * np.pi * k / petals
            x = r * np.cos(t)
            y = r * np.sin(t)
            xr = x * np.cos(ang) - y * np.sin(ang)
            yr = x * np.sin(ang) + y * np.cos(ang)
            c = mcolors.hsv_to_rgb([(hue0 + k / petals) % 1.0, 0.85, 0.95])
            self.ax.plot(xr, yr, z, color=c, linewidth=1.5 + 1.5 * volume, alpha=0.7)

    def run(self):
        print('启动 万花筒5（PyAudio 版）')
        print('提示: 若无麦克风权限或设备，程序会自动使用模拟音频')
        try:
            while self.running:
                # 帧率限制
                now = time.time()
                if now - self._last < 1.0 / self.target_fps:
                    time.sleep(0.001)
                    continue
                self._last = now

                feats = self.get_features()
                if feats is None:
                    time.sleep(0.01)
                    continue

                self.draw(feats)
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
        except KeyboardInterrupt:
            pass
        finally:
            self.on_close(None)

    def on_key(self, event):
        if not event or not event.key:
            return
        key = event.key.lower()
        if key in ['1', '2', '3', '4']:
            self.style_mode = int(key) - 1
            self.auto_cycle = False
            self._last_switch = time.time()
            print(f'切换样式 -> {self.style_mode}')
        elif key == 'a':
            self.auto_cycle = not self.auto_cycle
            print(f'自动轮播: {"开启" if self.auto_cycle else "关闭"}')

    def on_close(self, _):
        print('正在关闭...')
        self.running = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
        if self.p:
            try:
                self.p.terminate()
            except Exception:
                pass
        print('已退出')


if __name__ == '__main__':
    app = Kaleidoscope5()
    app.run()