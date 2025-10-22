"""
Audio-driven 3D Kaleidoscope Visualization System + Particle Effects + Ring Noise
Using PyAudio for audio input
Dependencies: numpy, matplotlib, pyaudio
Installation: pip install numpy matplotlib pyaudio
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
    """Particle class, based on the provided code logic"""
    def __init__(self, canvas_width=4, canvas_height=4):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.init()
        
    def init(self):
        """Initialize particle parameters"""
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
        """HSV to RGB color conversion"""
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
        """Update color based on audio features"""
        brightness = min(1.0, 0.3 + energy * 0.7)
        hue_shift = (self.audio_freq_hue_shift + freq_hue) % 1.0
        
        # Adjust saturation based on frequency energy
        # Bass uses high saturation, non-bass uses low saturation
        saturation = 0.3 + low_freq_energy * 0.7  # Bass has high saturation
        saturation += mid_freq_energy * 0.2  # Mid frequencies have medium saturation
        saturation += high_freq_energy * 0.1  # High frequencies have low saturation
        saturation = min(1.0, saturation)
        
        r, g, b = self.hsv_to_rgb(hue_shift, saturation, brightness)
        self.col = (r, g, b, 0.8)
    
    def move(self, audio_energy=0, dominant_freq=0):
        """Update particle state"""
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
    """Ring noise system, based on the provided code logic"""
    def __init__(self):
        self.rings = []
        self.time = 0
        self.audio_energy = 0
        self.dominant_freq = 0
        self.low_freq_energy = 0
        self.mid_freq_energy = 0
        self.high_freq_energy = 0
        
    def update(self, audio_energy, dominant_freq, low_freq_energy, mid_freq_energy, high_freq_energy, frame):
        """Update noise system"""
        self.audio_energy = audio_energy
        self.dominant_freq = dominant_freq
        self.low_freq_energy = low_freq_energy
        self.mid_freq_energy = mid_freq_energy
        self.high_freq_energy = high_freq_energy
        self.time = frame * 0.01  # Time accumulation
        
    def generate_ring_points(self, num_rings=12, points_per_ring=200):
        """Generate ring noise points"""
        points = []
        colors = []
        sizes = []
        
        # Main loop: 360° every 30° one ring
        for ring_index, d in enumerate(range(360, 0, -30)):
            ring_radius = d / 180.0  # Normalized radius
            
            for i in range(points_per_ring):
                r = (i / points_per_ring) * 2 * math.pi
                
                # Noise sampling (consistent with provided parameters)
                n = self.simplex_noise(d * 0.01, r * 0.02, -self.time * 0.01)
                
                # tangent transparency factor
                T = math.tan(n * 9)
                if abs(T) < 0.001:  # Avoid division by zero
                    continue
                    
                alpha = min(1.0, 1.0 / abs(T))
                
                # Hue based on angle and audio frequency
                hue = ((r * 57) % 360) / 360.0 + (self.dominant_freq / 2000)
                
                # Adjust saturation based on frequency energy
                # Bass uses high saturation, non-bass uses low saturation
                saturation = 0.3 + self.low_freq_energy * 0.7  # Bass has high saturation
                saturation += self.mid_freq_energy * 0.2  # Mid frequencies have medium saturation
                saturation += self.high_freq_energy * 0.1  # High frequencies have low saturation
                saturation = min(1.0, saturation)
                
                brightness = 0.8 + self.audio_energy * 0.2
                
                # HSV to RGB
                color = self.hsv_to_rgb(hue % 1.0, saturation, brightness)
                color = (*color, alpha)  # Add transparency
                
                # Rotation angle A drifts with time
                A = r - self.time * 0.01 + math.radians(d)
                
                # 3D position calculation
                x = math.cos(A) * ring_radius
                y = math.sin(A) * ring_radius
                z = math.sin(self.time + ring_index) * 0.5  # Add Z-axis fluctuation
                
                # Audio energy affects radius
                radius_scale = 1.0 + self.audio_energy * 2
                x *= radius_scale
                y *= radius_scale
                
                points.append((x, y, z))
                colors.append(color)
                sizes.append(2 + alpha * 8)  # Point size based on transparency
                
        return points, colors, sizes
    
    def simplex_noise(self, x, y, z):
        """Simplified noise function (alternative to Perlin noise)"""
        # Use trigonometric combination to simulate noise effect
        return (math.sin(x * 10 + self.time) * 0.3 +
                math.cos(y * 8 + self.time * 1.3) * 0.3 +
                math.sin(z * 12 + self.time * 0.7) * 0.4)
    
    def hsv_to_rgb(self, h, s, v):
        """HSV to RGB color conversion"""
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
        # Audio parameters
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.audio_buffer = deque(maxlen=5)
        
        # PyAudio instance
        self.audio = pyaudio.PyAudio()
        
        # Visualization parameters - using square window
        self.fig = plt.figure(figsize=(10, 10))  # Square window
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Audio features
        self.energy = 0.0
        self.dominant_freq = 0.0
        self.low_freq_energy = 0.0
        self.mid_freq_energy = 0.0
        self.high_freq_energy = 0.0
        self.spectrum = np.zeros(self.chunk_size // 2)
        
        # Control parameters
        self.is_playing = True
        self.paused = False
        self.rotation_speed = 0.0
        self.scale_factor = 1.0
        self.color_shift = 0.0
        self.light_position = [1, 1, 1]
        
        # Particle system
        self.particles = []
        self.num_particles = 50
        self._init_particles()
        
        # Noise ring system
        self.noise_rings = NoiseRingSystem()
        
        # Current display mode
        self.current_mode = 'hybrid'
        
        self._setup_audio()
        self._setup_visualization()
    
    def _init_particles(self):
        """Initialize particle system"""
        for _ in range(self.num_particles):
            self.particles.append(Particle())
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio audio callback function"""
        if status:
            print(f"Audio status: {status}")
        
        if self.is_playing and not self.paused:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_buffer.append(audio_data.copy())
        
        return (None, pyaudio.paContinue)
    
    def _setup_audio(self):
        """Set up PyAudio input stream"""
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
            print("PyAudio audio input started - listening to microphone...")
            
        except Exception as e:
            print(f"Cannot start audio input: {e}")
            print("Please check if microphone is available, or try changing audio device")
    
    def _setup_visualization(self):
        """Set up visualization parameters"""
        # Set axis limits to make graphics cover the entire window
        self.ax.set_xlim([-3, 3])
        self.ax.set_ylim([-3, 3])
        self.ax.set_zlim([-3, 3])
        self.ax.set_facecolor('black')
        self.ax.grid(False)
        self.ax.axis('off')
        
        # Adjust subplot position to make graphics fill the entire window
        self.ax.set_position([0, 0, 1, 1])
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        print("Visualization initialized - using keyboard control:")
        print("Space: Pause/Resume")
        print("Up/Down/Left/Right: Adjust parameters")
        print("1/2/3: Switch modes")
        print("+/-: Adjust particle count")
    
    def _update_mode(self, mode_name):
        """Update current display mode"""
        self.current_mode = mode_name
    
    def _analyze_audio(self):
        """Analyze audio data, extract features"""
        if not self.audio_buffer:
            return
        
        audio_data = np.concatenate(list(self.audio_buffer))
        
        if len(audio_data) < self.chunk_size:
            return
        
        # Apply Hanning window
        window = np.hanning(len(audio_data))
        windowed_data = audio_data * window
        
        # FFT transform
        fft_data = np.fft.fft(windowed_data)
        freq_magnitudes = np.abs(fft_data[:len(fft_data)//2])
        freqs = np.fft.fftfreq(len(windowed_data), 1/self.sample_rate)[:len(freq_magnitudes)]
        
        # Calculate energy (RMS)
        self.energy = np.sqrt(np.mean(audio_data**2))
        
        # Find dominant frequency
        if len(freq_magnitudes) > 0:
            dominant_idx = np.argmax(freq_magnitudes)
            self.dominant_freq = freqs[dominant_idx]
        
        # Calculate energy in different frequency bands
        # Low frequency (0-150Hz) - Bass
        low_freq_mask = (freqs >= 0) & (freqs <= 150)
        self.low_freq_energy = np.mean(freq_magnitudes[low_freq_mask]) if np.any(low_freq_mask) else 0
        
        # Mid frequency (150-1000Hz)
        mid_freq_mask = (freqs > 150) & (freqs <= 1000)
        self.mid_freq_energy = np.mean(freq_magnitudes[mid_freq_mask]) if np.any(mid_freq_mask) else 0
        
        # High frequency (above 1000Hz)
        high_freq_mask = freqs > 1000
        self.high_freq_energy = np.mean(freq_magnitudes[high_freq_mask]) if np.any(high_freq_mask) else 0
        
        # Normalize frequency band energy
        total_freq_energy = self.low_freq_energy + self.mid_freq_energy + self.high_freq_energy
        if total_freq_energy > 0:
            self.low_freq_energy /= total_freq_energy
            self.mid_freq_energy /= total_freq_energy
            self.high_freq_energy /= total_freq_energy
        
        self.spectrum = freq_magnitudes
    
    def _calculate_visual_parameters(self):
        """Calculate visualization parameters based on audio features"""
        self.rotation_speed = self.energy * 50
        self.scale_factor = 1.0 + self.energy * 3
        self.color_shift = (self.dominant_freq / 1000) % 1.0
    
    def _update_particles(self):
        """Update all particle states"""
        for particle in self.particles:
            particle.move(self.energy, self.dominant_freq)
            particle.update_color(self.energy, self.color_shift, 
                                self.low_freq_energy, self.mid_freq_energy, self.high_freq_energy)
    
    def _draw_particles(self):
        """Draw all particles"""
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
        """Draw noise rings"""
        # Update noise system
        self.noise_rings.update(self.energy, self.dominant_freq, 
                               self.low_freq_energy, self.mid_freq_energy, self.high_freq_energy, frame)
        
        # Generate noise points
        points, colors, sizes = self.noise_rings.generate_ring_points()
        
        if not points:
            return []
        
        # Separate coordinates
        x_pos = [p[0] for p in points]
        y_pos = [p[1] for p in points]
        z_pos = [p[2] for p in points]
        
        # Draw noise points
        scatter = self.ax.scatter(x_pos, y_pos, z_pos, 
                                 s=sizes, c=colors, 
                                 alpha=0.8, marker='o',
                                 edgecolors='none', depthshade=True)
        return [scatter]
    
    def _on_key_press(self, event):
        """Handle keyboard events"""
        if event.key == ' ':
            self.paused = not self.paused
            print(f"{'Paused' if self.paused else 'Resumed'} audio processing")
        
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
            print("Switched to particles only mode")
        
        elif event.key == '2':
            self._update_mode('noise_rings')
            print("Switched to noise rings mode")
        
        elif event.key == '3':
            self._update_mode('hybrid')
            print("Switched to hybrid mode")
        
        elif event.key == '+':
            self.num_particles = min(200, self.num_particles + 10)
            while len(self.particles) < self.num_particles:
                self.particles.append(Particle())
            print(f"Particle count: {self.num_particles}")
        
        elif event.key == '-':
            self.num_particles = max(10, self.num_particles - 10)
            self.particles = self.particles[:self.num_particles]
            print(f"Particle count: {self.num_particles}")
    
    def update_visualization(self, frame):
        """Update visualization display"""
        if self.paused:
            return []
        
        # Clear previous graphics
        self.ax.clear()
        
        # Set axis limits to make graphics cover the entire window
        self.ax.set_xlim([-3, 3])
        self.ax.set_ylim([-3, 3])
        self.ax.set_zlim([-3, 3])
        self.ax.set_facecolor('black')
        self.ax.grid(False)
        self.ax.axis('off')
        
        # Analyze audio
        self._analyze_audio()
        self._calculate_visual_parameters()
        
        visual_elements = []
        
        # Draw corresponding visualization elements based on current mode
        if self.current_mode == 'particles_only':
            self._update_particles()
            visual_elements.extend(self._draw_particles())
        
        elif self.current_mode == 'noise_rings':
            visual_elements.extend(self._draw_noise_rings(frame))
        
        elif self.current_mode == 'hybrid':
            # Hybrid mode: display all elements
            self._update_particles()
            visual_elements.extend(self._draw_particles())
            visual_elements.extend(self._draw_noise_rings(frame))
        
        # Set title to display audio features
        self.ax.text2D(0.02, 0.98, 
                      f'Audio Visualization | Energy: {self.energy:.3f} | Dominant Freq: {self.dominant_freq:.1f} Hz\n'
                      f'Low Freq: {self.low_freq_energy:.2f} | Mid Freq: {self.mid_freq_energy:.2f} | High Freq: {self.high_freq_energy:.2f}\n'
                      f'Particle Count: {self.num_particles} | Mode: {self.current_mode}',
                      transform=self.ax.transAxes, 
                      color='white', 
                      fontsize=10,
                      verticalalignment='top')
        
        # Add gray hint text in bottom right corner (English)
        self.ax.text2D(0.98, 0.02, 
                      'The louder the sound is played, the better the bass effect will be.',
                      transform=self.ax.transAxes, 
                      color='gray', 
                      fontsize=8,
                      horizontalalignment='right',
                      verticalalignment='bottom',
                      alpha=0.7)
        
        return visual_elements
    
    def start_visualization(self):
        """Start visualization"""
        print("Starting 3D audio visualization + particle system + noise rings...")
        self.animation = FuncAnimation(
            self.fig, 
            self.update_visualization, 
            interval=50,
            blit=False,
            cache_frame_data=False
        )
        plt.show()
    
    def cleanup(self):
        """Clean up resources"""
        self.is_playing = False
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()
        print("Resources cleaned up")

def main():
    """Main function"""
    try:
        visualizer = AudioVisualizer()
        visualizer.start_visualization()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        if 'visualizer' in locals():
            visualizer.cleanup()
       

if __name__ == "__main__":
    main()