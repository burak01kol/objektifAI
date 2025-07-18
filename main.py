import cv2
import torch
import numpy as np
import time
from collections import deque
import argparse
import os
from datetime import datetime
import threading
from queue import Queue

class OptimizedYOLOv5Detector:
    def __init__(self, model_name='yolov5x', conf_threshold=0.6, iou_threshold=0.45, device='auto'):
        """
        Ultra-Optimized YOLOv5x Detector - Maximum FPS Performance
        """
        # Parametreleri önce ata
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # GPU kullanımı için otomatik cihaz seçimi
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA kullanılamıyor, CPU'ya geçiliyor...")
            device = 'cpu'
        
        self.device = device
        
        # PyTorch optimizasyonları
        if device == 'cuda':
            torch.backends.cudnn.benchmark = True  # cuDNN otomatik tuning
            torch.backends.cudnn.deterministic = False
            # Mixed precision için
            self.use_half = True
        else:
            self.use_half = False
            
        # Model yükleme ve optimizasyon
        print(f"🚀 Model yükleniyor: {model_name} (Cihaz: {device})")
        try:
            self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
            self.model.conf = conf_threshold
            self.model.iou = iou_threshold
            self.model.to(device)
            
            # Model optimizasyonları
            self.model.eval()  # Evaluation mode için
            if device == 'cuda':
                if self.use_half:
                    self.model.half()  # FP16 precision
                # Model warmup
                self._warmup_model()
                
                gpu_name = torch.cuda.get_device_name(0)
                print(f"🎮 GPU kullanılıyor: {gpu_name}")
                print(f"⚡ Half precision: {'Aktif' if self.use_half else 'Deaktif'}")
            else:
                print("🖥️  CPU kullanılıyor")
                
            print(f"✅ Model optimizasyonu tamamlandı!")
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            raise
        
        # Optimized color palette - sadece gerekli renkleri hesapla
        self.colors = self._generate_colors()
        
        # FPS tracking - ring buffer ile optimize
        self.fps_history = deque(maxlen=30)
        self.last_time = time.perf_counter()
        self.frame_count = 0
        
        # Statistics tracking
        self.detection_stats = {}
        self.total_detections = 0
        
        # Threading için
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.processing = False
        
    def _warmup_model(self):
        """Model warmup - ilk çalıştırmadaki gecikmeyi önler"""
        print("🔥 Model warmup yapılıyor...")
        dummy_input = torch.zeros(1, 3, 640, 640).to(self.device)
        if self.use_half:
            dummy_input = dummy_input.half()
        
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(dummy_input)
        print("✅ Warmup tamamlandı!")
        
    def _generate_colors(self):
        """Optimize edilmiş renk paleti"""
        np.random.seed(42)  # Tutarlı renkler için
        return np.random.randint(0, 255, size=(80, 3), dtype=np.uint8)
    
    @torch.no_grad()  # Gradient hesaplamasını deaktive et
    def detect_objects_optimized(self, frame):
        """Ultra-optimize edilmiş nesne tanıma"""
        # Input preprocessing - optimize
        if self.device == 'cuda' and self.use_half:
            # GPU'da half precision kullan
            tensor_frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).half().to(self.device, non_blocking=True)
        else:
            tensor_frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device, non_blocking=True)
        
        # Normalize
        tensor_frame /= 255.0
        
        # Inference
        with torch.cuda.amp.autocast(enabled=self.use_half):
            results = self.model(tensor_frame)
        
        return results
    
    def process_detections_fast(self, results, frame_shape):
        """Hızlı detection processing"""
        detections = results.pandas().xyxy[0].values
        
        if len(detections) == 0:
            return []
        
        # Numpy array olarak işle (pandas'tan daha hızlı)
        boxes = []
        h, w = frame_shape[:2]
        
        for detection in detections:
            x1, y1, x2, y2, conf, class_id = detection[:6]
            
            # Boundary check
            x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
            
            boxes.append({
                'bbox': (x1, y1, x2, y2),
                'conf': conf,
                'class_id': int(class_id),
                'class_name': results.names[int(class_id)]
            })
            
        return boxes
    
    def draw_detections_fast(self, frame, detections):
        """Ultra-hızlı çizim işlemi"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['conf']
            class_id = det['class_id']
            class_name = det['class_name']
            
            # İstatistik güncelle
            self.total_detections += 1
            if class_name in self.detection_stats:
                self.detection_stats[class_name] += 1
            else:
                self.detection_stats[class_name] = 1
            
            # Renk - cache'lenmiş
            color = tuple(map(int, self.colors[class_id % len(self.colors)]))
            
            # Hızlı kutu çizimi
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Optimize edilmiş text rendering
            label = f"{class_name}: {conf:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Text background
            cv2.rectangle(frame, (x1, y1 - label_height - 8), 
                         (x1 + label_width + 4, y1), color, -1)
            
            # Text
            cv2.putText(frame, label, (x1 + 2, y1 - 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def calculate_fps_optimized(self):
        """Optimize edilmiş FPS hesaplama"""
        current_time = time.perf_counter()
        fps = 1.0 / (current_time - self.last_time)
        self.fps_history.append(fps)
        self.last_time = current_time
        
        # Moving average
        return sum(self.fps_history) / len(self.fps_history)
    
    def draw_performance_overlay(self, frame):
        """Performans bilgileri overlay"""
        h, w = frame.shape[:2]
        
        # FPS - optimize edilmiş
        fps = self.calculate_fps_optimized()
        fps_text = f"FPS: {fps:.1f}"
        
        # FPS color coding
        if fps > 25:
            fps_color = (0, 255, 0)  # Yeşil
        elif fps > 15:
            fps_color = (0, 255, 255)  # Sarı
        else:
            fps_color = (0, 0, 255)  # Kırmızı
            
        cv2.putText(frame, fps_text, (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, fps_color, 2)
        
        # Compact info panel
        info_texts = [
            f"GPU: {'✓' if self.device == 'cuda' else '✗'}",
            f"Half: {'✓' if self.use_half else '✗'}",
            f"Det: {self.total_detections}",
            f"Conf: {self.conf_threshold:.1f}"
        ]
        
        for i, text in enumerate(info_texts):
            cv2.putText(frame, text, (w - 150, 25 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    def save_frame_fast(self, frame):
        """Hızlı frame kaydetme"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_{timestamp}.jpg"
        # Non-blocking save
        threading.Thread(target=cv2.imwrite, args=(filename, frame)).start()
        print(f"📷 Kayıt başlatıldı: {filename}")
    
    def reset_stats(self):
        """Stats reset"""
        self.detection_stats.clear()
        self.total_detections = 0
        print("🔄 İstatistikler sıfırlandı!")
    
    def print_performance_stats(self):
        """Performans istatistikleri"""
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        print(f"\n{'='*50}")
        print(f"⚡ PERFORMANS İSTATİSTİKLERİ")
        print(f"{'='*50}")
        print(f"Ortalama FPS: {avg_fps:.1f}")
        print(f"Minimum FPS: {min(self.fps_history):.1f}" if self.fps_history else "N/A")
        print(f"Maksimum FPS: {max(self.fps_history):.1f}" if self.fps_history else "N/A")
        print(f"Toplam Tespit: {self.total_detections}")
        print(f"GPU Kullanımı: {'Aktif' if self.device == 'cuda' else 'Deaktif'}")
        print(f"Half Precision: {'Aktif' if self.use_half else 'Deaktif'}")
        
        if self.detection_stats:
            print(f"\nEn Çok Tespit Edilen:")
            sorted_stats = sorted(self.detection_stats.items(), key=lambda x: x[1], reverse=True)
            for class_name, count in sorted_stats[:5]:
                percentage = (count / self.total_detections) * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
        print(f"{'='*50}")

def setup_camera_optimized(camera_index, width, height):
    """Optimize edilmiş kamera kurulumu"""
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        return None
    
    # Performans optimizasyonları
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Yüksek FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimum buffer
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # MJPEG codec
    
    # Auto-exposure ve focus deaktive et (sabit performans için)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    
    return cap

def main():
    """Ultra-Optimize Edilmiş Ana Fonksiyon"""
    parser = argparse.ArgumentParser(description='🚀 YOLOv5x Ultra-Optimized Real-Time Detection')
    parser.add_argument('--model', default='yolov5x', choices=['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'],
                       help='Model türü (varsayılan: yolov5x)')
    parser.add_argument('--conf', type=float, default=0.6, help='Güven eşiği')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU eşiği')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Cihaz seçimi')
    parser.add_argument('--camera', type=int, default=0, help='Kamera indeksi')
    parser.add_argument('--width', type=int, default=1280, help='Görüntü genişliği')
    parser.add_argument('--height', type=int, default=720, help='Görüntü yüksekliği')
    parser.add_argument('--no-display', action='store_true', help='Görüntü gösterimi kapalı (max FPS)')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark modu')
    
    args = parser.parse_args()
    
    print("🚀 YOLOv5x ULTRA-OPTIMIZED DETECTOR")
    print("=" * 50)
    
    # Sistem bilgileri
    print(f"🖥️  PyTorch: {torch.__version__}")
    print(f"🎮 CUDA: {'✓' if torch.cuda.is_available() else '✗'}")
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Detector başlatma
    try:
        detector = OptimizedYOLOv5Detector(
            model_name=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device
        )
    except Exception as e:
        print(f"❌ Detector hatası: {e}")
        return
    
    # Optimize edilmiş kamera kurulumu
    print(f"📹 Kamera optimize ediliyor...")
    cap = setup_camera_optimized(args.camera, args.width, args.height)
    if cap is None:
        print(f"❌ Kamera açılamadı: {args.camera}")
        return
    
    # Gerçek kamera ayarları
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📐 Çözünürlük: {actual_width}x{actual_height}")
    print(f"🎬 Kamera FPS: {actual_fps}")
    print("=" * 50)
    
    print("⚡ ULTRA-OPTIMIZED DETECTION BAŞLADI!")
    print("\n🎮 Kontroller:")
    print("  Q/ESC - Çıkış")
    print("  S     - Hızlı kayıt")
    print("  R     - Stats reset")
    print("  +/-   - Güven ayarı")
    print("  P     - Performans stats")
    print("=" * 50)
    
    # Ana döngü - maksimum performans
    try:
        window_name = 'YOLOv5x ULTRA-OPTIMIZED' if not args.no_display else None
        
        # Benchmark değişkenleri
        if args.benchmark:
            benchmark_start = time.time()
            benchmark_frames = 0
        
        while True:
            # Frame okuma - optimize
            ret, frame = cap.read()
            if not ret:
                print("❌ Frame okunamadı!")
                break
            
            # Benchmark sayacı
            if args.benchmark:
                benchmark_frames += 1
                if benchmark_frames >= 300:  # 300 frame sonra dur
                    elapsed = time.time() - benchmark_start
                    avg_fps = benchmark_frames / elapsed
                    print(f"\n🏁 BENCHMARK TAMAMLANDI!")
                    print(f"⚡ Ortalama FPS: {avg_fps:.2f}")
                    print(f"📊 Toplam Frame: {benchmark_frames}")
                    print(f"⏱️  Süre: {elapsed:.2f}s")
                    break
            
            # Ultra-hızlı inference
            results = detector.detect_objects_optimized(frame)
            detections = detector.process_detections_fast(results, frame.shape)
            
            # Hızlı çizim
            detector.draw_detections_fast(frame, detections)
            
            # Performance overlay
            detector.draw_performance_overlay(frame)
            
            # Display (opsiyonel)
            if not args.no_display:
                cv2.imshow(window_name, frame)
                
                # Optimize edilmiş key handling
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:
                    break
                elif key == ord('s'):
                    detector.save_frame_fast(frame)
                elif key == ord('r'):
                    detector.reset_stats()
                elif key == ord('+') or key == ord('='):
                    detector.conf_threshold = min(0.95, detector.conf_threshold + 0.1)
                    detector.model.conf = detector.conf_threshold
                    print(f"📈 Güven: {detector.conf_threshold:.1f}")
                elif key == ord('-'):
                    detector.conf_threshold = max(0.1, detector.conf_threshold - 0.1)
                    detector.model.conf = detector.conf_threshold
                    print(f"📉 Güven: {detector.conf_threshold:.1f}")
                elif key == ord('p'):
                    detector.print_performance_stats()
                    
    except KeyboardInterrupt:
        print("\n⚠️  İnterrupt ile durduruldu.")
    
    finally:
        # Temizlik
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        
        # Final performans raporu
        detector.print_performance_stats()
        print("\n✅ Program sonlandırıldı.")

if __name__ == "__main__":
    main()