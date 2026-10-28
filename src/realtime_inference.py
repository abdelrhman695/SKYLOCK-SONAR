"""
SudarshanChakra - Real-time Serial Inference
==============================================
يقرأ بيانات صوتية خام (Raw PCM) من ESP32 عبر منفذ Serial،
يجمعها في بافر بحجم ثانية واحدة، ويمررها لنفس موديل الاستدلال
الأصلي (ThreatDetector.detect) عن طريق ملف WAV مؤقت في الذاكرة.
"""

import sys
import time
import wave
import io
from pathlib import Path
from typing import Optional

import numpy as np
import serial

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import Config
from src.inference import ThreatDetector  # نفس الكلاس الأصلي عندك بدون أي تعديل


# ============== إعدادات الصوت - لازم تطابق كود الأردوينو ==============
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2          # int16 (بعد التحويل في الأردوينو)
CHANNELS = 1
WINDOW_SECONDS = 1.0          # طول نافذة الاستدلال (ثانية واحدة)
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
WINDOW_BYTES = WINDOW_SAMPLES * BYTES_PER_SAMPLE


def pcm_to_temp_wav(pcm_bytes: bytes) -> io.BytesIO:
    """
    يحوّل بيانات PCM خام (int16) إلى ملف WAV صحيح في الذاكرة (مش على الديسك)
    عشان نقدر نمرره مباشرة لنفس AudioTransform بتاعك من غير أي تعديل فيها.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(BYTES_PER_SAMPLE)   # 2 بايت = 16-بت
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    buf.seek(0)
    return buf


class SerialAudioStreamer:
    """
    يقرأ من منفذ Serial، يجمع بافر صوتي بحجم نافذة كاملة،
    ثم يستدعي موديل الاستدلال الأصلي فوراً على كل نافذة.
    """

    def __init__(self, port: str, baud: int, detector: ThreatDetector,
                 save_temp_to_disk: bool = False, temp_dir: Optional[Path] = None):
        self.detector = detector
        self.save_temp_to_disk = save_temp_to_disk
        self.temp_dir = temp_dir or Path("temp_audio")
        if self.save_temp_to_disk:
            self.temp_dir.mkdir(exist_ok=True)

        self.ser = serial.Serial(port, baud, timeout=1)
        self.buffer = bytearray()
        self._frame_counter = 0

        print(f"\n[SERIAL] Connected to {port} @ {baud} baud")
        time.sleep(2)  # وقت لاستقرار اتصال الـ ESP32 بعد فتح البورت
        self.ser.reset_input_buffer()

    def _read_window(self) -> Optional[bytes]:
        """يقرأ من السيريال حتى يتجمع بافر كامل (ثانية كاملة من الصوت)."""
        available = self.ser.in_waiting
        if available > 0:
            self.buffer.extend(self.ser.read(available))

        if len(self.buffer) < WINDOW_BYTES:
            return None

        chunk_bytes = bytes(self.buffer[:WINDOW_BYTES])
        del self.buffer[:WINDOW_BYTES]
        return chunk_bytes

    def run(self):
        print("[STREAM] Listening for audio... (Ctrl+C to stop)\n")
        try:
            while True:
                pcm_chunk = self._read_window()

                if pcm_chunk is None:
                    time.sleep(0.01)
                    continue

                self._frame_counter += 1

                wav_buffer = pcm_to_temp_wav(pcm_chunk)

                if self.save_temp_to_disk:
                    temp_path = self.temp_dir / f"frame_{self._frame_counter:06d}.wav"
                    with open(temp_path, "wb") as f:
                        f.write(wav_buffer.getvalue())
                    audio_source = temp_path
                else:
                    audio_source = wav_buffer

                try:
                    result = self.detector.detect(audio_source)
                    self.detector.print_alert(result)
                except Exception as e:
                    print(f"[ERROR] Inference failed on frame {self._frame_counter}: {e}")

        except KeyboardInterrupt:
            print("\n[STREAM] Stopped by user.")
        finally:
            self.ser.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Real-time UAV Acoustic Threat Detection")
    parser.add_argument("--port", "-p", type=str, required=True,
                         help="Serial port (e.g. COM5 or /dev/ttyUSB0)")
    parser.add_argument("--baud", "-b", type=int, default=921600,
                         help="Baud rate (must match ESP32 code, default: 921600)")
    parser.add_argument("--model", "-m", type=str, default=None,
                         help="Path to model checkpoint")
    parser.add_argument("--threshold", "-t", type=float,
                         default=Config.THREAT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--save-frames", action="store_true",
                         help="Save each audio window as a .wav file for debugging")

    args = parser.parse_args()

    model_path = Path(args.model) if args.model else None
    detector = ThreatDetector(model_path=model_path, threshold=args.threshold)

    streamer = SerialAudioStreamer(
        port=args.port,
        baud=args.baud,
        detector=detector,
        save_temp_to_disk=args.save_frames,
    )
    streamer.run()


if __name__ == "__main__":
    main()