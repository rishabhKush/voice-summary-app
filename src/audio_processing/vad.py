   import torch
   from silero_vad import vad
   import config

   class VoiceActivityDetector:
       def __init__(self, threshold = config.VAD_THRESHOLD, window_size=config.VAD_WINDOW_SIZE):
           self.model = vad.VAD()
           self.threshold = threshold
           self.window_size = window_size

       def detect_speech_segments(self, audio_data, sampling_rate):
           """Detects speech segments using Silero VAD"""
           int16_audio = audio_data
           speech_segments = self.model(
                torch.from_numpy(int16_audio).float(),
                sampling_rate=sampling_rate,
                window=self.window_size
           )
           return speech_segments
