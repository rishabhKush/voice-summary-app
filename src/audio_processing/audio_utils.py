    import librosa
    import numpy as np
    import speech_recognition as sr
    import io
    import wave
    def load_audio(audio_file_path):
        """Loads an audio file and returns the audio data and sample rate"""
        with sr.AudioFile(audio_file_path) as source:
            r = sr.Recognizer()
            audio = r.record(source)
            audio_data = np.frombuffer(audio.frame_data, np.int16)
            sampling_rate = source.SAMPLE_RATE
            return audio_data, sampling_rate

    def fix_audio_length(audio_data, sampling_rate):
       """Fixes audio length to make it compatible with the VAD model"""
       return librosa.util.fix_length(audio_data, len(audio_data) % (sampling_rate // 100))

    def convert_bytes_to_array(audio_bytes, sampling_rate=16000):
        """converts bytes to array"""
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wave_file:
            frame_rate = wave_file.getframerate()
            n_channels = wave_file.getnchannels()
            sample_width = wave_file.getsampwidth()
            audio_data = wave_file.readframes(wave_file.getnframes())
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
        if frame_rate!=sampling_rate:
            audio_array = librosa.resample(audio_array.astype('float32'), orig_sr=frame_rate, target_sr=sampling_rate)

        return audio_array
