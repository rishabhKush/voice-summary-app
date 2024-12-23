    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import torch
    import config

    class SpeechToText:
        def __init__(self, model_path = config.WHISPER_MODEL_PATH):
            self.processor = WhisperProcessor.from_pretrained(model_path)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        def transcribe(self, audio_data, sampling_rate):
            """Transcribes audio to text using Whisper"""
            try:
                input_features = self.processor(audio=audio_data, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(self.device)
                predicted_ids = self.model.generate(input_features)
                transcript = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                return transcript
            except Exception as e:
                print("Error during transcription:", e)
                return "Error"
