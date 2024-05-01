import keyboard
import pyaudio
import wave
import os
import whisper
import pyautogui
import tempfile
import time
import uuid
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_dict = {'distil_english_only': "distil-whisper/distil-medium.en",
              'small': 'openai/whisper-small',
              'base':'openai/whisper-base',
              'large': 'openai/whisper-large-v3'}

model_id = model_dict['small']

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    #max_new_tokens=128,
    #chunk_length_s=15,
    device=device,
    return_language=True
)


def get_temp_filename():
    return os.path.join(tempfile.gettempdir(), str(uuid.uuid4()) + '.wav')

def record_audio(filename):
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 44100

    p = pyaudio.PyAudio()
    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    frames = []

    while keyboard.is_pressed('ctrl+shift+x'):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))


def transcribe_audio(filename):
    time.sleep(0.5)  # Small delay to ensure file is released
    #result = pipe(filename)
    model = whisper.load_model("small")
    result = model.transcribe(filename)
    print(result)
    return result["text"]


def paste_transcription(transcription):
    pyautogui.write(transcription)


def main():
    print("Press and hold ctrl+shift+x to start recording, release to stop.")
    message = "Recording!"

    while True:
        if keyboard.is_pressed('ctrl+shift+x'):
            temp_filename = get_temp_filename()
            print("Recording started...")
            record_audio(temp_filename)
            #pyautogui.write(message, interval=0.02)
            print("Recording stopped.")
            transcription = transcribe_audio(temp_filename)
            #for _ in message:
            #    pyautogui.press('backspace')
            #    time.sleep(0.05)
            os.remove(temp_filename)
            paste_transcription(transcription)
            print("Transcription pasted.")
        time.sleep(0.1)


if __name__ == "__main__":
    main()
