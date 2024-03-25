from fastapi import FastAPI, UploadFile, File
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

app = FastAPI()

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    # Save the uploaded file
    with open("temp.wav", "wb") as buffer:
        buffer.write(await file.read())

    # Load the audio file
    speech, rate = librosa.load("temp.wav", sr=16000)

    # Tokenize the audio
    input_values = tokenizer(speech, return_tensors='pt').input_values

    # Perform transcription
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcriptions = tokenizer.decode(predicted_ids[0])

    return {"transcription": transcriptions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
