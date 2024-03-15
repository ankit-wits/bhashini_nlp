from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForSeq2SeqLM
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer

app = FastAPI()


class TranslationRequest(BaseModel):
    strings: list[str] = ["Hello, how are you?"]
    source_lang: str = "eng_Latn"
    target_lang: str = "hin_Deva"
    direction: str = "en-indic"


# eng_Latn, ben_Beng

@app.get("/")
async def application():
    return 'Welcome to Bhashini Translation Service'


@app.get("/languages")
async def language_codes():
    return {"language_codes": {
        "English": "eng_Latn",
        "Hindi": "hin_Deva",
        "Bengali": "ben_Beng",
        "Malayalam": "mal_Mlym",
        "Kannada": "kan_Knda",
        "Telugu": "tel_Telu",
        "Tamil": "tam_Taml",
        "Gujarati": "guj_Gujr",
        "Marathi": "mar_Deva",
        "Punjabi": "pan_Guru",
        "Urdu": "urd_Arab",
        "Oriya": "ory_Orya",
        "Assamese": "asm_Beng",
        "Bhojpuri": "bho_Deva",
    }}


@app.post("/translate/")
async def translate_strings(request: TranslationRequest):
    try:
        tokenizer = IndicTransTokenizer(direction=request.direction)
        ip = IndicProcessor(inference=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(f"ai4bharat/indictrans2-{request.direction}-dist-200M",
                                                      trust_remote_code=True)

        batch = ip.preprocess_batch(request.strings, src_lang=request.source_lang, tgt_lang=request.target_lang)
        batch = tokenizer(batch, src=True, return_tensors="pt")

        with torch.inference_mode():
            outputs = model.generate(**batch, num_beams=5, num_return_sequences=1, max_length=256)

        outputs = tokenizer.batch_decode(outputs, src=False)
        outputs = ip.postprocess_batch(outputs, lang=request.target_lang)
        return {"translated_strings": outputs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
