
import re
import torch
import logging
from functools import lru_cache
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MarianTokenizer,
    MarianMTModel,
)
from deep_translator import GoogleTranslator
from langdetect import detect as langdetect_detect, LangDetectException
import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
Model_Name="kaniskaran/medical-finetuned"
MEDICAL_KEYWORDS: set = set(
    """
    health illness disease diabetes disorder infection injury condition symptom
    treatment diagnosis therapy prevention recovery examination patient doctor
    nurse hospital clinic medicine prescription dosage vaccination injection
    consultation appointment ward emergency operation surgeon specialist referral
    healthcare monitoring pulse blood pressure oxygen respiration heartbeat
    temperature pain nutrition hydration hygiene checkup admission discharge
    laboratory report test x-ray mri scan ultrasound ecg ct biopsy result chart
    record insurance consent procedure protocol icu opd telemedicine pharmacist
    anesthesia sterilization syringe drip infusion saline stretcher wheelchair
    sanitizer bandage wound dressing first aid rehabilitation fever cough cold
    sore throat headache fatigue nausea vomiting dizziness weakness swelling rash
    chills diarrhea constipation breathlessness wheezing appetite insomnia anxiety
    stress depression confusion sweating trembling blurred itching bleeding cramps
    numbness tingling fainting palpitations sneezing yellowing indigestion bloating
    dehydration stiffness phlegm hallucinations tremor surgery chemotherapy
    radiation physiotherapy dialysis antibiotic antiviral analgesic sedative
    suturing transfusion nebulization counseling psychotherapy insulin ointment
    inhaler nebulizer ventilator catheterization endoscopy stent pacemaker laser
    bacteria virus fungi parasite genetic hereditary allergy inflammation deficiency
    trauma obesity smoking toxins mutation autoimmune antibody enzyme metabolic
    cancer tumor clot blockage ischemia necrosis viral pcr urinalysis liver kidney
    hormone biopsy cytology pathology radiology hematology toxicology hypertension
    asthma tuberculosis pneumonia malaria dengue typhoid influenza covid heart
    stroke hepatitis arthritis anemia thyroid schizophrenia migraine epilepsy ulcer
    gastritis appendicitis bronchitis sinusitis tonsillitis eczema psoriasis acne
    dermatitis cholera measles mumps chickenpox rabies tetanus polio hiv aids lupus
    parkinson alzheimer glaucoma cataract leukemia lymphoma cirrhosis gallstones
    copd meningitis gout fracture burns sepsis jaundice malnutrition cardiac angina
    paralysis
    """.split()
)
def load_model_and_tokenizer(model_name: str = Model_Name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    mdl.to(device)
    return tok, mdl, device

def clean_text(text: str) -> str:
    if isinstance(text, list):
        text = " ".join(text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_medical_query(text: str) -> bool:
    text = clean_text(text)
    return any(f" {keyword} " in f" {text} " for keyword in MEDICAL_KEYWORDS)

def detect_language(text: str) -> str:
    try:
        lang = langdetect_detect(text)
        return lang.split("-")[0] if lang else config.DEFAULT_LANG
    except LangDetectException as e:
        logger.warning(f"Language detection failed: {e}")
        return config.DEFAULT_LANG

_marian_cache: dict = {}
def _load_marian(src: str, dest: str):
    key = f"{src}-{dest}"
    if key in _marian_cache:
        return _marian_cache[key]
    model_name = f"Helsinki-NLP/opus-mt-{src}-{dest}"
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = MarianTokenizer.from_pretrained(model_name)
        mdl = MarianMTModel.from_pretrained(model_name).to(device)
        _marian_cache[key] = (tok, mdl)
        logger.info(f"Loaded MarianMT model: {model_name}")
    except Exception as e:
        logger.error(f"Could not load MarianMT {model_name}: {e}")
        _marian_cache[key] = None
    return _marian_cache[key]

def _marian_translate(text: str, src: str, dest: str ="en") -> str:
    pair = _load_marian(src, dest)
    if pair is None:
        return text
    tok, mdl = pair
    device = next(mdl.parameters()).device
    batch = tok([text], return_tensors="pt", padding=True, truncation=True, max_length=512)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        gen = mdl.generate(**batch, max_new_tokens=256)
    return tok.decode(gen[0], skip_special_tokens=True)

@lru_cache(maxsize=1000)
def translate_text(text: str, src: str = None, dest: str = "en") -> str:
    if not text or not isinstance(text, str):
        return text or ""
    src_lang = (src or "").split("-")[0]
    dest_lang = dest.split("-")[0]
    if src_lang and src_lang == dest_lang:
        return text
    try:
        source = src_lang if src_lang else "auto"
        result = GoogleTranslator(source=source, target=dest_lang).translate(text)
        if result and isinstance(result, str) and result.strip():
            return result.strip()
        raise ValueError(f"Empty translation result: {repr(result)}")
    except Exception as e:
        logger.warning(f"deep-translator failed ({e}). Trying MarianMT...")
    try:
        src_resolved = (src_lang or detect_language(text)).split("-")[0]
        return _marian_translate(text, src=src_resolved, dest=dest_lang)
    except Exception as e:
        logger.error(f"MarianMT also failed: {e}. Returning original.")
        return text

def generate_medical_response(prompt: str, tok, mdl, device, max_new_tokens: int = 180) -> str:
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        return "Empty input provided."

    def extract_user_query(prompt: str) -> str:
        try:
            if "Question:" in prompt and "Answer:" in prompt:
                return prompt.split("Question:")[1].split("Answer:")[0].strip()
            return prompt.strip()
        except Exception:
            return prompt.strip()

    try:
        user_query = extract_user_query(prompt)

        if len(user_query.split()) < 2:
            return "Please enter a more detailed medical question for better results."
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = mdl.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                early_stopping=True
            )
        response = tok.decode(outputs[0], skip_special_tokens=True).strip()
        if len(response.split()) < 25:
            follow_up_prompt = f"{response}\n\nPlease elaborate with causes, symptoms, diagnosis, and treatments."
            
            inputs = tok(follow_up_prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = mdl.generate(
                    **inputs,
                    max_new_tokens=400,  
                    temperature=0.3,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    do_sample=True,
                    early_stopping=True
                )

            response = tok.decode(outputs[0], skip_special_tokens=True).strip()

    except Exception as e:
        logger.error(f"Generation error: {e}")
        return "Sorry, I cannot provide a response now. Please consult a medical professional."

    return response