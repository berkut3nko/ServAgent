import torch
from transformers import AutoModelForCTC, Wav2Vec2BertProcessor, BitsAndBytesConfig

model_name = "Yehor/w2v-bert-uk-v2.1"

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True 
)

asr_model = AutoModelForCTC.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    device_map="auto",
    quantization_config=quantization_config
)

processor = Wav2Vec2BertProcessor.from_pretrained(model_name)
print("Model was loaded")

torch.save(asr_model.state_dict(), "w2v-bert-uk.pt")

print("✅ Model saved as w2v-bert-uk.pt")