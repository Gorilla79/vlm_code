import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# 모델명 지정
model_name = "Salesforce/blip2-opt-2.7b"  # 또는 blip2-flan-t5-base

# processor와 model 다운로드 및 캐시 저장
processor = Blip2Processor.from_pretrained(model_name)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16  # 메모리 사용량 절약
)
