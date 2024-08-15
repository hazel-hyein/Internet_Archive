"""
Sentiment Analysis
"""

__date__ = "2024-08-12"
__author__ = "HyeinKim"
__version__ = "0.1"

# %% --------------------------------------------------------------------------
# Import Modules
import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
# -----------------------------------------------------------------------------

# 감성 분석 파이프라인 설정
model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # 사전 학습된 감성 분석 모델
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Load Hansard corpus data from csv file

dfs = pd.read_csv('output1850-1853.csv')

# 감성 분석 결과를 저장할 리스트
sentiment_results = []

for text in df['Content']:
    if pd.notna(text):  # 텍스트가 비어 있지 않은지 확인
        sentiment = sentiment_pipeline(text[:512])  # BERT 모델의 입력 길이 제한 고려
        sentiment_results.append({
            'text': text,
            'label': sentiment[0]['label'],
            'score': sentiment[0]['score']
        })

# 감성 분석 결과를 DataFrame으로 변환
sentiment_df = pd.DataFrame(sentiment_results)

# 감성 분석 결과 출력
print(sentiment_df.head())

# %%
# -----------------------------------------------------------------------------
