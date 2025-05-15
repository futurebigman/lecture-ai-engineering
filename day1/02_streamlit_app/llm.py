# llm.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import streamlit as st
import time
from config import MODEL_NAME
from huggingface_hub import login

# モデルをキャッシュして再利用
@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    try:

        # アクセストークンを保存
        hf_token = st.secrets["huggingface"]["token"]
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}") # 使用デバイスを表示
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        pipe = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            torch_dtype="auto",
            quantization_config=quant_config
            )
        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return tokenizer, pipe
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error("GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。")
        return None

def generate_response(tokenizer, pipe, user_question):
    """LLMを使用して質問に対する回答を生成する"""
    if pipe is None:
        return "モデルがロードされていないため、回答を生成できません。", 0

    try:
        start_time = time.time()

        inputs = tokenizer(user_question, return_tensors="pt", return_attention_mask=False).to(device=model.device)
        # max_new_tokensを調整可能にする（例）
        outputs = model.generate(**inputs, max_length=512)

        # 最後のassistantのメッセージを取得
        assistant_response = tokenizer.batch_decode(outputs)[0]


        end_time = time.time()
        response_time = end_time - start_time
        print(f"Generated response in {response_time:.2f}s") # デバッグ用
        return assistant_response, response_time

    except Exception as e:
        st.error(f"回答生成中にエラーが発生しました: {e}")
        # エラーの詳細をログに出力
        import traceback
        traceback.print_exc()
        return f"エラーが発生しました: {str(e)}", 0