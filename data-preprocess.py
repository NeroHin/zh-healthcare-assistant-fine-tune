import pandas as pd
import re
import ollama
from tqdm import tqdm
import requests
import json
from openai import AsyncOpenAI
import os
from datetime import datetime
import unicodedata
import dotenv
import csv
import asyncio
from typing import List, Dict, Any

dotenv.load_dotenv()

# === 資料讀取與初始化 ===
def load_data():
    """讀取原始資料並記錄筆數"""
    df = pd.read_csv('data/tweh_items_keywords.csv', engine='pyarrow')
    
    # 確保必要欄位存在
    required_columns = ['article_department', 'article_question', 'article_answer']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"缺少必要欄位：{col}")
    
    print(f"[初始化] 原始資料筆數：{len(df)}")
    print("-" * 100)
    
    return df

# === 篩選類 (Filtering) ===
def filter_short_answers(df, min_length=20):
    """篩選回覆過短的資料"""
    filtered_df = df[df['article_answer'].str.len() >= min_length]
    print(f"[篩選類] 回覆過短：刪除 {len(df) - len(filtered_df)} 筆，剩餘 {len(filtered_df)} 筆")
    return filtered_df

def filter_non_unicode(df):
    """篩選非 Unicode 編碼的資料"""
    def is_valid_unicode(text):
        try:
            if pd.isna(text):
                return False
            # 檢查是否能正常編碼解碼
            text.encode('utf-8').decode('utf-8')
            # 檢查是否包含過多控制字符
            control_chars = sum(1 for char in text if unicodedata.category(char).startswith('C'))
            return control_chars < len(text) * 0.1
        except:
            return False
    
    valid_mask = (
        df['article_question'].apply(is_valid_unicode) &
        df['article_answer'].apply(is_valid_unicode)
    )
    
    filtered_df = df[valid_mask]
    print(f"[篩選類] 非 Unicode 編碼：刪除 {len(df) - len(filtered_df)} 筆，剩餘 {len(filtered_df)} 筆")
    return filtered_df

def filter_symbol_only(df):
    """篩選僅含符號的資料"""
    def is_meaningful_text(text):
        if pd.isna(text):
            return False
        # 移除空白
        text = text.strip()
        if not text:
            return False
        # 檢查是否全為標點符號或特殊符號
        text_chars = re.sub(r'[^\w\s]', '', text)
        return len(text_chars) > 0
    
    valid_mask = (
        df['article_question'].apply(is_meaningful_text) &
        df['article_answer'].apply(is_meaningful_text)
    )
    
    filtered_df = df[valid_mask]
    print(f"[篩選類] 僅含符號：刪除 {len(df) - len(filtered_df)} 筆，剩餘 {len(filtered_df)} 筆")
    return filtered_df

def filtering_stage(df):
    """執行篩選類處理"""
    print("[篩選類] 開始篩選階段...")
    
    # 1. 回覆過短
    df = filter_short_answers(df, min_length=20)
    
    # 2. 非 Unicode 編碼
    df = filter_non_unicode(df)
    
    # 3. 僅含符號
    df = filter_symbol_only(df)
    
    print(f"[篩選類] 篩選完成，總計剩餘 {len(df)} 筆")
    print("-" * 100)
    
    return df

# === 篩選後處理類 (Post-filter Processing) ===
class PromptTemplates:
    """Prompt 模板類，將所有 prompt 集中管理"""
    
    @staticmethod
    def get_system_prompt_for_cleaning():
        """獲取清理資料的系統 prompt"""
        return "你是一個專業的資料清理專家，請只回傳清理後的文字，不要回傳任何其他文字。"
    
    @staticmethod
    def get_url_cleaning_prompt(text):
        """獲取清理網址的 prompt"""
        return f"""
        你是一個專業的資料清理專家，請你將以下文字中的網址去除，也請清理像是指向網址的文字的內容，
        並回傳清理後的文字。請你只回傳清理後的文字，不要回傳任何其他文字，輸出使用繁體中文。
        文字：{text}
        """
    
    @staticmethod
    def get_system_prompt_for_symbol_add(text):
        """獲取新增標點符號的 prompt"""
        return f"""
        你是一位專業的資料清理與重寫專家，請你為下列文字新增適當標點符號並保留原意，確保：
        1. 不偏離原始文字意涵；
        2. 只回傳重寫後的文字，且僅以繁體中文輸出，禁止包含其他文字。
        文字：{text}
        """
    
    @staticmethod
    def get_system_prompt_for_question_rewrite(question, answer):
        """獲取問題重寫的系統 prompt"""
        return f"""
        你是一位專業的資料清理與重寫專家，請根據「回覆」內容適當補充並精簡「問題」，確保：
        1. 不偏離原始問題意涵；
        2. 不新增回覆中未提及的資訊；
        3. 要比原回覆更完整、不要刪減回覆內容；
        4. 只回傳重寫後的問題，且僅以繁體中文輸出，禁止包含其他文字。
        回覆：{answer}
        問題：{question}
        """

class AsyncModelAPI:
    """統一的異步模型 API 調用介面"""
    
    def __init__(self, openai_client: AsyncOpenAI = None):
        self.openai_client = openai_client
        # 按照 README 需求設定模型優先順序
        self.model_priority = [
            # {"type": "openai", "model": "gemini/gemini-2.0-flash-lite"},
            # {"type": "openai", "model": "groq/llama-3.1-8b-instant"},
            # {"type": "openai", "model": "groq/llama-3.3-70b-versatile"},
            {"type": "openai", "model": "gpt-4.1-nano"},
            {"type": "ollama", "model": "glm4:9b"}
        ]
    
    async def call_with_priority(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
        """按照優先順序調用模型（異步版本）"""
        for model_config in self.model_priority:
            try:
                if model_config["type"] == "openai":
                    response = await self.openai_client.chat.completions.create(
                        model=model_config["model"],
                        messages=messages,
                        temperature=temperature
                    )
                    return response.choices[0].message.content
                elif model_config["type"] == "ollama":
                    # 使用 asyncio.to_thread 將同步調用轉為異步
                    response = await asyncio.to_thread(
                        ollama.chat,
                        model=model_config["model"],
                        messages=messages,
                    )
                    return response.message.content
            except Exception as e:
                print(f"模型 {model_config['model']} 調用失敗：{str(e)}")
                continue
        
        raise Exception("所有模型都調用失敗")

def find_punctuation_missing(text):
    """檢查文字是否缺少標點符號"""
    if pd.isna(text):
        return False
    # 檢查是否完全不含常見標點符號
    punctuation_pattern = r'[。，！？；：、（）【】「」『』]'
    return not bool(re.search(punctuation_pattern, text))

def find_url_containing(question, answer):
    """檢查文字是否包含 URL"""
    url_pattern = r'https?://|www\.'
    return (
        pd.notna(question) and bool(re.search(url_pattern, question)) or
        pd.notna(answer) and bool(re.search(url_pattern, answer))
    )

def find_question_rewrite_candidates(question, answer):
    """檢查是否需要問題重寫"""
    return (
        pd.notna(question) and len(question) < 10 and
        pd.notna(answer) and len(answer) > 30
    )

async def process_single_row(row, model_api: AsyncModelAPI):
    """異步處理單筆資料"""
    processed_row = row.copy()
    
    # 1. 檢查並處理缺少標點符號
    if find_punctuation_missing(row['article_answer']):
        try:
            prompt = PromptTemplates.get_system_prompt_for_symbol_add(row['article_answer'])
            messages = [{"role": "user", "content": prompt}]
            processed_answer = await model_api.call_with_priority(messages)
            
            if len(processed_answer) >= 20:
                processed_row['article_answer'] = processed_answer
        except Exception as e:
            print(f"處理標點符號時發生錯誤：{str(e)}")
    
    # 2. 檢查並處理 URL
    if find_url_containing(processed_row['article_question'], processed_row['article_answer']):
        try:
            # 同時處理問題和答案中的 URL（並行處理）
            question_prompt = PromptTemplates.get_url_cleaning_prompt(processed_row['article_question'])
            answer_prompt = PromptTemplates.get_url_cleaning_prompt(processed_row['article_answer'])
            
            question_messages = [{"role": "user", "content": question_prompt}]
            answer_messages = [{"role": "user", "content": answer_prompt}]
            
            # 並行處理兩個請求
            question_task = model_api.call_with_priority(question_messages)
            answer_task = model_api.call_with_priority(answer_messages)
            
            cleaned_question, cleaned_answer = await asyncio.gather(question_task, answer_task)
            
            if len(cleaned_question) >= 5 and len(cleaned_answer) >= 20:
                processed_row['article_question'] = cleaned_question
                processed_row['article_answer'] = cleaned_answer
        except Exception as e:
            print(f"處理 URL 時發生錯誤：{str(e)}")
    
    # 3. 檢查並處理問題重寫
    if find_question_rewrite_candidates(processed_row['article_question'], processed_row['article_answer']):
        try:
            prompt = PromptTemplates.get_system_prompt_for_question_rewrite(
                processed_row['article_question'], processed_row['article_answer']
            )
            messages = [{"role": "user", "content": prompt}]
            rewritten_question = await model_api.call_with_priority(messages)
            
            if len(rewritten_question) >= 5:
                processed_row['article_question'] = rewritten_question
        except Exception as e:
            print(f"處理問題重寫時發生錯誤：{str(e)}")
    
    return processed_row

def create_output_file():
    """創建輸出檔案並返回檔名"""
    current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'data/tweh_items_keywords_processed_v2_{current_date}.csv'
    
    # 創建 CSV 檔案並寫入標題行
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'article_department',
            'article_question',
            'article_answer',
            'article_content_length',
            'article_answer_length'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    return output_filename

def append_to_csv(filename, row_data):
    """將單筆資料追加到 CSV 檔案"""
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'article_department',
            'article_question',
            'article_answer',
            'article_content_length',
            'article_answer_length'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(row_data)

async def post_filter_processing_stream(df, model_api: AsyncModelAPI):
    """執行篩選後處理類（異步串流版本）"""
    print("[篩選後處理] 開始篩選後處理階段（異步串流模式）...")
    
    # 創建輸出檔案
    output_filename = create_output_file()
    
    processed_count = 0
    
    # 異步處理資料
    async def process_batch(batch_rows):
        """處理一批資料"""
        nonlocal processed_count
        
        # 並行處理這批資料
        tasks = []
        for index, row in batch_rows:
            task = process_single_row(row, model_api)
            tasks.append((index, task))
        
        # 等待所有任務完成
        for index, task in tasks:
            try:
                processed_row = await task
                
                # 計算特徵欄位
                processed_row['article_content_length'] = len(processed_row['article_question'])
                processed_row['article_answer_length'] = len(processed_row['article_answer'])
                
                # 最終長度檢查
                if processed_row['article_answer_length'] >= 20:
                    # 準備寫入 CSV 的資料
                    csv_row = {
                        'article_department': processed_row['article_department'],
                        'article_question': processed_row['article_question'],
                        'article_answer': processed_row['article_answer'],
                        'article_content_length': processed_row['article_content_length'],
                        'article_answer_length': processed_row['article_answer_length']
                    }
                    
                    # 寫入 CSV
                    append_to_csv(output_filename, csv_row)
                    processed_count += 1
                    
            except Exception as e:
                print(f"處理索引 {index} 時發生錯誤：{str(e)}")
    
    # 分批處理資料以控制並行度
    batch_size = 10  # 每批處理 10 筆資料
    batch_rows = []
    
    with tqdm(total=len(df), desc="處理資料") as pbar:
        for index, row in df.iterrows():
            batch_rows.append((index, row))
            
            if len(batch_rows) >= batch_size:
                await process_batch(batch_rows)
                pbar.update(len(batch_rows))
                batch_rows = []
        
        # 處理剩餘的資料
        if batch_rows:
            await process_batch(batch_rows)
            pbar.update(len(batch_rows))
    
    print(f"[篩選後處理] 篩選後處理完成，總計處理 {processed_count} 筆")
    print(f"[篩選後處理] 輸出檔案：{output_filename}")
    print("-" * 100)
    
    return output_filename, processed_count

# === 原有的特徵欄位計算與最終檢查函數（現在不需要了，因為已經整合到串流處理中） ===

# === 輸出與儲存 ===
def save_final_data(df):
    """儲存最終資料"""
    print("[儲存] 開始儲存最終資料...")
    
    # 按照 README 需求設定欄位順序
    column_order = [
        'article_department',
        'article_question',
        'article_answer',
        'article_content_length',
        'article_answer_length'
    ]
    
    # 選擇指定欄位並重新排序
    final_df = df[column_order].copy()
    
    # 生成帶有處理版本與日期的檔名
    current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'data/tweh_items_keywords_processed_v2_{current_date}.csv'
    
    # 儲存為 CSV
    final_df.to_csv(output_filename, index=False, encoding='utf-8')
    
    print(f"[儲存] 資料已儲存至：{output_filename}")
    print(f"[儲存] 最終資料筆數：{len(final_df)}")
    print("-" * 100)
    
    return output_filename

# === API 設定 ===
def setup_clients():
    """設定 API 客戶端（異步版本）"""
    client = AsyncOpenAI(
        api_key=os.getenv('OPENAI_API_KEY')
    )
    return client

async def test_api_connections(client: AsyncOpenAI):
    """測試 API 連接（異步版本）"""
    # 測試 OpenAI API 連接
    try:
        await client.models.list()
        print("OpenAI API 連接成功！")
    except Exception as e:
        print(f"OpenAI API 連接測試失敗：{str(e)}")
        return False
    
    # 測試 Ollama 連接
    try:
        await asyncio.to_thread(ollama.chat, model="glm4:9b", messages=[{"role": "user", "content": "test"}])
        print("Ollama API 連接成功！")
    except Exception as e:
        print(f"Ollama API 連接測試失敗：{str(e)}")
        return False
    
    return True

# === 主要執行流程 ===
async def main():
    """主要執行流程（異步版本）"""
    print("=" * 100)
    print("台灣 E 院線上諮詢資料處理系統（異步版本）")
    print("=" * 100)
    
    try:
        # 1. 設定 API 客戶端
        print("[初始化] 設定 API 客戶端...")
        client = setup_clients()
        
        # 2. 測試 API 連接
        print("[初始化] 測試 API 連接...")
        if not await test_api_connections(client):
            print("API 連接測試失敗，程式終止")
            return
        
        # 3. 初始化 AsyncModelAPI
        model_api = AsyncModelAPI(client)
        
        # 4. 讀取原始資料
        df = load_data()
        
        # 5. 篩選類處理
        df = filtering_stage(df)
        
        # 6. 篩選後處理類（異步串流版本）
        output_file, processed_count = await post_filter_processing_stream(df, model_api)
        
        print("=" * 100)
        print("資料處理完成！")
        print(f"輸出檔案：{output_file}")
        print(f"最終資料筆數：{processed_count}")
        print("=" * 100)
        
    except Exception as e:
        print(f"程式執行過程中發生錯誤：{str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
