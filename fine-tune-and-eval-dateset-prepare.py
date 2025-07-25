import pandas as pd
import numpy as np

df = pd.read_csv('data/tweh_items_keywords_processed_v2_20250718_035346.csv')

def sampling_eval_data(df: pd.DataFrame):
    """
    從資料集中抽取驗證資料集和微調資料集
    
    參數:
    df (DataFrame): 原始資料集
    """
    # 初始化驗證資料集 DataFrame
    eval_df_list = []
    
    # 取得前 10 個最多資料的部門
    top_10_departments = df['article_department'].value_counts().index[:10]
    
    # 針對每個部門抽樣
    for department in top_10_departments:
        # 取得該部門的資料
        department_df = df[df['article_department'] == department]
        
        # 計算該部門回答長度的中位數
        median_length = department_df['article_answer_length'].median()
        
        # 篩選出大於中位數長度的資料
        filtered_df = department_df[department_df['article_answer_length'] > median_length]
        
        # 隨機抽取 10 筆資料
        sampled_df = filtered_df.sample(n=10, random_state=42)
        
        # 加入驗證資料集列表
        eval_df_list.append(sampled_df)
    
    # 合併所有驗證資料
    final_eval_df = pd.concat(eval_df_list)
    
    # 儲存驗證資料集
    final_eval_df.to_csv('data/tweh_items_keywords_fine_tune_eval.csv', index=False)
    
    # 從原始資料集中移除驗證資料集的資料
    fine_tune_df = df[~df.index.isin(final_eval_df.index)]
    
    # 儲存微調資料集
    fine_tune_df.to_csv('data/tweh_items_keywords_fine_tune.csv', index=False)


def save_data(df: pd.DataFrame, file_name: str):
    """
    儲存資料集
    """
    df.to_csv(f'data/{file_name}.csv', index=False)

sampling_eval_data(df)


    
 
