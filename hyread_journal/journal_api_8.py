from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
load_dotenv() 

import os
import re
import json
import traceback
import logging
import httpx
import asyncio
import time

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict
from elasticsearch import AsyncElasticsearch
from openai import AsyncOpenAI
from typing import List, AsyncGenerator


# 初始化 logger 與 OpenAI client
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("找不到 OPENAI_API_KEY，請確認 .env 設定正確")
openaiclient = AsyncOpenAI(api_key=api_key)

# 建立 APIRouter
router = APIRouter()

# Elasticsearch
es = AsyncElasticsearch("https://service.ebook.hyread.com.tw/es9", request_timeout=100)

INDEX_NAME = "hyread_journalv4"
EMBEDDING_API_URL = "http://35.201.234.34:8888/jina/v3/embedding"

# 重要期刊關鍵字常數定義
PREMIUM_JOURNAL_KEYWORDS = ["A&HCI", "CSSCI", "EconLit", "EI", "MEDLINE", "SCI", "SCIE", "Scopus", "SSCI", "THCI", "THCI CORE", "TSCI", "TSSCI"]

class SearchRequest(BaseModel):
    query: str
    top_k: int = 100
    require_summary: bool = True

# 判斷中英文
def is_chinese(text: str) -> bool:
    return re.search(r'[\u4e00-\u9fff]', text) is not None if len(text) > 0 else False

# 非同步呼叫 Embedding API
async def embed_query(query: str) -> List[float]:
    if not query:
        raise ValueError("Query string is empty.")
    payload = {"input": query, "config": {"task": "retrieval.query", "truncate_dim": 512}}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(EMBEDDING_API_URL, json=payload)
    if resp.status_code != 200:
        logger.error(f"Embedding API error {resp.status_code}: {resp.text}")
        raise RuntimeError(f"Embedding API error: {resp.status_code}")
    return resp.json()["embedding"]

def get_num_candidates(top_k: int) -> int:
    """計算合理的 num_candidates 數量"""
    if top_k >= 100:
        return top_k
    else: 
        return top_k * 2

# 通用搜尋函式
async def execute_phrase_search(
    query: str,
    title_field: str,
    language: str,
    count: int,
    is_premium: bool = False,
    require_summary: bool = True,
    extra_filters: List[Dict] = None
) -> Dict:
    """通用詞組搜尋函數，封裝所有 phrase 搜尋的共同邏輯"""
    try:
        # 建構過濾條件
        filters = []
        
        # 語言過濾
        if language == "chinese":
            filters.append({"exists": {"field": "title_C"}})
            if require_summary:
                filters.append({"exists": {"field": "summary_C"}})
        elif language == "english":
            filters.append({"exists": {"field": "title_E"}})
            if require_summary:
                filters.append({"exists": {"field": "summary_E"}})
        
        # 期刊等級過濾
        if is_premium:
            filters.append({"terms": {"keyword": PREMIUM_JOURNAL_KEYWORDS}})
        
        # 額外過濾條件
        if extra_filters:
            filters.extend(extra_filters)
        
        # 建構查詢主體
        body = {
            "query": {
                "bool": {
                    "must": [{"match_phrase": {title_field: {"query": query}}}],
                    "filter": filters
                }
            },
            "size": count
        }
        
        # 執行搜尋
        response = await es.search(index=INDEX_NAME, body=body)
        return response
        
    except Exception as e:
        journal_type = "重要期刊" if is_premium else "一般期刊"
        logger.error(f"{journal_type}詞組搜尋失敗: {e}")
        return {"hits": {"hits": []}}

async def execute_dual_knn_search(
    query_vector: list,
    language: str,
    count: int,
    is_premium: bool = False,
    require_summary: bool = True,
    extra_filters: List[Dict] = None
) -> List[Dict]:
    """通用雙語言向量搜尋函數，封裝所有 dual-knn 搜尋的共同邏輯"""
    try:
        # 設定向量欄位和摘要過濾條件
        if language == "chinese":
            # knn query (field: chinese_vector, filter: 有中文摘要)
            primary_field = "chinese_vector"
            primary_summary_filter = {"exists": {"field": "summary_C"}}
            # knn query (field: english_vector, filter: 無中文摘要但有英文摘要)
            secondary_field = "english_vector"
            secondary_summary_filter = {
                "bool": {
                    "must": [{"exists": {"field": "summary_E"}}],
                    "must_not": [{"exists": {"field": "summary_C"}}]
                }
            }
        else:
            # knn query (field: english_vector, filter: 有英文摘要)
            primary_field = "english_vector"
            primary_summary_filter = {"exists": {"field": "summary_E"}}
            # knn query (field: chinese_vector, filter: 無英文摘要但有中文摘要)
            secondary_field = "chinese_vector"
            secondary_summary_filter = {
                "bool": {
                    "must": [{"exists": {"field": "summary_C"}}],
                    "must_not": [{"exists": {"field": "summary_E"}}]
                }
            }

        # 建構基礎過濾條件
        base_filters = []
        
        # 期刊等級過濾
        if is_premium:
            base_filters.append({"terms": {"keyword": PREMIUM_JOURNAL_KEYWORDS}})
        
        # 額外過濾條件
        if extra_filters:
            base_filters.extend(extra_filters)

        # 準備兩個搜尋的過濾條件
        primary_filters = base_filters + [primary_summary_filter] if require_summary else base_filters
        secondary_filters = base_filters + [secondary_summary_filter] if require_summary else base_filters

        # 建構查詢主體
        primary_body = {
            "query": {
                "knn": {
                    "field": primary_field,
                    "query_vector": query_vector,
                    "k": count,
                    "num_candidates": get_num_candidates(count),
                    "filter": primary_filters
                }
            },
            "size": count
        }

        secondary_body = {
            "query": {
                "knn": {
                    "field": secondary_field,
                    "query_vector": query_vector,
                    "k": count,
                    "num_candidates": get_num_candidates(count),
                    "filter": secondary_filters
                }
            },
            "size": count
        }

        # 並行執行兩個搜尋
        primary_response, secondary_response = await asyncio.gather(
            es.search(index=INDEX_NAME, body=primary_body),
            es.search(index=INDEX_NAME, body=secondary_body),
            return_exceptions=True
        )

        # 統一處理所有回應
        journal_type = "重要期刊" if is_premium else "一般期刊"
        responses = [primary_response, secondary_response]
        results_map = {}

        for response in responses:
            # 處理異常
            if isinstance(response, Exception):
                logger.error(f"{journal_type}向量搜尋失敗: {response}")
                continue
            
            # 處理正常結果
            if "hits" in response and "hits" in response["hits"]:
                for hit in response["hits"]["hits"]:
                    sysid = hit["_source"].get("sysid")
                    if sysid:
                        if sysid in results_map:
                            # 累加分數
                            results_map[sysid]["_score"] += hit.get("_score", 0.0)
                        else:
                            # 新結果
                            results_map[sysid] = hit

        # 轉換為列表並按分數排序
        final_results = list(results_map.values())
        final_results.sort(key=lambda x: x.get("_score", 0), reverse=True)
        
        # 返回前 count 筆結果
        return final_results[:count]

    except Exception as e:
        journal_type = "重要期刊" if is_premium else "一般期刊"
        logger.error(f"{journal_type}雙語言向量搜尋失敗: {e}")
        return []

async def execute_four_layer_search(query: str, query_vec: list, title_field: str, top_k: int, language: str = None, require_summary: bool = True, extra_filters: List[Dict] = None):
    """
    四層搜尋架構：使用兩個通用搜尋函式
    1. 重要期刊 match_phrase 搜尋（5篇）
    2. 重要期刊 雙語言knn 搜尋（10篇，排除重複）
    3. 一般期刊 match_phrase 搜尋（top_k//2 篇，排除重複）
    4. 一般期刊 雙語言knn 搜尋（top_k 篇，排除重複）
    """
    
    try:
        # 使用 asyncio.gather 並行執行四個搜尋
        responses = await asyncio.gather(
            # 重要期刊 match_phrase 搜尋，5 篇
            execute_phrase_search(query, title_field, language, 5, is_premium=True, require_summary=require_summary, extra_filters=extra_filters),
            # 重要期刊 雙語言knn 搜尋，10 篇
            execute_dual_knn_search(query_vec, language, 10, is_premium=True, require_summary=require_summary, extra_filters=extra_filters),
            # 一般期刊 match_phrase 搜尋，top_k // 2 篇
            execute_phrase_search(query, title_field, language, top_k // 2, is_premium=False, require_summary=require_summary, extra_filters=extra_filters),
            # 一般期刊 雙語言knn 搜尋，top_k 篇
            execute_dual_knn_search(query_vec, language, top_k, is_premium=False, require_summary=require_summary, extra_filters=extra_filters)
        )
        
        # 處理結果，確保返回格式一致
        processed_responses = []
        for i, response in enumerate(responses):
            if i in [1, 3]:  # knn搜尋結果需要包裝成標準格式
                processed_responses.append({"hits": {"hits": response}})
            else:
                processed_responses.append(response)
        
        return processed_responses
        
    except Exception as e:
        logger.error(f"四層搜尋整體失敗: {e}")
        # 返回空結果格式，保持與原來一致
        return [{"hits": {"hits": []}} for _ in range(4)]

def process_search_responses(responses: list, title_field: str, summary_field: str, include_extra_fields: bool = False, top_k: int = None) -> list:
    """結果處理函數，支援雙語言向量搜尋的結果標記，並加入標題回退邏輯"""
    final_results = []
    all_sysids = set()
    
    # 定義搜尋來源標記
    search_sources = ["premium_phrase", "premium_dual_knn", "general_phrase", "general_dual_knn"]
    
    # 確定回退欄位
    fallback_title_field = "title_E" if title_field == "title_C" else "title_C"
    fallback_summary_field = "summary_E" if summary_field == "summary_C" else "summary_C"

    # 遍歷每個搜尋結果
    for response_idx, response in enumerate(responses):
        if "hits" in response and "hits" in response["hits"]:
            for hit in response["hits"]["hits"]:
                sysid = hit["_source"].get("sysid", "N/A")
                if sysid not in all_sysids:
                    # 標題處理：優先使用目標語言，回退到另一語言
                    title = hit["_source"].get(title_field)
                    if not title or title.strip() == "":
                        title = hit["_source"].get(fallback_title_field, "N/A")
                    
                    # 摘要處理：同樣邏輯
                    summary = hit["_source"].get(summary_field)
                    if not summary or summary.strip() == "":
                        summary = hit["_source"].get(fallback_summary_field, "N/A")
                    
                    result = {
                        "title": title,
                        "keyword": hit["_source"].get("keyword", []),
                        "summary": summary,
                        "sysid": sysid,
                        "search_source": search_sources[response_idx],
                        # "score": hit.get("_score", 0.0)
                    }
                    
                    # 為雙語言knn搜尋添加向量搜尋來源信息
                    if response_idx in [1, 3] and "_search_source" in hit:
                        result["vector_source"] = hit["_search_source"]
                    
                    # 為 AI 搜尋添加額外欄位
                    if include_extra_fields:
                        result.update({
                            "date": hit["_source"].get("publishdate", "N/A"),
                            "authors": hit["_source"].get("author", []),
                            "journal": hit["_source"].get("j_name", "N/A")
                        })
                    
                    final_results.append(result)
                    all_sysids.add(sysid)
    
    # 確定截斷結果到 top_k 筆
    if top_k is not None and len(final_results) > top_k:
        final_results = final_results[:top_k]
    
    return final_results

# 搜尋端點
@router.post("/search")
async def search_articles(req: SearchRequest):
    """搜尋端點，使用四個獨立搜尋函式"""
    query, top_k = req.query.strip(), req.top_k
    is_chinese_text = is_chinese(query)
    title_field   = "title_C"      if is_chinese_text else "title_E"
    summary_field = "summary_C"    if is_chinese_text else "summary_E"
    language = "chinese" if is_chinese_text else "english"

    # 獲取查詢向量
    query_vec = await embed_query(query)
    
    # 使用新的四層搜尋邏輯
    responses = await execute_four_layer_search(query, query_vec, title_field, top_k, language, req.require_summary)
    
    # 使用新的結果處理函數
    final_results = process_search_responses(responses, title_field, summary_field, top_k=top_k)

    return {"language": "Chinese" if is_chinese_text else "English", "count": len(final_results), "results": final_results}


class AISearchRequest(BaseModel):
    query: str
    top_k: int = 100
    show_ai_params: bool = False
    require_summary: bool = True

# openai 用 JSON Schema 
SEARCH_PARAMETERS = {
  "name": "search_parameters_schema",
  "schema": {
    "type": "object",
    "properties": {
      "language": {
        "type": "string",
        "enum": [
          "chinese",
          "english"
        ],
        "description": "查詢語言：chinese 中文 或 english 英文"
      },
      "optimized_query": {
        "type": "string",
        "description": "優化後的搜尋，使用與輸入相同的語言，使用自然語言查詢，去掉時間、作者、期刊等細節，將縮寫詞與口語化處理為完整用語",
      },
      "authors": {
        "type": "array",
        "description": "相關作者名稱列表",
        "items": {
          "type": "string"
        }
      },
      "publishdate_range": {
        "type": "object",
        "description": "發表時間範圍，包含 start_date 和 end_date (YYYYMM格式)",
        "properties": {
          "start_date": {
            "type": "string"
          },
          "end_date": {
            "type": "string"
          }
        },
        "required": [
          "start_date",
          "end_date"
        ],
        "additionalProperties": False
      },
      "journal_names": {
        "type": "array",
        "description": "相關期刊名稱列表",
        "items": {
          "type": "string"
        }
      }
    },
    "required": [
      "language",
      "optimized_query",
      "authors",
      "publishdate_range",
      "journal_names"
    ],
    "additionalProperties": False
  },
  "strict": True
}

# AI 生成 query 函數
async def ai_generate_search(query: str) -> Dict:
    """使用 GPT Structured Output 生成 Elasticsearch 搜尋參數"""
    
    system_prompt = """
        你是 Hyread Journal 台灣全文資料庫的你是學術搜尋優化專家。
        請分析用戶的查詢意圖，並生成 Elasticsearch 搜尋參數。

        ## 核心任務：
        1. 分析查詢語言（chinese/english）
        2. 生成優化的自然語言查詢（optimized_query）
        3. 提取時間、作者、期刊等過濾條件

        ## 語言判斷：
        - 主要中文字符 → "chinese"
        - 主要英文字符 → "english"
        - optimized_query 必須使用相同語言

        ## optimized_query 優化策略（按順序執行）：
        1. **（重要）縮寫詞轉換（必須執行）**
        - **將日常縮寫和專業術語轉換為完整形式**
        - 都更 → 都市更新
        - 央行 → 中央銀行  
        - 立委 → 立法委員
        - 食安 → 食品安全
        - AI → 人工智慧
        - ML → 機器學習

        2. 移除過濾資訊
        - 保持自然語言形式
        - 去除時間、作者、期刊等將用於過濾的資訊，只留下核心搜尋主題        
       
        ### 範例：
        * 輸入：「2023年張三發表的COVID-19疫苗研究」→ 優化為：「COVID-19疫苗研究」（去掉時間、作者）
        * 輸入：「2020年後科技管理學刊對人工智慧的研究」→ 優化為：「人工智慧」（去掉時間、期刊）
        * 輸入：「recent AI applications in healthcare by MIT」→ 優化為：「AI applications in healthcare」（去掉作者、時間）
        * 輸入：「都更政策對台灣房價的影響→ 優化為：「都市更新政策對台灣房地產價格的影響」（將縮寫轉換成完整形式）

        ## 時間範圍處理（格式：YYYYMM）：
        - 具體年份：如「2023年」→ {"start_date": "202301", "end_date": "202312"}
        - 時間段：如「2020-2023」→ {"start_date": "202001", "end_date": "202312"}
        - 開放式：如「2020年以後」→ {"start_date": "202001", "end_date": ""}
        - 無時間資訊 → {"start_date": "", "end_date": ""}
        - 若被問到「近幾年」等，以現在的時間（2025年）為基準去推，如「近五年」→ {"start_date": "202101", "end_date": "202512"}

        ## 實體提取規則：
        - authors：只提取明確提到的作者姓名，無則為空陣列 []
        - journal_names：只提取明確提到的期刊名稱，無則為空陣列 []
        - 不要過度推測或擴展

        ## 重要原則：
        1. optimized_query 應該是流暢的自然語言，而非關鍵字組合
        2. 保持查詢的學術性和專業性
        3. 空值處理：無時間用空字串，無作者/期刊用空陣列
        4. 縮寫詞與口語化處理（必須執行）
        """
    
    # 檢查是否為空查詢    
    try:
        gpt_start = time.time()
        completion = await openaiclient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"請分析這個查詢，生成 Elasticsearch 搜尋參數：{query}"}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": SEARCH_PARAMETERS
            }
        )
        gpt_end = time.time()
        logger.info(f"AI 查詢分析耗時: {gpt_end - gpt_start:.2f} 秒")
        return json.loads(completion.choices[0].message.content)
    
    except Exception as e:
        logger.error(f"AI 查詢分析失敗: {e}")
        # 回退到基本分析
        is_chinese_text = is_chinese(query)
        return {
            "language": "chinese" if is_chinese_text else "english",
            "optimized_query": query,
            "authors": None,
            "publishdate_range": None,
            "journal_names": None
        }

# 接收 ai 回應並解析並執行搜尋
async def execute_ai_search(ai_params: Dict, top_k: int, require_summary: bool = True) -> List[Dict]:
    """執行 AI 增強搜尋，使用四個獨立搜尋函式"""
    
    # 根據 AI 分析的語言設定欄位
    is_chinese_language = ai_params.get("language") == "chinese"
    title_field = "title_C" if is_chinese_language else "title_E"
    summary_field = "summary_C" if is_chinese_language else "summary_E"
    
    # 獲取查詢向量
    query_vec = await embed_query(ai_params.get("optimized_query", ""))
    
    # 建構額外過濾條件
    extra_filters = []

    # 時間範圍過濾
    if ai_params.get("publishdate_range"):
        date_range = ai_params["publishdate_range"]
        range_filter = {}
        if date_range.get("start_date"):
            range_filter["gte"] = date_range["start_date"]
        if date_range.get("end_date"):
            range_filter["lte"] = date_range["end_date"]
        if range_filter:
            extra_filters.append({"range": {"publishdate": range_filter}})

    # 作者過濾
    if ai_params.get("authors"):
        extra_filters.append({"terms": {"author": ai_params["authors"]}})

    # 期刊過濾
    if ai_params.get("journal_names"):
        journal_conditions = []
        for journal_name in ai_params["journal_names"]:
            journal_conditions.append({"match_phrase": {"j_name": journal_name}})
        
        extra_filters.append({
            "bool": {
                "should": journal_conditions,
                "minimum_should_match": 1
            }
        })
    
    # 執行四層搜尋
    responses = await execute_four_layer_search(ai_params.get("optimized_query", ""), query_vec, title_field, top_k, ai_params.get("language"), require_summary, extra_filters)
    
    # 使用結果處理函數
    return process_search_responses(responses, title_field, summary_field, include_extra_fields=True, top_k=top_k)


# AI 搜尋端點
@router.post("/ai_search")
async def ai_search_articles(req: AISearchRequest):
    """AI 增強搜尋，使用四個獨立搜尋函式"""
    try:
        # 1. AI 生成查詢
        ai_params = await ai_generate_search(req.query)
        logger.info(f"AI 生成結果: {ai_params}")
        
        # 2. 執行 AI 增強搜尋
        results = await execute_ai_search(ai_params, req.top_k, req.require_summary)
        
        # 3. 組織回應
        response = {}
        
        # 4. 可選顯示 AI 參數
        if req.show_ai_params:
            response["ai_params"] = ai_params
            
        response.update({
            "language": "English" if ai_params.get("language") == "english" else "Chinese",
            "count": len(results),
            "results": results
        })
            
        return response
        
    except Exception as e:
        logger.error(f"AI 搜尋失敗: {e}")
        logger.debug(traceback.format_exc())
        # 回退到基本搜尋
        fallback_result = await search_articles(SearchRequest(query=req.query, top_k=req.top_k))
        fallback_result["ai_params"] = {"error": "AI 分析失敗，使用基本搜尋"} if req.show_ai_params else None
        return fallback_result
    
async def stream_rag_summary(req: AISearchRequest) -> AsyncGenerator[str, None]:
    try:
        # 1. 搜尋文獻
        search_res = await ai_search_articles(req)
        results = search_res["results"]
        
        optimized_query = search_res.get("ai_params", {}).get("optimized_query") or req.query

        # 2. 回傳參考文獻 metadata
        references = [{"sysid": item["sysid"], "title": item["title"]} for item in results]
        yield json.dumps({"references": references}) + "\n"

        # 為每篇論文建立 summary 任務（每篇一個 GPT）
        async def summarize_tabel(item):
            messages = [
                {"role": "system", "content": (
                    "你是 Hyread Journal 台灣全能資料庫的文獻分析 AI。請模仿專家分析風格，針對使用者提出的研究問題，"
                    "從以下期刊內容中提取針對性的洞察。\n"
                    "你要提供**詳細論文分析表格**：針對用戶問題深度解析每篇論文的具體貢獻\n\n"
                    "## 表格要求 ##\n"
                    "| 論文標題 | 論文觀點 | 研究方法/限制 \n"
                    "不需要回傳標頭\n"
                    "- **核心洞察欄位**：必須針對用戶具體問題，從該論文中提取最相關的發現、數據、結論或觀點\n"
                    "- 提供具體的研究證據、數據或引述\n"
                    "- 每個洞察應該獨特且對研究人員有實用價值\n"
                    "- 避免泛泛而談，要有針對性和深度"
                )},
                {"role": "user", "content": (
                    f"## 研究問題 ##\n{optimized_query}\n\n"
                    f"## 論文標題 ##\n{item['title']}\n\n"
                    f"## 論文摘要 ##\n{item['summary']}\n\n"
                    f"請你輸出一行 Markdown 表格格式，內容分成三欄："
                    f"論文標題 | 論文觀點 | 研究方法/限制\n"
                    f"請確保整行是用 `|` 分隔的單行格式，不要輸出任何標頭，也不要額外換行或文字。"
                )}
            ]
            response = await openaiclient.chat.completions.create(
                model="gpt-4o-mini", messages=messages, stream=False
            )
            return response.choices[0].message.content.strip()
        
            # 3.啟動所有 GPT 表格摘要任務（不馬上等結果）
        summary_task = asyncio.gather(*[summarize_tabel(item) for item in results])


            # 4. 綜合摘要
        contexts = []
        
        for item in results:
            contexts.append(f"論文標題：{item['title']}\n摘要：{item['summary']}")
            
        joined_ctx = "\n\n".join(contexts)

        integration_prompt = [
            {"role": "system", "content": (
                "你是 Hyread Journal 的智慧分析助手，請根據以下多篇論文的核心洞察，"
                "撰寫一段整合性摘要（2-3段），點出關鍵趨勢、研究貢獻與研究缺口。請避免重複摘要內容，要有洞察力與總結性觀點。"
                "## 摘要要求 ##\n"
                "- 保持簡潔，避免冗長描述\n"
                "- 重點突出領域趨勢與研究缺口\n"
                "- 為後續表格分析提供背景脈絡\n\n"
            )},
            {"role": "user", "content": (
                f"## 研究問題 ##\n{optimized_query}\n\n"
                f"## 文獻資料 ##\n{joined_ctx}\n\n"
                f"請針對上述研究問題，運用您的專業知識對這些文獻進行深度分析。"
                f"特別關注：研究趨勢、方法學差異、核心貢獻、研究空白，以及對未來研究的啟示。"
            )}
        ]

        stream = await openaiclient.chat.completions.create(
            model="gpt-4o-mini", messages=integration_prompt, stream=True
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield json.dumps({"summary": delta}) + "\n"

        # 同時發送所有 GPT 任務
        summaries = await summary_task

        # 5. 回傳表格表頭
        yield json.dumps({"table_row": "| 論文標題 | 論文觀點 | 研究方法/限制 |"}) + "\n"
        yield json.dumps({"table_row": "|-----------|---------------------|------------------|"}) + "\n"

        # 每一條 GPT 表格摘要逐一送出
        for row in summaries:
            # 避免亂格式的 pipe
            clean_row = row.replace("\n", " ").replace("|", "\\|") if row.count("|") < 2 else row
            yield json.dumps({"table_row": clean_row.strip()}) + "\n"

        

        # 6. 結尾標記
        yield json.dumps({"done": True}) + "\n"

    except Exception as e:
        logger.error(f"RAG summary 失敗: {e}")
        logger.debug(traceback.format_exc())
        yield json.dumps({"summary": "抱歉，無法生成文獻總結，請稍後再試。"}) + "\n"
        yield json.dumps({"done": True}) + "\n"

@router.post("/rag")
async def rag_endpoint(req: AISearchRequest):
    """
    RAG 文獻總結：
    1) 呼叫 /search 拿到前 k 篇 *包含 sysid 的* title+summary
    2) 回傳 references（sysid 與 title），再 streaming 回 summary
    """
    print(req)
    
    return StreamingResponse(
        stream_rag_summary(req),
        media_type="application/x-ndjson"
    )
