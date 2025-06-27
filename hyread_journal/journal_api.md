# Journal API 業務邏輯文檔

## 系統概述
Hyread Journal 台灣全文資料庫搜尋API，提供學術期刊文章的智慧搜尋功能。

### 1. 四層搜尋策略
系統採用四層搜尋架構，確保重要期刊優先，最後回傳去除重複，截取 top_k 筆：

```
1. 重要期刊 + 標題精確搜尋 (5篇)
2. 重要期刊 + 語義向量搜尋 (10篇)  
3. 一般期刊 + 標題精確搜尋 (top_k/2篇)
4. 一般期刊 + 語義向量搜尋 (top_k篇)
```

**重要期刊定義**：包含 A&HCI, CSSCI, EconLit, EI, MEDLINE, SCI, SCIE, Scopus, SSCI, THCI, THCI CORE, TSCI, TSSCI 等指標期刊

### 2. 雙語支援機制
- **語言偵測**：檢查是否包含中文字符（\u4e00-\u9fff），包含則判定為中文，否則為英文
- **欄位 mapping**：
  - 中文：`title_C`, `summary_C`, `chinese_vector`
  - 英文：`title_E`, `summary_E`, `english_vector`

### 3. AI 增強搜尋
使用 GPT-4o-mini 生成 Structual Output 搜尋 query：
- **query 優化**：縮寫詞轉換（如：都更→都市更新，AI→人工智慧）
- **實體提取**：
  - __時間範圍過濾__：如「2023年」、「2020-2024」、「2022年以後」
  - __作者過濾__：識別查詢中的作者姓名
  - __期刊過濾__：支援期刊名稱精確匹配

## API端點說明

### `/search` - 基本搜尋
**用途**：標準的學術文章搜尋
**輸入**：
```json
{
  "query": "人工智慧在教育的應用",
  "top_k": 100,
  "require_summary": true
}
```

**流程**：
1. 語言偵測 → 2. 查詢向量化 → 3. 四層搜尋 → 4. 去重排序

### `/ai_search` - AI增強搜尋
**用途**：智慧查詢分析，支援複雜查詢意圖
**輸入**：
```json
{
  "query": "2020年後台灣遠距教學政策對教育品質影響的研究",
  "top_k": 100,
  "show_ai_params": true,
  "require_summary": true
}
```

**流程**：
1. GPT分析查詢意圖 → 2. 提取過濾條件 → 3. 優化搜尋詞 → 4. 執行四層搜尋

## 關鍵函數解析

### 1. 基礎工具函數

#### `get_num_candidates()`
- **作用**：計算 Elasticsearch KNN 搜尋的候選數量
- **邏輯**：top_k ≥ 100 時直接使用 top_k，否則 top_k × 2
- **目的**：平衡搜尋品質與效能

### 2. 搜尋核心函數

#### `execute_phrase_search()`
- **作用**：通用詞組搜尋函數，封裝所有 phrase 搜尋的共同邏輯
- **參數**：
  - `query`: 搜尋查詢字串
  - `title_field`: 標題欄位（title_C 或 title_E）
  - `language`: 語言設定（chinese 或 english）
  - `count`: 回傳結果數量
  - `is_premium`: 是否為重要期刊搜尋
  - `require_summary`: 是否需要摘要
  - `extra_filters`: 額外過濾條件（AI搜尋用）
- **過濾邏輯**：
  - 語言過濾（確保對應標題欄位存在）
  - 摘要過濾（根據 require_summary 和語言決定）
  - 期刊等級過濾（重要期刊關鍵字匹配）
  - 額外過濾條件（時間、作者、期刊名稱）

#### `execute_dual_knn_search()`
- **作用**：通用雙語言向量搜尋函數，實現雙語言搜尋策略
- **雙搜尋策略**：（中文查詢為例）
  - **主要搜尋**：搜尋 chinese_vector（有中文摘要）
  - **次要搜尋**：搜尋 english_vector（無中文摘要但有英文摘要）
- **分數處理**：合併結果，對相同 sysid 的結果進行分數累加

### 3. 核心搜尋策略

#### `execute_four_layer_search()`
- **作用**：執行四層並行搜尋策略
- **四層架構**：
  1. 重要期刊 + 精確搜尋（5篇）
  2. 重要期刊 + 向量搜尋（10篇）
  3. 一般期刊 + 精確搜尋（top_k/2篇）
  4. 一般期刊 + 向量搜尋（top_k篇）
- **優化**：使用 `asyncio.gather` 並行執行

#### `process_search_responses()`
- **作用**：處理四層搜尋結果，提供結果整合
- **核心功能**：
  - 按搜尋順序合併結果（重要期刊優先）
  - 基於 `sysid` 去除重複文章
  - 添加搜尋來源標記，便於結果分析
  - 限制最終結果數量至 top_k

### 4. AI增強搜尋

#### `ai_generate_search()`
- **作用**：使用 GPT-4o-mini 分析查詢意圖
- **核心功能**：
  - 縮寫詞轉換（都更→都市更新，AI→人工智慧）
  - 查詢語言識別
  - 實體提取（作者、期刊、時間）
  - 查詢優化（去除過濾資訊，保留核心主題）
- **JSON Schema**：使用 Structured Output 確保格式正確
- **回退機制**：AI失敗時使用基本語言判斷

#### `execute_ai_search()`
- **作用**：執行 AI 增強搜尋
- **流程**：
  1. 根據 AI 分析結果設定搜尋欄位
  2. 建構額外過濾條件（時間、作者、期刊）
  3. 呼叫四層搜尋 `execute_four_layer_search()`
  4. 回傳包含額外欄位的結果
- **擴充資訊**：date, authors, journal 等額外欄位


### `/rag`- RAG 文獻摘要系統

### 功能簡介

文獻搜尋與摘要整合工具，基於 RAG，使用者輸入一個研究問題，將執行：

1. AI增強搜尋，搜尋符合的前 `k` 篇文獻（含 `title` 與 `summary`）。
2. 即時回傳參考文獻清單。
3. 整合所有摘要內容，撰寫跨文獻的總結洞察。
4. 針對每篇文獻呼叫 GPT 生成表格摘要（逐一串流回傳）。


---
### 使用模型

* 表格摘要生成：`gpt-4o-mini`（非 streaming）
* 整合性摘要：`gpt-4o-mini`（streaming）

---

### 請求格式

```json
POST /rag

{
  "query": "ChatGPT 對高中生英文寫作的幫助",
  "top_k": 100,
  "show_ai_params": false,
  "require_summary": true
}
```

### 請求參數說明

| 欄位名稱              | 型別     | 預設值     | 說明                            |
| ----------------- | ------ | ------- | ----------------------------- |
| `query`           | `str`  | 無       | 使用者的研究問題或查詢                   |
| `top_k`           | `int`  | `100`   | 搜尋文獻的最大數量                     |
| `show_ai_params`  | `bool` | `false` | 是否回傳 AI 分析用的參數（例如最佳化後的 query） |
| `require_summary` | `bool` | `true`  | 是否需要摘要（否則只搜尋）                 |

---

## 處理邏輯（`stream_rag_summary`）

1. **檢索文章**

   * 呼叫 `ai_search_articles()` 取得前 `k` 篇文章（包含 `title`, `summary`, `sysid`）。
   * 如果有 `ai_params.optimized_query`，則優先使用該 query 作為後續摘要依據。

2. **回傳參考文獻清單**
   * 從`ai_search_articles()`回傳的結果抓取`sysid`, `title`
   * 參考資料回傳格式如下：

     ```json
     {"references": [{"sysid": "...", "title": "..."}, ...]}
     ```

3. **建立 GPT 表格摘要任務** `summarize_tabel(item)`

   * 每篇文獻建立一個 GPT 任務，輸出 1 行 Markdown 表格（無標頭）。
   * 要求 AI 針對使用者問題提取核心洞察、研究方法、研究限制。
   * 任務使用 `asyncio.gather` 並行執行，不會立刻等待結果。

4. **整合摘要任務（summary analysis）**

   * 把所有文獻的標題與摘要，送入 GPT 撰寫一段跨文獻的整體趨勢分析。
   * 此部分使用 OpenAI 的 streaming 接口，一段段推送 summary。
   * 串流格式如下：

     ```json
     {"summary": "<summary content>"}
     ```

5. **表格摘要回傳**

   * 等待所有 `summarize_table()` 任務完成後，依序回傳每篇文獻的 GPT 表格摘要。
   * 回傳格式為 NDJSON，每行為一欄表格 row：

     ```json
     {"table_row": "| 論文標題 | 針對問題的核心洞察 | 研究方法/限制 |"}
     {"table_row": "|-----------|---------------------|------------------|"}
     {"table_row": "xxx | xxx | xxx"}
     ```

6. **串流結尾**

   * 串流最後回傳：

     ```json
     {"done": true}
     ```

7. **錯誤處理**

   * 回傳錯誤訊息與終止標記：

     ```json
     {"summary": "抱歉，無法生成文獻總結，請稍後再試。"}
     {"done": true}
     ```

---

### 回傳格式（Streaming NDJSON）

每一筆為一行 JSON：

```json
{"references": [{"sysid": "123", "title": "xxx"}, ...]}
{"summary": "第一段..." }
{"summary": "第二段..." }
{"table_row": "| 論文標題 | 針對問題的核心洞察 | 研究方法/限制 |"}
{"table_row": "|-----------|---------------------|------------------|"}
{"table_row": "xxx | xxx | xxx"}
{"done": true}
```

---




## 配置與部署

### 環境變數
```bash
OPENAI_API_KEY=your_openai_api_key
```

### 外部依賴
- **Elasticsearch**：`https://service.ebook.hyread.com.tw/es9`
- **Embedding API**：`http://35.201.234.34:8888/jina/v3/embedding`
- **索引名稱**：`hyread_journalv4`

## 故障排除

### 常見問題
1. **搜尋無結果**
   - 檢查Elasticsearch連線狀態
   - 確認索引 `hyread_journalv4` 是否存在
   - 檢查查詢語言是否正確偵測

2. **AI搜尋失敗**
   - 檢查 `OPENAI_API_KEY` 是否正確設定
   - 確認GPT API配額是否充足
   - 系統會自動回退到基本搜尋

3. **向量搜尋錯誤**
   - 檢查 Jina Embedding API 連線
   - 確認 `EMBEDDING_API_URL` 可正常存取

---
## 基本搜尋 (/search) 測試查詢

### 關鍵字
- COVID-19 疫苗
- 永續發展
- 人工智慧
- 癌症研究
- 5G
- 半導體
- 數位轉型

### 自然語言
- 台灣高齡化社會對照護政策的挑戰與策略
- 氣候變遷對台灣水資源的影響
- 糖尿病患者飲食與生活的研究
- 智慧城市中的物聯網技術
- 台灣新創企業的資金來源與挑戰
- 偏鄉醫療資源分配不均的改善方案

### 範例：
```json
{
  "query": "氣候變遷對台灣水資源的影響",
  "top_k": 100,
  "require_summary": true
}
```
| 標題 | 主要關鍵字 | 來源 |
|------|------------|------|
| [以高解析度大氣環流模式資料推估氣候變遷下北部水資源之衝擊](https://www.hyread.com.tw/hyreadnew/search_detail_new.jsp?sysid=00653321) | 氣候變遷, 供水系統, 系統動力模式, TSCI | premium_knn |
| [以荷蘭創新水資源管理系統為個案探討台灣因應氣候變遷之技術革新及政策規劃方向](https://www.hyread.com.tw/hyreadnew/search_detail_new.jsp?sysid=00286176) | 氣候變遷, 創新水資源管理系統, 技術革新, 政策規劃 | premium_knn |
| [以TCCIP AR6統計降尺度日資料探討臺灣未來水資源衝擊](https://www.hyread.com.tw/hyreadnew/search_detail_new.jsp?sysid=00660576) | TSCI |  premium_knn |


## AI 增強搜尋 (/ai_search) 測試查詢

- 都更政策對房價的影響（與基本搜尋結果有差異，會將縮寫詞轉換）
- 2020年後台灣遠距教學政策對教育品質影響的研究
- 近五年來 fintech 發展對傳統銀行業的衝擊
- 經濟論文叢刊上對金融海嘯的研究
- 張東浩在癌症上的研究
- 2021年後教育研究月刊對 AI 應用的研究
- 2010-2020 年王志弘在台灣社會研究上針對台北市的研究

### 範例
```json
{
  "query": "2010-2020 年王志弘在台灣社會研究上針對台北市的研究",
  "top_k": 100,
  "show_ai_params": true,
  "require_summary": true
}
```
**AI優化參數：** 
```json
"ai_params": {
  "language": "chinese",
  "optimized_query": "台北市的研究",
  "authors": [
    "王志弘"
  ],
  "publishdate_range": {
    "start_date": "201001",
    "end_date": "202012"
  },
  "journal_names": [
    "台灣社會研究"
  ]
},
```

| 標題 | 主要關鍵字 | 發表日期 | 作者  | 來源 |
|------|------------|----------|------|------|
| [台北市人行空間治理與徒步移動性](https://www.hyread.com.tw/hyreadnew/search_detail_new.jsp?sysid=00361404) | 移動, 步行, 人行空間, 都市治理, 治理術, 台北 | 201209 | 王志弘  | premium_knn |
| [都市自然的治理與轉化新北市二重疏洪道](https://www.hyread.com.tw/hyreadnew/search_detail_new.jsp?sysid=00340579) | 自然, 自然治理, 疏洪道, 綠地, 水岸 | 201309 | 王志弘, 林純秀 | premium_knn |

### 弱點：

1. 會把日期全部當成 publish date，不一定跟研究的內容有關
 - 如 "2019年以來台灣觀光業發展" 會找到 2019 年後的研究，但研究的可能是幾十年前的觀光業
2. embedding model 不太懂口語化、縮寫等，希望以 LLM 搜尋完整化縮寫。但在 prompt 提供範例會有長度問題。

## 有問題的資料
(關鍵字中有亂碼)(https://www.hyread.com.tw/hyreadnew/search_detail_new.jsp?sysid=00644232)

---

## 補充 
### Elasticsearch 匯入流程
1. 修正中英文欄位錯誤（篇名、摘要）
2. 準備 embedding 資料組合
    - 中文：篇名、摘要、關鍵字
    - 英文：title, summary, keyword
3. 使用 jina embeddiing
4. 匯入 Elasticsearch