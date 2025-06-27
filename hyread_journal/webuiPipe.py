from typing import Optional
import requests
import json
from pydantic import BaseModel, Field
import pprint


class Pipe:
    class Valves(BaseModel):
        API_URL: str = Field(
            default="http://host.docker.internal:8000/rag",
            description="RAG Streaming API 的 URL",
        )
        MODEL_NAME: str = Field(
            default="Hyread Journal", description="顯示於 UI 的模型名稱"
        )
        MODEL_ID: str = Field(
            default="hyread-journal-rag", description="模型的唯一識別碼"
        )
        SEARCH_TOP_K: int = Field(default=10, description="每次查詢取得的文獻數量")
        SHOW_AI_PARAM: bool = Field(default=False, description="回傳優化的搜尋參數")

    def __init__(self):
        self.name = "Hyread_journal"
        self.valves = self.Valves()
        print(f"初始化 Journal RAG Pipe，API URL: {self.valves.API_URL}")

    async def on_startup(self):
        print("Journal RAG 助手啟動中...")

    async def on_shutdown(self):
        print("Journal RAG 助手關閉中...")

    def pipes(self):
        return [{"id": self.valves.MODEL_ID, "name": self.valves.MODEL_NAME}]

    def pipe(self, **kwargs):
        body = kwargs.get("body", kwargs)
        try:
            print("\n===== OpenWebUI 傳入參數 =====")
            pprint.pprint(body, indent=2)
            print("==============================\n")

            messages = body.get("messages", [])
            query = next(
                (
                    msg.get("content")
                    for msg in reversed(messages)
                    if msg.get("role") == "user"
                ),
                None,
            )

            if not query:
                return "未找到用戶輸入內容"

            print(f"接收到查詢: {query}")

            payload = {
                "query": query,
                "top_k": self.valves.SEARCH_TOP_K,
                "show_ai_params": self.valves.SHOW_AI_PARAM,
            }

            response = requests.post(
                self.valves.API_URL,
                json=payload,
                stream=True,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            def generate_stream():
                references = []
                table_rows = []
                for line in response.iter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line.decode("utf-8"))

                        if "references" in data:
                            references = data["references"]
                            continue

                        if "summary" in data:
                            yield data["summary"]
                            continue

                        if "table_row" in data:
                            table_rows.append(data["table_row"])  # 改成收集
                            continue

                    except json.JSONDecodeError as e:
                        yield f"\n\nJSON 解析錯誤: {e}"
                    except Exception as e:
                        yield f"\n\n錯誤: {e}"
                if table_rows:
                    yield "\n\n### 表格資料\n"
                    yield "\n".join(table_rows) + "\n"  # 最後加一個換行

                if references:
                    yield "\n\n### 參考文獻\n"
                    for i, ref in enumerate(references, 1):
                        title = ref.get("title", "無標題")
                        sysid = ref.get("sysid", "")
                        if sysid:
                            url = f"https://www.hyread.com.tw/hyreadnew/search_detail_new.jsp?sysid={sysid}"
                            yield f"{i}. [{title}]({url}) (ID: {sysid})\n"
                        else:
                            yield f"{i}. {title}\n"

            return generate_stream()

        except requests.exceptions.RequestException as e:
            return f"無法連接 API：{e}"
        except Exception as e:
            return f"處理過程發生錯誤：{e}"
