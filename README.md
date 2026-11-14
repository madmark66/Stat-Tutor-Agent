# Stat-Tutor-Agent

基於 Python、Streamlit、SQLite 與 Google Gemini API 開發的「雙腦協作統計學 AI 家教」Prototype。

## 專案概述與目前狀態

本專案為統計學蘇格拉底式 AI 家教原型（Prototype），核心概念為結合：
- **右腦（Gemini LLM）**：負責自然語言互動、蘇格拉底式反問引導與觀念診斷。
- **左腦（Python 計算引擎）**：負責精確的統計與數學計算（如樣本變異數與自由度計算），避免語言模型產生數值幻覺。
- **長期記憶庫（SQLite）**：記錄學生的學習弱點與常見觀念錯誤（`tutor_memory.db`），提供跨對話的個人化教學引導。

目前版本處於 **Phase 1 開發環境穩定化 Baseline**，包含以下主要模組與測試工具：
- `app.py`：Streamlit 網頁版雙腦 AI 家教介面。
- `cli_tutor.py`：命令列（CLI）互動式統計家教（採用 REST API 與 Function Calling）。
- `main.py`：日文版（日本統計檢定準一級）蘇格拉底家教對話腳本。
- `check_models.py` / `test.py`：Gemini API 可用模型查詢與基礎連線測試工具。

---

## 開發環境需求與建立指南

### 1. Python 版本需求

本專案基準開發環境使用 **Python 3.13.8**（建議使用 Python 3.13.x 系列）。

建立虛擬環境前，請先依您的作業系統或開發環境確認 Python 3.13 執行檔的位置：

```bash
# 方法 A：檢查系統 PATH 中是否有 python3.13
which python3.13
python3.13 --version

# 方法 B：在 GitHub Codespaces / Dev Containers 環境中檢查預裝路徑
ls -d /usr/local/python/3.13*
# 例如在 Codespaces 中可能位於 /usr/local/python/3.13.8/bin/python
```

### 2. 建立 `.venv` 虛擬環境

請將下方 `<PATH_TO_PYTHON_3.13>` 替換為您在步驟 1 確認的 Python 3.13 執行檔路徑（例如 `python3.13` 或 `/usr/local/python/3.13.8/bin/python`）：

```bash
# 建立名為 .venv 的虛擬環境
<PATH_TO_PYTHON_3.13> -m venv .venv

# 確認虛擬環境的 Python 版本
.venv/bin/python --version

# 升級 pip
.venv/bin/python -m pip install --upgrade pip
```

### 3. 安裝依賴套件（`requirements.txt` vs `requirements-lock.txt`）

專案提供兩份依賴管理檔案，用途區分如下：

- **`requirements.txt`（核心直接依賴）**：
  僅列出專案程式碼直接 `import` 的 5 個核心套件及其精確版本（`streamlit`、`numpy`、`requests`、`python-dotenv`、`google-generativeai`）。適合檢視專案架構依賴。
  *(註：目前為維持 Prototype 相容性暫時使用 `google-generativeai`，未來重構時將遷移至官方新版 `google-genai` SDK)*。
- **`requirements-lock.txt`（完整鎖定依賴清單）**：
  由 `pip freeze` 產出，完整鎖定所有直接依賴與底層轉移依賴（Transitive Dependencies，如 `grpcio`、`protobuf`、`pydantic`、`pyarrow` 等）的精確版本。

**建議安裝方式（100% 重現相同開發環境）：**

```bash
.venv/bin/pip install -r requirements-lock.txt
```

若僅需安裝核心直接依賴，則可執行：

```bash
.venv/bin/pip install -r requirements.txt
```

---

## 環境變數設定（`.env`）

本專案透過環境變數讀取 Google Gemini API Key。請在專案根目錄建立 `.env` 檔案（此檔案已被 `.gitignore` 排除，**切勿提交至 Git**）：

```bash
# 於專案根目錄建立 .env
cat << 'EOF' > .env
GEMINI_API_KEY="AIzaSyDUMMY_EXAMPLE_KEY_REPLACE_WITH_YOUR_OWN"
EOF
```

請將 `AIzaSyDUMMY_EXAMPLE_KEY_REPLACE_WITH_YOUR_OWN` 替換為您真實的 Gemini API Key。

設定完成後，可執行以下指令驗證 API 連線與可用模型：

```bash
.venv/bin/python check_models.py
```

---

## 啟動與執行方式

### 1. 啟動 CLI 統計家教（`cli_tutor.py`）

```bash
.venv/bin/python cli_tutor.py
```
- 進入互動模式後，可直接輸入統計問題或數據。
- 輸入 `reset` 可清除當次對話歷史。
- 輸入 `quit` 或 `exit` 可安全結束程式。

### 2. 啟動 Streamlit 網頁介面（`app.py`）

```bash
.venv/bin/streamlit run app.py
```
- 啟動後預設會監聽 `http://localhost:8501`。
- 若在無瀏覽器的遠端或測試環境中，可加上 `--server.headless true` 參數。
