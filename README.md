# TWN Fear & Greed Index（台股恐慌貪婪指數）

一個仿照 CNN Fear & Greed Index 精神、但針對**台灣股市市場結構**重新設計的情緒指標。
分數範圍 0–100：

| 分數區間 | 標籤 | 意涵 |
|---|---|---|
| 0–20 | 極度恐慌 Extreme Fear | 市場情緒已到極端悲觀，過去統計上常伴隨短中期反彈機率上升 |
| 20–40 | 恐慌 Fear | 悲觀情緒偏濃，仍可能延續但已進入觀察區 |
| 40–60 | 中立 Neutral | 多空力道大致平衡 |
| 60–80 | 貪婪 Greed | 樂觀情緒偏濃，追價風險升高 |
| 80–100 | 極度貪婪 Extreme Greed | 情緒過熱，過去統計上常伴隨拉回/反轉機率上升 |

> 本指標是**情緒／擁擠度量表**，不是買賣訊號本身。極端值代表「反轉機率上升」，
> 不代表「必然反轉」，須搭配基本面與趨勢確認使用。

---

## 設計理念

CNN 原版指數用 7 個訊號（動能、強度、廣度、Put/Call、避險需求、垃圾債利差、波動度），
是針對美股/美債市場設計的。台股有幾個和美股不同、但情緒訊號效果更強的市場結構特徵：

1. **外資（三大法人）主導度極高**：外資現貨買賣超、台指期未平倉多空比，是台股情緒的核心風向球，
   美股的 F&G 指數完全沒有對應項目。
2. **散戶融資融券文化明顯**：融資餘額（散戶用槓桿做多）的暴增/斷頭式急縮，是台股特有、
   歷史上和短期高點/低點高度相關的訊號，美股沒有這麼發達的融資融券市場。
3. **有自己的波動率指數**：台灣期交所編製「台指選擇權波動率指數」（TAIEX VIX），
   可直接取代美股的 VIX。
4. **台指選擇權 Put/Call 量能**：可直接沿用 CNN 的避險需求邏輯。

因此本設計保留 CNN 的「動能、強度、廣度、Put/Call、波動度、避險需求」六項核心邏輯，
並新增「融資餘額動能」「外資買賣超與期貨部位」兩項台股特有訊號，共 **8 個子指標**。

---

## 八大子指標

| # | 子指標 | 對應資料 | FinLab 資料集 | 方向 |
|---|---|---|---|---|
| 1 | 動能 Momentum | 加權指數（TAIEX）收盤 vs 125 日均線乖離率 | `taiex_total_index:收盤指數` | 乖離率越正越貪婪 |
| 2 | 強度 Stock Strength | 創 52 週新高家數 vs 新低家數（淨值） | `price:收盤價`（全市場逐股矩陣，本地算 rolling 52週高低） | 淨新高越多越貪婪 |
| 3 | 廣度 Market Breadth | 上漲家數/成交量 vs 下跌家數/成交量（累積） | `price:收盤價` + `price:成交股數` | 廣度越強越貪婪 |
| 4 | 選擇權 Put/Call Ratio | 台指選擇權（TXO）Put 量 / Call 量 | FinLab 無此資料，改用 **FinMind** `TaiwanOptionDaily`（見下方說明） | 比率越高越恐慌（反向） |
| 5 | 波動度 Volatility | TAIEX 已實現波動率（20日年化）相對 50 日均值偏離 | `taiex_total_index:收盤指數`（自算已實現波動率，取代隱含波動率 VIX） | 波動越高越恐慌（反向） |
| 6 | 避險需求 Safe Haven Demand | 台股 20 日報酬 − 債券 ETF 20 日報酬 | `price:收盤價`（如 `00679B` 等已上市債券 ETF） | 股優於債越多越貪婪 |
| 7 | 融資動能 Margin Sentiment | 融資餘額 20 日變化率（散戶槓桿多單） | `margin_balance:融資券總餘額`（全市場合計） | 融資暴增越貪婪、急縮越恐慌 |
| 8 | 外資部位 Foreign Positioning | 外資現貨 20 日累計買賣超股數 + 台指期未平倉多空淨口數 | `institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)` + `futures_institutional_investors_trading_summary:多空未平倉口數淨額` | 買超/偏多越多越貪婪 |

> **Put/Call 指標的缺口，已用 FinMind 補上**：FinLab 是以個股基本面/技術面資料為主的平台，沒有
> 選擇權成交量資料集。`data_sources_finmind.py` 對接 [FinMind](https://github.com/FinMind/FinMind)
> 的 `TaiwanOptionDaily` 資料集，取台指選擇權（TXO）逐日 Put/Call 成交量。若完全不接 FinMind，
> `compute_all_raw_indicators` 仍會自動跳過這個子指標、權重由其餘 7 項按比例重新分配，不會補 0 分。
> 台指選擇權波動率指數（VIX，隱含波動率）目前仍無資料源，用 TAIEX 已實現波動率代替（見上表第 5 項）；
> 若之後想接真正的 VIX，可用 `data_sources.py` 裡的 TAIFEX OpenAPI 端點 `OptVixIndex`。
>
> **⚠️ Token 不能共用**：FinMind 的 API token 是它自己獨立的帳號系統核發的，**跟 FinLab 的 API Key
> 不是同一組**，不能拿 FinLab 的 key 去打 FinMind API。去 FinMind 自己的網站/GitHub 註冊拿 token，
> 設成 `FINMIND_API_TOKEN` 環境變數（一樣別寫進程式碼或提交進 git）。

### 正規化方法（每個子指標都轉成 0–100）

沿用 CNN 的作法：**百分位排名（percentile rank）**，而非簡單線性映射，因為情緒訊號的
極端值本來就該用「相對自己歷史的位置」衡量，而非固定門檻（門檻在不同年代會失效）。

```
score_t = percentile_rank(value_t, 過去 N 個交易日的 value 分布) × 100
```

- 預設 `N = 756`（約 3 年交易日），可調整。
- 對於「方向為反向」的指標（Put/Call、VIX），用 `100 − percentile_rank`。
- 資料不足 N 天時，用現有全部樣本計算（樣本太少時分數會標示為低信心 `low_confidence=True`）。

### 綜合分數（加權平均）

| 子指標 | 權重 |
|---|---|
| 動能 | 15% |
| 強度 | 15% |
| 廣度 | 15% |
| Put/Call | 12% |
| 波動度 | 15% |
| 避險需求 | 10% |
| 融資動能 | 10% |
| 外資部位 | 8% |

```
composite = Σ(weight_i × score_i)
```

權重可在 `TWNFearGreedIndex(weights=...)` 自訂；核心三項（動能/強度/廣度）與波動度權重較高，
是因為這四項資料品質最穩定、雜訊最少；融資與外資部位雖然是台股特色訊號，但單一天的數字
雜訊較大，故權重略低，主要用來「確認」而非「主導」分數。

---

## 極端與反轉邏輯（Overheat / Oversold Flags）

單日分數容易被雜訊干擾，因此額外提供兩種持續性判斷：

1. **持續極端天數**：分數連續 ≥5 個交易日落在 ≥80（過熱）或 ≤20（超賣）區間，
   標記 `overheat_streak` / `oversold_streak`，統計上這類持續性極端比單日極端更值得關注。
2. **情緒急殺/急拉**：10 個交易日內分數從 >70 驟降至 <40（或反向），
   標記 `sentiment_collapse` / `sentiment_spike`，代表情緒轉向速度異常快，
   常見於恐慌性拋售或軋空行情初期。

這些 flag 不是買賣訊號，而是「需要提高警覺、對照價格與籌碼面確認」的提示。

---

## 專案結構

```
src/twn_fear_greed/
  scoring.py                # 百分位正規化、加權合成、極端旗標邏輯（純函式，無網路依賴）
  indicators.py              # 由原始價量/籌碼資料 DataFrame 計算子指標原始值（缺資料的來源會自動略過）
  data_sources.py            # 對接 TWSE / TAIFEX 開放資料 API 的抓取函式（備用/選擇權資料源）
  data_sources_finlab.py     # 對接 FinLab（https://finlab.finance/）的抓取函式（主要資料源，需網路+API Key）
  data_sources_finmind.py    # 對接 FinMind 的抓取函式（補 Put/Call 選擇權資料，需網路+獨立 Token）
  index.py                   # TWNFearGreedIndex：組裝以上三者，輸出最終指數與歷史序列
examples/
  run_example.py             # 用合成資料跑通整條 pipeline（不需網路，可離線驗證邏輯）
tests/
  test_scoring.py                  # 正規化與合成邏輯的單元測試
  test_data_sources_finmind.py     # Put/Call 聚合邏輯的單元測試（純函式，不需網路）
```

## 安裝與使用

```bash
pip install -r requirements.txt

# 離線示範（合成資料，驗證邏輯，不需網路/API Key）
python examples/run_example.py
```

### 用 FinLab 資料源正式運行

**API Key 不要寫進程式碼或提交進 git**，一律用環境變數帶入：

```bash
export FINLAB_API_KEY="你的 FinLab API Key"
```

```python
from twn_fear_greed.data_sources_finlab import login, fetch_all_raw_data
from twn_fear_greed.index import TWNFearGreedIndex

login()  # 自動讀取 FINLAB_API_KEY 環境變數
raw = fetch_all_raw_data(start="2022-01-01")
idx = TWNFearGreedIndex()
result = idx.compute(raw)
print(result[["composite", "label", "overheat_flag", "oversold_flag"]].tail())
```

### 加上 FinMind，補 Put/Call 選擇權資料

```bash
export FINLAB_API_KEY="你的 FinLab API Key"
export FINMIND_API_TOKEN="你的 FinMind Token（跟 FinLab key 不同一組，見上方警告）"
```

```python
from twn_fear_greed.data_sources_finlab import login, fetch_all_raw_data
from twn_fear_greed.data_sources_finmind import augment_with_options
from twn_fear_greed.index import TWNFearGreedIndex

login()
raw = fetch_all_raw_data(start="2022-01-01")
raw = augment_with_options(raw, start="2022-01-01")  # 補上 put_call 子指標

idx = TWNFearGreedIndex()
result = idx.compute(raw)
print(result[["composite", "label", "put_call_score"]].tail())
```

FinLab 是全市場逐股歷史資料（不只是單日快照），所以「52週新高新低家數」「漲跌家數/量能廣度」
這兩項在純 TWSE OpenAPI 版本原本因為缺歷史資料而無法計算，改用 FinLab 後可以直接算。
但 FinLab 沒有選擇權/VIX 資料集，Put/Call 子指標會被自動略過（其餘 7 項權重按比例重新分配），
波動度子指標則改用 TAIEX 已實現波動率（20日年化）取代隱含波動率 VIX，詳見上方子指標表格。

## 資料來源備註

- **主要**：`data_sources_finlab.py` 對接 [FinLab](https://finlab.finance/)（付費訂閱制資料平台，
  需要 API Key，但提供乾淨的全市場歷史矩陣，不用自己維護逐股歷史）。
- **備用/選擇權資料**：`data_sources.py` 對接 TWSE / 期交所開放資料（免費、免 API Key，但個股類
  端點只回傳單日快照，52週新高新低/廣度需自行持續累積歷史）：
  - 證交所 OpenAPI：<https://openapi.twse.com.tw/>
  - 期交所 OpenAPI：<https://openapi.taifex.com.tw/>（唯一有台指選擇權 Put/Call 與 VIX 的來源）

兩個資料源回傳的 DataFrame 欄位是刻意對齊的（見 `indicators.py` 檔頭的 schema 說明），
所以可以混用：例如用 FinLab 取 6 項核心指標，再用 TAIFEX OpenAPI 補上 Put/Call，
兩份 dict 用 `{**finlab_raw, **taifex_options_raw}` 合併即可丟進同一個 `TWNFearGreedIndex`。
