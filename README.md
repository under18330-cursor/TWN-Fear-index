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

| # | 子指標 | 對應資料 | 資料來源（台灣） | 方向 |
|---|---|---|---|---|
| 1 | 動能 Momentum | 加權指數（TAIEX）收盤 vs 125 日均線乖離率 | TWSE 每日收盤行情 | 乖離率越正越貪婪 |
| 2 | 強度 Stock Strength | 創 52 週新高家數 vs 新低家數（淨值） | TWSE 個股日成交資訊 | 淨新高越多越貪婪 |
| 3 | 廣度 Market Breadth | 上漲家數/成交量 vs 下跌家數/成交量（累積） | TWSE 每日市場成交彙總 | 廣度越強越貪婪 |
| 4 | 選擇權 Put/Call Ratio | 台指選擇權（TXO）Put 量 / Call 量 | TAIFEX 選擇權每日交易資訊 | 比率越高越恐慌（反向） |
| 5 | 波動度 Volatility | 台指選擇權波動率指數（TAIEX VIX）相對 50 日均值偏離 | TAIFEX / 期交所 VIX 指數 | VIX 越高越恐慌（反向） |
| 6 | 避險需求 Safe Haven Demand | 台股 20 日報酬 − 公債/貨幣市場基金 20 日報酬 | TWSE 指數 + 債券指數 | 股優於債越多越貪婪 |
| 7 | 融資動能 Margin Sentiment | 融資餘額 20 日變化率（散戶槓桿多單） | TWSE 信用交易統計 | 融資暴增越貪婪、急縮越恐慌 |
| 8 | 外資部位 Foreign Positioning | 外資現貨 20 日累計買賣超金額 + 台指期未平倉多空比 | TWSE 三大法人買賣超 + TAIFEX 期貨未平倉 | 買超/偏多越多越貪婪 |

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
  scoring.py      # 百分位正規化、加權合成、極端旗標邏輯（純函式，無網路依賴）
  indicators.py   # 由原始價量/籌碼資料 DataFrame 計算 8 個子指標原始值
  data_sources.py # 對接 TWSE / TAIFEX 開放資料 API 的抓取函式（需要網路）
  index.py        # TWNFearGreedIndex：組裝以上三者，輸出最終指數與歷史序列
examples/
  run_example.py  # 用合成資料跑通整條 pipeline（不需網路，可離線驗證邏輯）
tests/
  test_scoring.py # 正規化與合成邏輯的單元測試
```

## 安裝與使用

```bash
pip install -r requirements.txt

# 離線示範（合成資料，驗證邏輯）
python examples/run_example.py

# 正式使用（需網路，抓 TWSE/TAIFEX 開放資料）
python -c "
from twn_fear_greed.data_sources import fetch_all_raw_data
from twn_fear_greed.index import TWNFearGreedIndex

raw = fetch_all_raw_data(start='2023-01-01')
idx = TWNFearGreedIndex()
result = idx.compute(raw)
print(result.tail())
"
```

## 資料來源備註

`data_sources.py` 對接以下公開資料（皆為台灣主管機關/交易所開放資料，免費、免申請 API Key）：

- 證交所 OpenAPI：<https://openapi.twse.com.tw/>（每日收盤行情、個股漲跌停/新高新低、三大法人買賣超、信用交易融資融券餘額）
- 期交所 OpenAPI：<https://openapi.taifex.com.tw/>（台指選擇權 Put/Call 成交量、台指選擇權波動率指數、期貨三大法人未平倉）

實際欄位名稱與端點路徑會隨交易所改版調整，`data_sources.py` 內已用常數集中管理端點，
若失效只需更新對應 URL/欄位對照即可，不影響 `scoring.py` / `indicators.py` 的核心邏輯。
