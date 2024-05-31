利用 BERT 之圍棋 AI 下子預測
BERT for Go Prediction
===

## Abstract
圍棋 AI 發展至今，一直是使用 ResNet 相關架構，將棋局轉換成圖片來處理。而近年來 Transformer 因為能處理文字序列成為深度學習的主流。.sgf 檔案內容為一串 token，這些 token 屬於一個有361個字的字典，像是一段句子，可以丟進 BERT 訓練！因此我使用 BERT 訓練圍棋落子選點器(圍棋 AI 的前半部分)，並模仿 Alpha Go，訓練了他使用的 ResNet 模型作為比較標準，分析比較兩者的結果。

---

## Dataset
### Source
* Data1:職業比賽(about 33000 games)
* Data2:野狐圍棋中九段玩家的棋局(about 140000 games)

### Definitions
* Train Data: 0~27000 games in Data1
* Eval Data: 27000~30000 games in Data1
* Test Data: Data2
* m: 一場棋局採用前幾步。
* Step-by-step data: 
    * 將一場棋局一步步拆開，使 (n,m) 的資料轉換成 (nxm,m) 的資料，範例如下:
    * `[...,[...],[5,4,3,2,1],[...],...]` ->`[...,[...],x1,x2,x3,x4,x5,[...],...]`
        * `x1 = [0,0,0,0,0]；y = 5`
        * `x2 = [5,0,0,0,0]；y = 4`
        * `x3 = [5,4,0,0,0]；y = 3`
        * `x4 = [5,4,3,0,0]；y = 2`
        * `x5 = [5,4,3,2,0]；y = 1`
    * 也就是說，使用 n 盤，會產生 nxm 筆資料
* Invalid move: 選擇的落子處已經有子存在
* Accuracy_n: 機率最高的前 n 選有選中的比率

### ResNet Data
* 輸入(1,channel,19,19)的圖片表示一個盤面
* 資料處理方式模仿 Alpha Go，但簡化內容使其記憶體容量跟 BERT Data 一樣，其channels 包含: 
    * white position(1)
    * black position(1)
    * next turn(1)
    * stones have 1~6^+^ liberties(1)(棋子的"氣")

### BERT Data
* 361個位置形成361個 token
* 加入 "SEP" 在結尾
* 輸入為三個 (1,m) 的 tensor，包含:
    * input_ids(落子位置順序)
    * masks(就是 mask)
    * token_type_ids(棋子的"氣")
* 將 token_types_ids 當成 embedding 資訊的管道
* 若要新增其他資訊，就使用相同方法，將其放入 embedding 層

### Compare
![image](https://hackmd.io/_uploads/B1KV6yPmC.png =80%x)

---

## Models
* 做一個 361 分類問題
* 模型大小: 4.4M parameters
* 皆沒有預訓練

### ResNet
* 使用 Leela zero 的 ResNet 架構
    * Leela zero 是全球知名圍棋開源專案，實做 Alpha Go 論文
    * 我擷取其中的"落子選點器"部分模型來使用
    * 他也曾經做過以"純人類棋譜"訓練的版本，達到 BayesElo 3610分
        * 換算後為 ELO 2638分
        * 世界頂尖好手約在 ELO 3600分
        * 參考我國女子選手俞俐均現為2759分，台灣職業四段
        * 2638分約為台灣職業三段
* [Leela zero github](https://github.com/leela-zero/leela-zero)

### BERT
* 使用 Huggingface 的模型，自訂模型參數
* 使用 mean of last hidden states 來分類
* 特殊參數設定:
```python!
config = BertConfig()
config.type_vocab_size = 7
config.vocab_size = 363
config.num_attention_heads = 1
config.position_embedding_type = "relative_key"
```

---
