# 完全準同型暗号を用いた連合学習  

pythonのバージョンは3.11を使用してください．  
ライブラリのインストール
```bash
pip install -r .artifacts/requirements.txt
```

完全準同型暗号を用いた連合学習を実行する．
```bash
python3 main_with_fhe.py
```

## OpenFHE (CKKS) のセットアップ

OpenFHE の Python バインディングは公式 PyPI では提供されていないため、ソースコードからビルドする必要があります。以下は一例です。

1. [OpenFHE のリポジトリ](https://github.com/openfheorg/openfhe-development)をクローンする。
2. `scripts/python` ディレクトリ以下の README に従って Python バインディングをビルド・インストールする。
3. `python -c "import openfhe"` を実行して import できることを確認する。

CKKS 集約を利用するスクリプト (`benchmark_fhe.py`) は `openfhe` モジュールがインポート可能であることを前提としています。

## ベンチマーク

平文・TFHE・CKKS の 3 方式を比較するベンチマークスクリプトを追加しました。

```bash
python3 benchmark_fhe.py
```

実行すると `benchmark_results/` 配下に以下が出力されます。

- `*.txt`: 各方式の精度・実行時間と整合性チェック結果を含むログ。
- `*_runtime.png`: ラウンドごとの実行時間グラフ。
- `*_accuracy.png`: ラウンドごとの精度グラフ。

ログ末尾の `RESULTS_JSON` を解析することで、平文の精度が暗号化手法よりも低くなっていないかなどの整合性を自動的に確認しています。
