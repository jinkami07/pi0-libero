# Physical AI Sandbox — Docker

Ubuntu 22.04 + CUDA 12.1 コンテナで OpenPI + LIBERO を動かす環境。
VNC セットアップは [AI Robot Book Docker](https://github.com/AI-Robot-Book-Humble/docker-ros2-desktop-ai-robot-book-humble) の構成を参考にしています。

## 前提

| 環境 | 必要なもの |
|---|---|
| ローカル Mac | Docker Desktop のみ（ビルド可、GPU 実行不可） |
| GCP VM (L4 等) | NVIDIA GPU + nvidia-container-toolkit |

## ディレクトリ構成

```
RT/
├── docker/
│   ├── Dockerfile        # Ubuntu 22.04 + CUDA 12.1 + TigerVNC + noVNC
│   ├── docker-compose.yml
│   ├── entrypoint.sh     # VNC 起動スクリプト (supervisor 管理)
│   └── README.md
└── src/
    ├── 01_openpi_libero_smoke_test.py
    └── 02_libero_openpi_adaptive_reach_demo.py
```

## 使い方

### 0. リポジトリの準備

```bash
cd RT
git clone https://github.com/hyorimitsu/physical-ai-sandbox.git
```

### 1. イメージビルド（Mac でも可）

```bash
cd RT/docker
docker compose build
```

### 2. コンテナ起動（VNC デスクトップ）

```bash
docker compose up -d
```

起動後、以下の 2 通りで GUI にアクセスできます：

| 方法 | URL / アドレス | パスワード |
|---|---|---|
| **ブラウザ (noVNC)** | http://localhost:6080 | sandbox |
| **VNC クライアント** | localhost:5900 | sandbox |

- Mac の場合: Finder → 移動 → サーバーへ接続 → `vnc://localhost:5900`
- パスワード変更: `docker-compose.yml` の `PASSWORD=sandbox` を書き換える

### 3. GPU 認識確認（GPU 環境のみ）

```bash
docker compose run --rm sandbox nvidia-smi
```

### 4. スクリプトを直接実行（VNC なし）

```bash
# スモークテスト (GPU 不要)
docker compose run --rm sandbox \
    conda run -n pi0 python src/01_openpi_libero_smoke_test.py
# 期待出力: actions: (50, 7)

# デモ実行 (GPU 必要)
docker compose run --rm sandbox \
    conda run -n pi0 python src/02_libero_openpi_adaptive_reach_demo.py
# 出力: /tmp/output/02_libero_openpi_adaptive_reach_demo.mp4
```

### 5. コンテナ停止

```bash
docker compose down
```

## GCP へのデプロイ

既存の `terraform/main.tf` をそのまま利用できます。
VM 起動後に nvidia-container-toolkit をインストールし、このリポジトリを clone して `docker compose build` してください。

## VNC の仕組み

参照リポジトリ ([AI-Robot-Book-Humble](https://github.com/AI-Robot-Book-Humble/docker-ros2-desktop-ai-robot-book-humble)) と同じ構成:

| コンポーネント | 役割 |
|---|---|
| **TigerVNC** (`vncserver :0`) | X11 デスクトップを VNC で提供 (port 5900) |
| **websockify + noVNC** | VNC をブラウザから使えるよう WebSocket にプロキシ (port 6080) |
| **supervisor** | VNC と noVNC の両プロセスを管理・自動再起動 |
| **Xfce4** | 軽量デスクトップ環境 (xstartup で起動) |

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| `nvidia-smi` が失敗する | ホスト側の nvidia-container-toolkit を確認 |
| VNC に接続できない | `docker compose ps` でコンテナ起動を確認 |
| VNC ログを確認したい | `docker compose exec sandbox cat /var/log/vnc.log` |
| noVNC が開かない | `docker compose exec sandbox cat /var/log/novnc.log` |
| `MUJOCO_GL=egl` エラー | `libglew-dev` が入っているか確認（Dockerfile に含まれている） |
| JAX が GPU を認識しない | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.7` に下げる |
