# Copilot 使用说明（面向 AI 协作编码代理）

**概览** ✅
- 这是一个轻量级的人脸识别演示项目，主要模块：
  - `enroll_if.py`：采集人脸、生成并追加 embedding 到 `data/embeddings.npz`，同时将用户写入 SQLite（`data/auth.db`）。
  - `recognize_if.py`：实时摄像头识别，使用 `insightface.FaceAnalysis`，计算 cosine 相似度并根据 `permissions` 表决定 ALLOW/DENY。
  - `db_init.py` / `db_admin.py`：初始化 SQLite schema，管理用户/设备/权限。
  - `client.py` / `server.py`：简单 socket 示例，用于演示主控端给被控端下发权限的流程（独立于主要识别流程）。

**关键文件与约定** 🔧
- embeddings 保存：`data/embeddings.npz`，包含 `names`（Python 列表）和 `embs`（float32 数组，形状 (N,512)）。请使用 `save_db()`/`load_db()` 接口读写。
- SQLite 路径：`data/auth.db`。表结构由 `db_init.py` 定义（`users`, `devices`, `permissions`, `logs`）。
- 命名约定：录入时建议使用英文或拼音（见 `enroll_if.py` 的提示），`device_id` 如 `machine_001`。
- 模型与设备：默认使用 InsightFace 的 `buffalo_l`，CPU 模式（`ctx_id=-1`）；嵌入长度 512。
- 判决阈值：在 `recognize_if.py` 常量 `THRESHOLD`（默认 0.55）。调整该值直接修改该常量以改变灵敏度。
- 选脸策略：项目中使用 `pick_largest_face`（选择面积最大的检测框）作为主脸样本。
- 采集控制：`enroll_if.py` 使用键盘交互：`S` 采样、`Q` 保存并退出；`recognize_if.py`：`Q` 退出。

**常用命令 / 开发流程** ⚙️
- 初始化数据库：
  - python db_init.py  # 创建 `data` 目录和 sqlite 表
- 录入样本：
  - python enroll_if.py  # 按提示输入名字，采样 S/Q
- 查看/修改权限：
  - python db_admin.py add_user <name>
  - python db_admin.py add_device <device_id> "Display Name"
  - python db_admin.py set_perm <name> <device_id> <0|1>
  - python db_admin.py list_perms <device_id>
- 运行识别：
  - python recognize_if.py
- 简易远程演示（在另一台或同机上运行）：
  - 在被控端运行 `client.py`（弹窗示例）
  - 在主控端运行 `server.py`，修改 `FACE_DB_PATH` 与 `PERMISSION_TABLE` 后自动下发权限

**实现细节与注意点** ⚠️
- `load_db()` 在 `data/embeddings.npz` 不存在时返回空库（便于首次运行）。
- `embs @ emb` 等价于 cosine 相似度（项目里 emb 已归一化）。
- `enroll_if.py` 在成功写入后会调用 `upsert_user_to_sqlite` 来确保 `users` 表有记录。
- `db_admin.py` 使用 SQLite 的 `ON CONFLICT` 来做 upsert，保持单一来源的权限更新逻辑。
- 摄像头索引：如果 `VideoCapture(0)` 打不开，尝试改为 `1` 或 `2`。

**依赖 / 环境提示** 🧩
- 主要依赖：`insightface`, `opencv-python`, `numpy`（以及 Python 内置 `sqlite3`）。
- 安装建议：
  - pip install insightface opencv-python numpy
- 默认模型在 CPU 上运行（`ctx_id=-1`），在需要 GPU 时修改 `ctx_id` 或环境。

**修改与扩展建议（供 AI 参考）** 💡
- 若增加批量导入或导出功能，优先复用 `load_db()`/`save_db()` 的数据布局（names, embs）。
- 添加单元测试时，mock 摄像头与 face 模型输出；可用小型 numpy 存档作为测试 DB。
- 新增设备/权限变更时，请遵循 `db_admin.py` 的命令行参数格式与 `permissions` 表结构。

---

如果有需要我可以：
1) 将这份内容合并到仓库（已创建为 `.github/copilot-instructions.md`），或
2) 根据你更关注的部分（比如扩展到 GPU、CI、或增加 API 层）补充示例与注意事项。

请告诉我哪部分需要更详细或被忽略了，我会据此迭代文档。 ✅