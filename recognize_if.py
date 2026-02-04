"""
recognize_if.py - 人脸识别与权限判断脚本

功能概述：
- 加载本地人脸模板库（data/embeddings.npz），每条模板包含名字 (`names`) 与 embedding (`embs`)
- 使用 InsightFace 实时检测摄像头画面中的人脸，计算 cosine 相似度进行识别
- 根据 `permissions` 表判断识别到的人是否被允许（ALLOW）或拒绝（DENY）

快速使用：
1. 确保已录入模板（运行 `python enroll_if.py`），并存在 `data/embeddings.npz`
2. 运行：python recognize_if.py；按 Q 退出

注：注释专为初学者编写，解释了关键变量、函数和常见修改点（如阈值调整）。
"""

import os  # 文件与路径操作
import cv2  # OpenCV：摄像头读取、绘制、显示窗口
import numpy as np  # 数组和数值运算
import sqlite3  # 操作 SQLite 数据库

from insightface.app import FaceAnalysis  # InsightFace 高层接口，用于检测与提取人脸特征

# ===================== 配置项（可按需修改） =====================
DATA_PATH = "data/embeddings.npz"  # 保存人脸模板的文件路径
DB_PATH = "data/auth.db"           # SQLite 数据库路径（包含 permissions 表）
DEVICE_ID = "machine_001"          # 设备 id，用于从 permissions 表查权限

# 相似度阈值（cosine，相似度范围 [0,1]）
# - 值越大：判定为同一人的条件越严格（误识别减少，但可能更多 Unknown）
# - 值越小：判定更宽松（识别率提高，但误识别风险增加）
THRESHOLD = 0.55

# 简单白名单示例（注释掉，保留供参考）
# ALLOWED_USERS = {"zansan"}
# ================================================================


def load_db():
    """加载本地的名字列表和 embeddings。

    返回：
      names: list[str]，每个元素为模板对应的名字
      embs: numpy.ndarray，形状 (N, 512)，dtype=float32

    实现细节说明（便于理解）：
      - 使用 np.load 读取 npz，该文件通常由 enroll_if.py 生成
      - 如果文件缺失，返回空的 names 与形状为 (0,512) 的 embs，保证后续代码不崩溃
    """
    if not os.path.exists(DATA_PATH):
        # 首次运行或尚未录入任何用户的情况
        return [], np.empty((0, 512), dtype=np.float32)

    db = np.load(DATA_PATH, allow_pickle=True)
    names = db["names"].tolist()        # 将 numpy 数组转换为 Python 列表，方便 count/append 等操作
    embs = db["embs"].astype(np.float32)  # 强制为 float32，便于点乘计算
    return names, embs


def is_allowed(name: str, device_id: str) -> bool:
    """检查指定名字在给定设备上是否被允许（permissions 表）。

    说明：
      - 若 name 为 "Unknown"，直接返回 False（未知人不允许）
      - permissions 表应包含字段 (name, device_id, allow)，allow=1 表示允许
    """
    if name == "Unknown":
        return False

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT allow FROM permissions WHERE name=? AND device_id=?", (name, device_id))
    row = cur.fetchone()
    conn.close()

    # 若查询到记录且 allow 字段为 1，则允许
    return (row is not None) and (int(row[0]) == 1)


def pick_largest_face(faces):
    """从检测到的人脸中选择面积最大的那一张作为主脸。

    参数：faces - insightface 返回的人脸对象列表
    返回：face 对象或 None（当没有检测到人脸时）

    选择理由：通常我们希望采集/识别画面中最主要的那张人脸（占比最大），可以避免误采小旁人或背景的人脸。
    """
    if not faces:
        return None
    # bbox 形如 [x1, y1, x2, y2]，面积 = (x2-x1)*(y2-y1)
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def main():
    """主流程：实时读取摄像头帧，检测人脸并进行识别/权限判断。"""

    # 1) 加载本地模板库（names, embs）
    names, embs = load_db()
    if len(names) == 0:
        print("库为空：请先运行 enroll_if.py 录入人员")
        return

    # 2) 初始化模型
    #    - name="buffalo_l" 是模型名；
    #    - ctx_id=-1 表示使用 CPU（有 GPU 的机器可改为 GPU id，例如 0）
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))

    # 3) 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("摄像头打开失败：VideoCapture(0) 失败。可尝试改成 1/2")
        return

    # 运行提示，帮助新手理解当前运行状态与参数
    print("\n运行中：按 Q 退出。")
    print(f"当前阈值 THRESHOLD={THRESHOLD}")
    print(f"设备 ID = {DEVICE_ID}\n")

    # 4) 主循环：读取帧、检测、识别、显示
    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取摄像头帧失败，退出")
            break

        # 检测：app.get 返回一个包含人脸信息的对象列表
        faces = app.get(frame)
        face = pick_largest_face(faces)

        if face is not None:
            # 获取 bbox（整数），以及归一化后的 embedding（长度 512）
            x1, y1, x2, y2 = face.bbox.astype(int)
            emb = face.normed_embedding.astype(np.float32)

            # 相似度计算：由于 embeddings 已经归一化（insightface 返回的 normed_embedding），
            # 两个向量的点乘等价于 cosine 相似度（范围 [-1,1]，正向越大越相似）
            sims = embs @ emb  # numpy 的向量化点乘，结果为 (N,) 的相似度数组

            # 找出相似度最高的模板及其得分
            idx = int(np.argmax(sims))
            best_name = names[idx]
            best_score = float(sims[idx])

            # 根据阈值判断是否成功识别
            if best_score >= THRESHOLD:
                who = best_name
            else:
                who = "Unknown"

            # 判断权限（是否 ALLOW）
            allow = is_allowed(who, DEVICE_ID)
            status = "ALLOW" if allow else "DENY"

            # 将结果画到画面上：人脸框、名字与相似度分数、ALLOW/DENY 状态
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, f"{who}  score={best_score:.3f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, status, (x1, y2 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)

        # 显示画面并处理按键
        cv2.imshow("Recognize (Q=quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break

    # 5) 退出前释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

