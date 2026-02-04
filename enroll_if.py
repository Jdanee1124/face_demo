"""
enroll_if.py - 录入人脸模板脚本（已添加逐行/逐块中文注释，便于学习）

功能概述：
- 使用 insightface 检测并提取人脸 embedding
- 允许用户通过按键采集多张样本并保存到本地 npz 文件
- 将用户信息同步写入 SQLite 用户表（通过 upsert）

使用说明（对于初学者）：
1. 运行前请确认已安装依赖：pip install insightface opencv-python numpy
2. 运行：python enroll_if.py，按提示输入名字，然后按 S 采样，按 Q 保存并退出
3. 采样建议：8~15 张，包含不同表情、角度与距离，可提升识别鲁棒性

注：文件中注释以中文逐行说明每一步的作用，便于新手阅读与学习。
"""

import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import sqlite3


# 常量：保存 embeddings 的文件路径（默认 data/embeddings.npz）
DATA_PATH = "data/embeddings.npz"
# SQLite 数据库路径（包含 users, devices, permissions 等表）
DB_PATH = "data/auth.db"

# 确保 data 目录存在，防止首次运行保存失败
os.makedirs("data", exist_ok=True)


def load_db():
    """从 DATA_PATH 载入名字和 embeddings。

    返回值：
      - names: Python list，长度为模板数量，元素为对应名字（字符串）
      - embs: numpy.ndarray，形状为 (N, 512)，dtype 为 float32

    行为说明（逐步）：
      1. 如果文件存在，用 np.load 读取 npz（支持 allow_pickle）
      2. 将 "names" 转为 Python 列表方便后续操作
      3. 将 "embs" 转为 float32；如果文件不存在则返回空数组（形状 (0,512)）
    """
    if os.path.exists(DATA_PATH):
        # np.load 返回的是类似 dict 的对象，可以通过键名访问保存的数据
        db = np.load(DATA_PATH, allow_pickle=True)
        # names 使用 tolist() 将 numpy 数组转为 Python 列表，便于 append/count 等操作
        names = db["names"].tolist()
        # embeddings 保持为 float32，后续的运算（如归一化、相似度计算）通常要求 float32
        embs = db["embs"].astype(np.float32)
        return names, embs

    # 当文件不存在时返回空数据库，方便首次录入时直接垂直堆叠（vstack）
    return [], np.empty((0, 512), dtype=np.float32)


def save_db(names, embs):
    """将 names 和 embs 保存到 DATA_PATH（npz 格式）。

    说明：
      - names 被转为 dtype=object 的 numpy 数组以支持字符串列表
      - embs 为 float32 的二维数组
    """
    np.savez(DATA_PATH, names=np.array(names, dtype=object), embs=embs)


def upsert_user_to_sqlite(name: str):
    """将用户名写入到 SQLite `users` 表（若已存在则忽略）。

    逐步说明：
      1. 检查 DB_PATH 是否存在，如果不存在提示用户先运行 db_init.py
      2. 用 INSERT OR IGNORE 语句保证不会重复插入同名用户
      3. 提交事务并关闭连接
    """
    if not os.path.exists(DB_PATH):
        print(f"警告：找不到数据库 {DB_PATH}，请先运行：python db_init.py")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # INSERT OR IGNORE 的意思是：如果已经存在相同主键/唯一约束的记录，则忽略本次插入
    cur.execute("INSERT OR IGNORE INTO users(name) VALUES(?)", (name,))
    conn.commit()
    conn.close()
    print(f"SQLite: 已登记用户 -> {name}")


def pick_largest_face(faces):
    """从检测到的 faces 列表中选择面积最大的检测框对应的人脸。

    参数：faces - insightface 返回的人脸对象列表（每个对象包含 bbox 等信息）
    返回：选中的 face 对象，或在没有检测到人脸时返回 None
    选择理由：通常想采集主脸（占画面面积最大），可以避免采集到旁边的小人脸或干扰。
    """
    if not faces:
        return None
    #  bbox 为 [x1, y1, x2, y2]，使用面积 (x2-x1)*(y2-y1) 进行比较
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def main():
    """主流程：交互式从摄像头采集人脸 embedding 并保存。

    逐步解释：
      1. 通过 input 获取要录入的名字（建议英文/拼音）
      2. 加载已有数据库，判断该名字是否已存在并询问是否追加
      3. 初始化 insightface 模型（默认 CPU）并打开摄像头
      4. 在循环中显示摄像头画面，按 S 采样，按 Q 退出并保存
      5. 将采集的 embedding 批量追加到本地 npz 文件并写入 SQLite
    """

    # 1) 获取录入名字
    name = input("输入要录入的名字（建议用英文/拼音，如 Tom）：").strip()
    if not name:
        print("名字不能为空")
        return

    # 2) 载入已有数据库并检查同名情况
    names, embs = load_db()
    existing_count = names.count(name)
    if existing_count > 0:
        # 如果已存在同名模板，询问用户是否继续追加
        ans = input(
            f"名字 '{name}' 已存在，已有 {existing_count} 条模板，是否继续追加？(y/N): "
        ).strip().lower()
        if ans != 'y':
            print("取消录入")
            return

    # 3) 初始化人脸分析模型（insightface）
    #    name="buffalo_l" 是选择的模型名称，可替换为其它模型
    #    ctx_id=-1 表示使用 CPU 运行，若有 GPU 则可以改为 GPU 的 ctx_id（如 0）
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))  # det_size 是检测输入尺寸，可调

    # 4) 打开摄像头
    cap = cv2.VideoCapture(0)  # 设备索引通常为 0，若失败可尝试 1 或 2
    if not cap.isOpened():
        print("摄像头打开失败：VideoCapture(0) 失败。可尝试改成 1/2")
        return

    # 用于保存本次录入的 embedding（list，每个元素为 embedding 向量）
    collected = []

    # 操作提示，方便小白理解按键功能
    print("\n操作：")
    print("  - 对准人脸（尽量正脸、光线充足）")
    print("  - 按 S：采集一张（建议 8~15 张，多角度/远近）")
    print("  - 按 Q：保存并退出\n")

    # 5) 循环读取摄像头并响应用户按键
    while True:
        ret, frame = cap.read()  # ret 表示是否成功读取到帧，frame 为图像数组
        if not ret:
            print("读取摄像头帧失败")
            break

        # 使用 insightface 检测人脸，返回的 faces 是对象列表
        faces = app.get(frame)
        face = pick_largest_face(faces)  # 选取面积最大的那张作为主脸

        if face is not None:
            # 将 detection 的 bbox 转为整数并绘制绿色矩形框
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # det_score 是检测置信度（用于判断是否采集），不是识别相似度
            det_score = getattr(face, "det_score", 0.0)

            # 在图像顶部显示检测置信度和已采样张数
            cv2.putText(
                frame,
                f"det={det_score:.2f}  samples={len(collected)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
        else:
            # 没有人脸时在左上角标注 No face（红色）
            cv2.putText(
                frame,
                "No face",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        # 显示当前帧，窗口名提示按键
        cv2.imshow("Enroll (S=save, Q=quit)", frame)
        # cv2.waitKey 返回的是按键的 ASCII 值（Windows 与 Linux 通常一致），取低八位用 & 0xFF
        key = cv2.waitKey(1) & 0xFF

        # 按 S 或 s 采样保存当前 face 的 embedding
        if key in (ord("s"), ord("S")):
            if face is None:
                print("未检测到人脸，无法采集")
                continue

            # 检查检测置信度，太低则跳过采集（避免采到错误区域）
            det_score = getattr(face, "det_score", 0.0)
            if det_score < 0.80:
                print(f"检测置信度太低(det={det_score:.2f})，换光线/靠近一点再采集")
                continue

            # face.normed_embedding 已是归一化后的向量（长度 512），转为 float32
            emb = face.normed_embedding.astype(np.float32)
            collected.append(emb)  # 保存到临时列表中
            print(f"已采集：{len(collected)} 张")

        # 按 Q 或 q 退出采集循环并保存数据
        if key in (ord("q"), ord("Q")):
            break

    # 释放摄像头资源并关闭所有 OpenCV 窗口
    cap.release()
    cv2.destroyAllWindows()

    # 如果没有采集任何样本就退出（不写入库）
    if len(collected) == 0:
        print("未采集任何样本，不写入库")
        return

    # 将 list->ndarray 并堆叠到现有数据库中
    collected = np.stack(collected, axis=0)  # 形状 (M, 512)
    for _ in range(collected.shape[0]):
        names.append(name)  # 为每个 embedding 追加名字

    # 垂直拼接旧的 embs 和新采集的 collected
    embs = np.vstack([embs, collected])

    # 保存到 DATA_PATH（npz）文件中
    save_db(names, embs)
    print(f"\n写入完成：{name} 新增 {collected.shape[0]} 条模板")
    print(f"当前模板总数：{len(names)}")

    # 将用户名写入 SQLite（若 users 表缺失则会提示）
    upsert_user_to_sqlite(name)


if __name__ == "__main__":
    main()
