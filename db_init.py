import os
import sqlite3

DB_PATH = "data/auth.db"

def main():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 用户表：存“名字”（以后可换 user_id）
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        name TEXT PRIMARY KEY,
        created_at TEXT DEFAULT (datetime('now'))
    )
    """)

    # 设备表：存 device_id
    cur.execute("""
    CREATE TABLE IF NOT EXISTS devices (
        device_id TEXT PRIMARY KEY,
        display_name TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    )
    """)

    # 权限表：某个人是否允许某台设备
    cur.execute("""
    CREATE TABLE IF NOT EXISTS permissions (
        name TEXT NOT NULL,
        device_id TEXT NOT NULL,
        allow INTEGER NOT NULL CHECK (allow IN (0,1)),
        updated_at TEXT DEFAULT (datetime('now')),
        PRIMARY KEY (name, device_id),
        FOREIGN KEY (name) REFERENCES users(name),
        FOREIGN KEY (device_id) REFERENCES devices(device_id)
    )
    """)

    # 日志表（可选，但建议留着）
    cur.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT DEFAULT (datetime('now')),
        device_id TEXT,
        who TEXT,
        score REAL,
        decision TEXT,
        reason TEXT
    )
    """)

    conn.commit()
    conn.close()
    print(f"OK: 数据库已初始化 -> {DB_PATH}")

if __name__ == "__main__":
    main()
