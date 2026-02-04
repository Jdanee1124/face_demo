import sqlite3
import sys
import os

DB_PATH = "data/auth.db"

def usage():
    print("""
用法（在 D:\\face_demo 下执行）:

1) 添加用户（名字必须和 enroll_if.py 录入的名字一致）
   python db_admin.py add_user zansan

2) 添加设备
   python db_admin.py add_device machine_001 "Demo Machine 1"

3) 设置授权（allow=1 允许，0 拒绝）
   python db_admin.py set_perm zansan machine_001 1
   python db_admin.py set_perm lis machine_001 0

4) 查看设备的所有授权
   python db_admin.py list_perms machine_001
""")

def connect():
    if not os.path.exists(DB_PATH):
        print("数据库不存在，请先运行：python db_init.py")
        sys.exit(1)
    return sqlite3.connect(DB_PATH)

def add_user(name):
    conn = connect()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO users(name) VALUES(?)", (name,))
    conn.commit()
    conn.close()
    print(f"OK: user -> {name}")

def add_device(device_id, display_name):
    conn = connect()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO devices(device_id, display_name) VALUES(?,?)",
                (device_id, display_name))
    conn.commit()
    conn.close()
    print(f"OK: device -> {device_id} ({display_name})")

def set_perm(name, device_id, allow):
    allow = int(allow)
    if allow not in (0, 1):
        print("allow 必须是 0 或 1")
        return

    conn = connect()
    cur = conn.cursor()

    # 确保 user/device 存在
    cur.execute("INSERT OR IGNORE INTO users(name) VALUES(?)", (name,))
    cur.execute("INSERT OR IGNORE INTO devices(device_id, display_name) VALUES(?,?)",
                (device_id, device_id))

    cur.execute("""
    INSERT INTO permissions(name, device_id, allow)
    VALUES(?,?,?)
    ON CONFLICT(name, device_id) DO UPDATE SET
        allow=excluded.allow,
        updated_at=datetime('now')
    """, (name, device_id, allow))

    conn.commit()
    conn.close()
    print(f"OK: perm -> name={name}, device={device_id}, allow={allow}")

def list_perms(device_id):
    conn = connect()
    cur = conn.cursor()
    cur.execute("""
    SELECT p.name, p.allow, p.updated_at
    FROM permissions p
    WHERE p.device_id=?
    ORDER BY p.name
    """, (device_id,))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print(f"(空) device_id={device_id} 没有任何授权记录")
        return

    print(f"device_id={device_id} 权限列表：")
    for name, allow, updated_at in rows:
        print(f"  - {name:12s}  allow={allow}  updated_at={updated_at}")

def main():
    if len(sys.argv) < 2:
        usage()
        return

    cmd = sys.argv[1].lower()

    if cmd == "add_user" and len(sys.argv) == 3:
        add_user(sys.argv[2])
    elif cmd == "add_device" and len(sys.argv) >= 3:
        device_id = sys.argv[2]
        display_name = sys.argv[3] if len(sys.argv) >= 4 else device_id
        add_device(device_id, display_name)
    elif cmd == "set_perm" and len(sys.argv) == 5:
        set_perm(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "list_perms" and len(sys.argv) == 3:
        list_perms(sys.argv[2])
    else:
        usage()

if __name__ == "__main__":
    main()
