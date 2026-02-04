# 被控端代码：接收主控端权限指令，弹出权限提示
import socket
import tkinter as tk
from tkinter import messagebox

# 配置参数（和主控端一致，无需修改）
PORT = 8888
# 隐藏tkinter主窗口（仅弹出提示框，不显示多余窗口）
root = tk.Tk()
root.withdraw()

# 初始化socket服务端，等待主控端连接
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("", PORT))  # 绑定本机所有IP，无需修改
s.listen(5)
print("本机已上线，等待主控端权限指令...")

# 循环接收主控端指令，永不退出（直到手动关闭窗口）
while True:
    conn, addr = s.accept()
    data = conn.recv(1024).decode("utf-8")
    conn.close()
    # 解析指令：拆分「用户名_权限等级」
    name, permission = data.split("_")
    # 执行对应权限操作（弹窗提示，小白可直接用）
    if permission == "full":
        messagebox.showinfo("权限通知", f"您好{name}，已为您开放全部权限！")
        # 后续可添加真实操作，比如控制硬件、打开软件，现阶段无需修改
    elif permission == "read":
        messagebox.showinfo("权限通知", f"您好{name}，已为您开放只读权限！")
    else:
        messagebox.showwarning("权限通知", "无有效权限！")
