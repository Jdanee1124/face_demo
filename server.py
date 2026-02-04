# 主控端代码：人脸识别+给被控端下发权限指令
import face_recognition
import cv2
import socket
import time
import os  # 用于读取人脸库文件夹

# 1. 配置基础参数（小白仅需修改 FACE_DB_PATH 这一行）
FACE_DB_PATH = "C:\\Users\\你的电脑用户名\\Desktop\\face_database"  # 替换成你的face_database文件夹路径
PERMISSION_TABLE = {
    "zhangsan": [["127.0.0.1"], "full"],  # 127.0.0.1 是本机回环地址，用于单机测试（主控端和被控端在同一台电脑）
}
PORT = 8888  # 通信端口，和被控端保持一致，无需修改

# 2. 加载人脸库，提取人脸特征（自动读取face_database文件夹里的.jpg照片）
known_face_encodings = []  # 存储已知人脸的特征值
known_face_names = []      # 存储已知人脸对应的用户名

for file in os.listdir(FACE_DB_PATH):
    if file.endswith(".jpg"):
        # 拼接完整的照片路径
        img_path = os.path.join(FACE_DB_PATH, file)
        # 加载照片并提取人脸特征
        face_img = face_recognition.load_image_file(img_path)
        face_encoding = face_recognition.face_encodings(face_img)[0]
        # 将特征和用户名存入列表
        known_face_encodings.append(face_encoding)
        known_face_names.append(file.replace(".jpg", ""))

# 3. 初始化电脑摄像头（0为默认摄像头，多摄像头可改为1/2）
video_capture = cv2.VideoCapture(0)

# 4. 定义：给被控机发送权限指令的函数
def send_permission(ip, name, permission):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((ip, PORT))
        # 发送指令格式：用户名_权限等级（编码为utf-8，避免中文乱码）
        s.send(f"{name}_{permission}".encode("utf-8"))
        s.close()
        print(f"已给{ip}下发{name}的{permission}权限")
    except Exception as e:
        print(f"{ip}机器离线或连接失败，错误信息：{e}")

# 5. 实时人脸识别+权限下发（循环运行，直到按Q退出）
while True:
    # 读取摄像头每一帧画面
    ret, frame = video_capture.read()
    if not ret:
        print("无法读取摄像头画面，请检查摄像头是否正常")
        break
    # 格式转换：OpenCV的BGR格式转为face_recognition需要的RGB格式
    rgb_frame = frame[:, :, ::-1]
    
    # 检测画面中的所有人脸，并提取特征
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    # 遍历检测到的每个人脸
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 对比人脸库，判断是否为已知人员
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "未知人员"
        
        # 若匹配成功，获取对应用户名并下发权限
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            
            # 从权限表中获取该用户的可操作机器和权限
            if name in PERMISSION_TABLE:
                ip_list, permission = PERMISSION_TABLE[name]
                for ip in ip_list:
                    send_permission(ip, name, permission)
                    time.sleep(0.5)  # 延迟0.5秒，防止指令发送过快导致丢失
        
        # 在摄像头画面上标注人脸框和用户名（方便查看识别结果）
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 显示摄像头识别画面
    cv2.imshow('Face Recognition (按Q退出)', frame)
    
    # 按Q键退出循环（关闭摄像头画面）
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. 释放资源（关闭摄像头和所有窗口）
video_capture.release()
cv2.destroyAllWindows()
