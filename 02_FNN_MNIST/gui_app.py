import tkinter as tk
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import os
from model import SimpleFNN

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别 (FNN)")
        # 设置窗口大小
        self.root.geometry("700x500")
        self.root.configure(bg="#2E2E2E") # 设置深色背景，护眼又好看
        self.root.resizable(False, False)

        # 1. 准备模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        
        # 2. 准备数据预处理 (要把画板上的画变成模型认识的样子)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # 3. 初始化画笔
        self.last_x, self.last_y = None, None
        self.pen_width = 20  # 画笔粗一点，模仿 MNIST 的笔触
        
        # 创建一个内存中的图片，用来给模型看 (黑底白字)
        # 我们画在一个大图上 (280x280)，最后缩放成 28x28
        self.image_size = 280
        self.image = Image.new("L", (self.image_size, self.image_size), "black")
        self.draw = ImageDraw.Draw(self.image)

        # 4. 搭建界面
        self.setup_ui()

    def load_model(self):
        """加载训练好的模型权重"""
        print("正在加载模型...")
        model = SimpleFNN().to(self.device)
        
        # 拼凑出模型文件的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'models', 'mnist_fnn.pth')
        
        if os.path.exists(model_path):
            # 加载权重
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval() # 切换到测试模式
            print("模型加载成功！")
        else:
            print(f"警告：没找到模型文件 {model_path}，请先运行 train.py！")
        
        return model

    def setup_ui(self):
        """画界面的代码"""
        # 主容器
        main_frame = tk.Frame(self.root, bg="#2E2E2E")
        main_frame.pack(expand=True, fill="both", padx=20, pady=20)

        # --- 左边：画板 ---
        left_frame = tk.Frame(main_frame, bg="#2E2E2E")
        left_frame.pack(side="left", padx=20)

        # 提示文字
        tk.Label(left_frame, text="请在下方写一个数字 (0-9)", font=("微软雅黑", 14), fg="#DDDDDD", bg="#2E2E2E").pack(pady=(0, 10))

        # 画布控件
        self.canvas = tk.Canvas(left_frame, width=self.image_size, height=self.image_size, bg="black", highlightthickness=0)
        self.canvas.pack(pady=10)
        
        # 绑定鼠标事件：按下、移动、松开
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.drawing)
        self.canvas.bind("<ButtonRelease-1>", self.finish_draw)

        # 清除按钮
        btn_clear = tk.Button(left_frame, text="清空画板", font=("微软雅黑", 12), 
                              command=self.clear_canvas, bg="#FF5555", fg="white", 
                              relief="flat", padx=20, pady=5, cursor="hand2")
        btn_clear.pack(pady=20)

        # --- 右边：结果显示 ---
        right_frame = tk.Frame(main_frame, bg="#2E2E2E")
        right_frame.pack(side="right", expand=True, fill="both", padx=20)

        # 结果大标题
        tk.Label(right_frame, text="AI 识别结果", font=("微软雅黑", 12), fg="#AAAAAA", bg="#2E2E2E").pack(pady=(20, 5))
        
        # 显示最大的那个数字
        self.lbl_result = tk.Label(right_frame, text="-", font=("Arial", 80, "bold"), fg="#4CAF50", bg="#2E2E2E")
        self.lbl_result.pack(pady=10)

        # 详细概率列表
        tk.Label(right_frame, text="置信度 (可能性)", font=("微软雅黑", 10), fg="#AAAAAA", bg="#2E2E2E").pack(pady=(20, 10))
        
        self.prob_labels = []
        self.prob_bars = []
        
        probs_frame = tk.Frame(right_frame, bg="#2E2E2E")
        probs_frame.pack(fill="x")

        # 创建显示前3名的格子
        for i in range(3):
            row = tk.Frame(probs_frame, bg="#2E2E2E")
            row.pack(fill="x", pady=5)
            
            # 数字
            lbl_num = tk.Label(row, text="?", font=("Arial", 14, "bold"), fg="white", bg="#2E2E2E", width=4)
            lbl_num.pack(side="left")
            
            # 进度条背景
            bar_bg = tk.Frame(row, height=15, bg="#444444", width=150)
            bar_bg.pack(side="left", fill="x", expand=True, padx=10)
            bar_bg.pack_propagate(False)
            
            # 进度条前景 (绿条)
            bar_fg = tk.Frame(bar_bg, height=15, bg="#4CAF50", width=0)
            bar_fg.pack(side="left")
            
            # 百分比文字
            lbl_pct = tk.Label(row, text="0%", font=("Arial", 10), fg="#DDDDDD", bg="#2E2E2E", width=5)
            lbl_pct.pack(side="right")

            self.prob_labels.append((lbl_num, lbl_pct))
            self.prob_bars.append(bar_fg)

    # --- 绘画逻辑 ---
    def start_draw(self, event):
        self.last_x, self.last_y = event.x, event.y

    def drawing(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            # 1. 在屏幕上画给自己看
            self.canvas.create_line(self.last_x, self.last_y, x, y, 
                                  width=self.pen_width, fill="white", capstyle=tk.ROUND, smooth=True)
            # 2. 在内存图片上画给 AI 看
            self.draw.line([self.last_x, self.last_y, x, y], fill="white", width=self.pen_width)
        
        self.last_x, self.last_y = x, y

    def finish_draw(self, event):
        self.last_x, self.last_y = None, None
        # 每次画完一笔，就自动识别一次
        self.predict()

    def clear_canvas(self):
        """清空画布和结果"""
        self.canvas.delete("all") # 清屏幕
        self.draw.rectangle((0, 0, self.image_size, self.image_size), fill="black") # 清内存图
        
        # 重置文字
        self.lbl_result.config(text="-")
        for i in range(3):
            self.prob_labels[i][0].config(text="?")
            self.prob_labels[i][1].config(text="0%")
            self.prob_bars[i].config(width=0)

    def predict(self):
        """调用模型进行预测"""
        # -----------------------------------------------------
        # 智能预处理：自动裁剪并居中
        # -----------------------------------------------------
        # 1. 找到内容的边界 (Bounding Box)
        bbox = self.image.getbbox()
        
        if bbox:
            # 裁剪出有内容的区域
            img_cropped = self.image.crop(bbox)
            
            # 2. 创建一个 28x28 的黑底新图
            # MNIST 里的数字大部分是居中的，大小约 20x20
            new_img = Image.new("L", (28, 28), "black")
            
            # 3. 计算缩放比例，把裁剪出的数字缩放到 20x20 以内
            # 保持长宽比
            width, height = img_cropped.size
            ratio = min(20.0 / width, 20.0 / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            # 缩放
            img_resized = img_cropped.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 4. 贴到中心位置
            # 算出中心坐标
            paste_x = (28 - new_width) // 2
            paste_y = (28 - new_height) // 2
            new_img.paste(img_resized, (paste_x, paste_y))
            
            final_img = new_img
        else:
            # 如果画板是空的，就直接用全黑图
            final_img = self.image.resize((28, 28))
            
        # -----------------------------------------------------
        
        # 5. 变成 Tensor
        img_tensor = self.transform(final_img).unsqueeze(0).to(self.device)
        
        # 6. 喂给模型
        with torch.no_grad():
            outputs = self.model(img_tensor)
            # 用 softmax 算出概率 (加起来等于1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
        # 7. 取前 3 名
        top3_prob, top3_idx = torch.topk(probs, 3)
        top3_prob = top3_prob[0].cpu().numpy()
        top3_idx = top3_idx[0].cpu().numpy()
        
        # 8. 更新界面显示
        self.lbl_result.config(text=str(top3_idx[0])) # 显示第一名
        
        for i in range(3):
            digit = top3_idx[i]
            prob = top3_prob[i]
            
            # 更新数字和百分比
            self.prob_labels[i][0].config(text=str(digit))
            self.prob_labels[i][1].config(text=f"{prob:.1%}")
            
            # 更新进度条长度 (总长按 150px 算)
            bar_width = int(prob * 150)
            self.prob_bars[i].config(width=bar_width)

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
