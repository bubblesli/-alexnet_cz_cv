"""author:XUFE_li"""

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from tkinter import ttk  
import tkinter  
from tkinter import *
from PIL import Image,ImageTk 
from model_alexnet import AlexNet
from tkinter.filedialog import askopenfilename
import tkinter.messagebox
import torch
from predict_alexnet import alex_predict,alex_predict2
import warnings
import os
warnings.filterwarnings("ignore")

# 设置窗体
root = Tk()
root.title("瓷砖完整度检测")

# 图片对话框信息
path = StringVar()
file_entry = Entry(root, state='readonly', text=path)

# 初始化
global now_img, alexnet_model
alexnet_model = None

# 选择图片
def choosepic():
    default_dir = r"./test/"
    path_ = askopenfilename(title=u'选择文件', initialdir=(os.path.expanduser(default_dir)))
    if len(path_) < 1:
        return
    path.set(path_)
    global now_img
    # 设置图片路径
    now_img = file_entry.get()
    #print(now_img)
    img_open = Image.open(file_entry.get())
    img_open = img_open.resize((360, 270))
    # 显示图片到窗体
    img = ImageTk.PhotoImage(img_open)
    image_label.config(image=img)
    image_label.image = img

# 退出软件
def btn_exit():
    exit(0)

# 调用模型
def btn_ai():
    global now_img
    global alexnet_model
    if alexnet_model is None:
        # create model
        alexnet_model = AlexNet(num_classes=4)
        # load model weights
        model_weight_path = "./models/alexnet.pth"
        alexnet_model.load_state_dict(torch.load(model_weight_path,map_location='cpu'))
        alexnet_model.eval()

    # 识别图像
    res = alex_predict(now_img,alexnet_model)
    tkinter.messagebox.showinfo('提示', '识别结果是：%s'%res)

# 显示识别率
def btn_acc():
    res = ''
    with open('./logs/alexnet.txt', 'r', encoding='utf-8') as fb:
        s = fb.read()
        res = res + '识别率是:%.2f%%'%(float(s)*100)
    tkinter.messagebox.showinfo('提示', res)

# 设置窗体
mainframe = ttk.Frame(root, padding="5 4 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)
# 配置按钮
ttk.Button(mainframe, text="选择图片", command=choosepic).grid(column=2, row=4, sticky=W)
ttk.Button(mainframe, text="AI模型", command=btn_ai).grid(column=4, row=4, sticky=W)
ttk.Button(mainframe, text="模型准确率", command=btn_acc).grid(column=5, row=4, sticky=W)
ttk.Button(mainframe, text="关闭软件", command=btn_exit).grid(column=6, row=4, sticky=W)
# 显示初始化
image_label = ttk.Label(root,compound=CENTER)
image_label.grid(column=0,row=5, sticky=W)
bg = "./bg.png"
pil_image = Image.open(bg)
pil_image = pil_image.resize((360, 270))
img = ImageTk.PhotoImage(pil_image)
image_label.configure(image=img)

root.mainloop()