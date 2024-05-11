"""author:XUFE_li"""


import torch
from PIL import Image
from torchvision import transforms
import json

# 调用AlexNet
def alex_predict(imgf,model):
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         #transforms.Grayscale(num_output_channels=3),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = Image.open(imgf)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    try:
        json_file = open('./config.json', 'r', encoding='utf-8')
        class_indict = json.load(json_file)
        #print(class_indict)
    except Exception as e:
        print(e)
        exit(-1)

    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    res=class_indict[str(predict_cla)]
    #print(res)
    return res



# 调用AlexNet
def alex_predict2(imgf,model):
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.Grayscale(num_output_channels=3),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = Image.open(imgf)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    try:
        json_file = open('./config2.json', 'r', encoding='utf-8')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    res=class_indict[str(predict_cla)]
    #print(res)
    return res