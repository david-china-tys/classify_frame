import torch
import torchvision.models as models
from Net.mobilenet.mobilenetv2 import Mobilenetv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os

class ModelPipline(object):
    def __init__(self):
        #进入模型的图片大小：为数据预处理和后处理做准备
        self.inputs_size=(224,224)
        #CPU or CUDA：为数据预处理和模型加载做准备
        self.device=torch.device('cuda')
        #载入模型结构和模型权重
        self.model=self.get_model()
        #载入标签，为数据后处理做准备
        with open('txt_labels/imagenet_label.txt', 'r')as f:
            label_names =f.readlines()
        self.label_names=[line.strip('\n') for line in label_names]

    def predict(self, image):
        #数据预处理
        inputs=self.preprocess(image)
        #数据进网络
        outputs=self.model(inputs)
        #数据后处理
        results=self.postprocess(outputs)
        return results

    def get_model(self):
        model = Mobilenetv2()
        pretrained_state_dict=torch.load('./weights/mobilenet_v2-b0353104.pth', map_location=lambda storage, loc: storage)
        #重新映射key、value
        a = {}
        for i, j in zip(model.state_dict().keys(), pretrained_state_dict.values()):
            a[i] = j
        model.load_state_dict(a,strict=True)
        model.to(self.device)
        model.eval()
        return model

    def preprocess(self, image):
        #resize--ToTensor--norm
        transform=transforms.Compose([
            transforms.Resize(self.inputs_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        inputs=transform(image)
        inputs=torch.unsqueeze(inputs,dim=0)
        inputs=inputs.to(self.device)
        return inputs

    def postprocess(self, outputs):
        #取softmax得到每个类别的置信度
        outputs=torch.softmax(outputs,dim=1)
        #取最高置信度的类别和分数
        score, label_id = torch.max(outputs, dim=1)
        #Tensor ——> float
        score, label_id =score.item(), label_id.item()
        #查找标签名称
        label_name=self.label_names[label_id]
        return label_name, score
        
        
        

if __name__=='__main__':
    #分类推理模型
    model_classify=ModelPipline()

    #图片路径
    root='images'
    paths=os.listdir(root)
    #reference
    for name in paths:
        image=Image.open(os.path.join(root,name))
        result=model_classify.predict(image)
        print(result)

