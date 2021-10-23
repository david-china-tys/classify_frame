import copy
import os

import onnx
import torch.nn as nn
import torch
import torchvision
from torchvision import transforms, datasets
import torch.utils.data
import torch.optim as optim
from PIL import Image
import torchvision.models as models
from tqdm import tqdm


# import onnxsim


def train():
    net.train()
    running_loss = 0.0
    count = len(train_loader) // 5
    for batch_idx, data in enumerate(train_loader, 1):
        inputs, target = data[0].to(device), data[1].to(device)

        outputs = net(inputs)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % count == 0:
            print('Epoch:{}/{} step:{}/{} loss: {:.3f}'.format(epoch, epochs, batch_idx + 1, len(train_loader),
                                                               running_loss / count))
            running_loss = 0.0


def val():
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %.2f %% ' % (100 * correct / total))

    return correct / total


def get_model(net_name):
    if net_name == 'mobilenet_v1':
        pass
        # net = MobilenetV1()
        # #修改fc层，类别改为2
        # fc_features = net.fc.in_features
        # net.fc = nn.Linear(fc_features, 2)
    elif net_name == 'mobilenet_v2':
        net = torchvision.models.mobilenet_v2(pretrained=False)
        state_dict = torch.load('../weights/mobilenet_v2-b0353104.pth')
        net.load_state_dict(state_dict, strict=True)

        for param in net.parameters():
            param.requires_grad = False
        new_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 2)
        )
        net.classifier = new_classifier

    elif net_name == 'resnet18':
        net = torchvision.models.resnet18(pretrained=False)
        state_dict = torch.load('../weights/resnet18-5c106cde.pth')
        net.load_state_dict(state_dict, strict=True)
        for param in net.parameters():
            param.requires_grad = False
        fc_inchannel = net.fc.in_features
        net.fc = nn.Linear(fc_inchannel, 2)

    # 查看每层的requires_grad
    # for i,j in net.named_parameters():
    #     print(i,j.shape,j.requires_grad)
    return net


def get_loss_function():
    loss_function = nn.CrossEntropyLoss()
    return loss_function


def get_optimization_function():
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # optimizer = optim.Adam(net.parameters(), lr=0.001)
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
    return optimizer


def model_export():
    # 导出pt模型
    torch.save(best_net, 'models/cat_dog.pt')

    # 导出onnx
    inputs = torch.ones([1, 3, 224, 224]).type(torch.float32).to(torch.device(device))

    torch.onnx.export(best_net, inputs, 'models/cat_dog.onnx', input_names=["input0"],
                      output_names=["output0"], verbose=False)

    # Checks
    model_onnx = onnx.load('models/cat_dog.onnx')  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx mode
    # print(onnx.helper.printable_graph(model_onnx.graph))


def inference():
    labels = ['cat', 'dog']
    acc = []
    for i in labels:
        path = 'data/cat_dog2000/' + i
        img_path = os.listdir(path)
        sum = 0
        for name in tqdm(img_path):
            image = Image.open(os.path.join(path, name))
            image = data_transform(image).to(device)
            image = image.unsqueeze(dim=0)
            out = best_net(image)
            out = torch.softmax(out, 1)
            _, pred = torch.max(out, 1)
            sum = sum + (labels[pred] == i)
        acc.append(sum / 1000)

    for i, j in zip(labels, acc):
        print('{}_acc:{:.2f}'.format(i, j))


if __name__ == '__main__':
    # 数据处理
    data_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data = datasets.ImageFolder(root='./data/cat_dog2000', transform=data_transform)

    # 查看类别对应的索引
    print(data.class_to_idx)

    # 划分训练集 验证集
    train_size = int(0.7 * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

    # 加载数据
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    # print(len(train_dataset))
    # print(len(test_dataset))

    # 超参数
    epochs = 10
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # #模型建立
    net = get_model('resnet18')
    net.to(device)

    # 损失函数
    criterion = get_loss_function()

    # 优化函数
    optimizer = get_optimization_function()

    best_net = None  # 记录效果最好的模型
    best = 0  # 最好的准确率
    # 模型训练
    for epoch in range(1, epochs + 1):
        train()
        acc = val()
        if acc > best:
            best = acc
            best_net = copy.deepcopy(net)

    # 保存模型前与最后测试前使用
    best_net.eval()

    # 模型导出.pt/.onnx
    model_export()

    # 用test_data下的数据进行最后测试
    inference()
