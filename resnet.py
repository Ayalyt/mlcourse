import torch
import torch.nn as nn
from torchsummary import summary

# 分类数目
num_class = 10
# 各层数目
resnet18_params = [2, 2, 2, 2]
resnet34_params = [3, 4, 6, 3]
resnet50_params = [3, 4, 6, 3]
resnet101_params = [3, 4, 23, 3]
resnet152_params = [3, 8, 36, 3]

# 浅层的残差结构
class BasicBlock(nn.Module):
    def __init__(self,in_places,places, stride=1, downsampling=False, expansion = 1):
        super(BasicBlock,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.basicblock = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        # 每个大模块的第一个残差结构需要改变步长
        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 实线分支
        residual = x
        out = self.basicblock(x)

        # 虚线分支
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


# 深层的残差结构
class Bottleneck(nn.Module):

    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 实线分支
        residual = x
        out = self.bottleneck(x)

        # 虚线分支
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,blocks, blockkinds, num_classes=num_class):
        super(ResNet,self).__init__()
        self.device="cuda"
        self.blockkinds = blockkinds
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        # 对应浅层网络结构
        if self.blockkinds == BasicBlock:
            self.expansion = 1
            # 64 -&gt; 64
            self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=1)
            # 64 -&gt; 128
            self.layer2 = self.make_layer(in_places=64, places=128, block=blocks[1], stride=2)
            # 128 -&gt; 256
            self.layer3 = self.make_layer(in_places=128, places=256, block=blocks[2], stride=2)
            # 256 -&gt; 512
            self.layer4 = self.make_layer(in_places=256, places=512, block=blocks[3], stride=2)
        # 对应深层网络结构
        elif self.blockkinds == Bottleneck:
            self.expansion = 4
            # 64 -&gt; 64
            self.layer1 = self.make_layer(in_places = 64, places= 64, block=blocks[0], stride=1)
            # 256 -&gt; 128
            self.layer2 = self.make_layer(in_places = 256,places=128, block=blocks[1], stride=2)
            # 512 -&gt; 256
            self.layer3 = self.make_layer(in_places=512,places=256, block=blocks[2], stride=2)
            # 1024 -&gt; 512
            self.layer4 = self.make_layer(in_places=1024,places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

        # 初始化网络结构
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 采用了何凯明的初始化方法
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):

        layers = []
        # 此步需要通过虚线分支，downsampling=True
        layers.append(self.blockkinds(in_places, places, stride, downsampling =True))
        # 此步需要通过实线分支，downsampling=False， 每个大模块的第一个残差结构需要改变步长
        for i in range(1, block):
            layers.append(self.blockkinds(places*self.expansion, places))
        return nn.Sequential(*layers)


    def forward(self, x):

        # conv1层
        x = self.conv1(x) 

        # conv2_x层
        x = self.layer1(x) 
        # conv3_x层
        x = self.layer2(x) 
        # conv4_x层
        x = self.layer3(x) 
        # conv5_x层
        x = self.layer4(x) 

        x = self.avgpool(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)

        return x

def ResNet18():
    return ResNet(resnet18_params, BasicBlock)

def ResNet34():
    return ResNet(resnet34_params, BasicBlock)

def ResNet50():
    return ResNet(resnet50_params, Bottleneck)

def ResNet101():
    return ResNet(resnet101_params, Bottleneck)

def ResNet152():
    return ResNet(resnet152_params, Bottleneck)


if __name__=='__main__':
    # model = torchvision.models.resnet50()

    # 模型测试
    # model = ResNet18()
    # model = ResNet34()
    # model = ResNet50()
    model = ResNet101()
    model = ResNet152()
    # model = torchvision.models.resnet152()
    model.to("cuda")
    # print(model)

    input = torch.randn(1, 1, 28, 28).to(device=model.device)
    out = model(input)
    print(out.shape)
    summary(model, input_size=(1, 28, 28))
    print(model)