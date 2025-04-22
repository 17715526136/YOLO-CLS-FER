import torch
import torch.nn as nn

class PSA_Channel(nn.Module):
    def __init__(self, c1) -> None:
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1)
        self.cv2 = nn.Conv2d(c1, 1, 1)
        self.cv3 = nn.Conv2d(c_, c1, 1)
        self.reshape1 = nn.Flatten(start_dim=-2, end_dim=-1)
        self.reshape2 = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(0)  # 修改 softmax 操作为第一维度
        self.layernorm = nn.LayerNorm([c1, 1, 1])

    def forward(self, x): # shape(channel, height, width)
        print(f"Input to PSA_Channel: {x.shape}")
        x1 = self.reshape1(self.cv1(x)) # shape(channel/2, height*width)
        print(f"After cv1 and reshape1: {x1.shape}")
        x2 = self.softmax(self.reshape2(self.cv2(x))) # shape(height*width)
        print(f"After cv2 and reshape2 (softmax applied): {x2.shape}")
        y = torch.matmul(x1, x2.unsqueeze(-1)).unsqueeze(-1) # 高维度下的矩阵乘法（最后两个维度相乘）
        print(f"After matrix multiplication and unsqueeze: {y.shape}")
        output = self.sigmoid(self.layernorm(self.cv3(y))) * x
        print(f"Final PSA_Channel output: {output.shape}")
        return output


class PSA_Spatial(nn.Module):
    def __init__(self, c1) -> None:
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1)
        self.reshape1 = nn.Flatten(start_dim=-2, end_dim=-1)
        self.globalPooling = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(0)  # 修改 softmax 操作为第一维度
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # shape(channel, height, width)
        print(f"Input to PSA_Spatial: {x.shape}")
        x1 = self.reshape1(self.cv1(x)) # shape(channel/2, height*width)
        print(f"After cv1 and reshape1: {x1.shape}")
        x2 = self.softmax(self.globalPooling(self.cv2(x)).squeeze(-1)) # shape(channel/2)
        print(f"After cv2 and global pooling (softmax applied): {x2.shape}")
        y = torch.bmm(x2.unsqueeze(0).permute(0, 2, 1), x1.unsqueeze(0)) # shape(1, 1, height*width)
        print(f"After batch matrix multiplication: {y.shape}")
        output = self.sigmoid(y.view(1, 1, x.shape[1], x.shape[2])).squeeze(0) * x
        print(f"Final PSA_Spatial output: {output.shape}")
        return output


class PSA(nn.Module):
    def __init__(self, in_channel, parallel=False) -> None:
        super().__init__()
        self.parallel = parallel
        self.channel = PSA_Channel(in_channel)
        self.spatial = PSA_Spatial(in_channel)

    def forward(self, x):
        print(f"Initial input: {x.shape}")
        if self.parallel:
            output = self.channel(x) + self.spatial(x)
        else:
            output = self.spatial(self.channel(x))
        print(f"Final PSA output: {output.shape}")
        return output

# 测试代码
if __name__ == "__main__":
    # 定义一个输入张量大小
    input_tensor = torch.randn(64, 32, 32)  # 示例张量 (channel, height, width)

    model = PSA(64, parallel=False)
    output = model(input_tensor)
