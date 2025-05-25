import torch
from torchvision import transforms
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
import numpy as np
class SemanticExtractor:
    def __init__(self):
        #self.device = 'cuda'
        #self.model = shufflenet_v2_x0_5(pretrained=True).to(device=self.device)
        self.model = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
    def semantic_extraction(self, crop_image):
        output_tensor = self.model(crop_image)
        global_avg_pooled = F.adaptive_avg_pool2d(output_tensor, (1, 1))
        feature_vector = global_avg_pooled.view(global_avg_pooled.size(0), -1).squeeze()
        return feature_vector
    def crop(self,image,boxes):
        features = []
        transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        for box in boxes:
            left, top, right, bottom = box
            # 裁剪出目标区域
            cropped_image = image.crop((left, top, right, bottom))
            # 预处理图像
            preprocessed_image = transform(cropped_image)
            # 增加一个维度作为批处理维度
            #preprocessed_image = preprocessed_image.unsqueeze(0).to(self.device)
            preprocessed_image = preprocessed_image.unsqueeze(0)
        return preprocessed_image

    def cosine_similarity(self,vector1, vector2):
        vector1 = vector1.detach().numpy()
        vector2 = vector2.detach().numpy()
        dot_product = np.dot(vector1, vector2)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)
        
        similarity = dot_product / (norm_vector1 * norm_vector2)
        
        return similarity
    
if __name__ == "__main__":
# 假设目标检测结果存储在一个名为detections的列表中，每个元素是[label, left, top, right, bottom, score]的形式
    from PIL import Image
    from time import time

    detections = [[1, 100, 50, 200, 150, 0.95],  # 示例的检测结果
                [2, 250, 100, 350, 200, 0.92],
                [3, 50, 60, 150, 160, 0.89],
                [1, 300, 80, 400, 180, 0.91],
                [2, 180, 30, 280, 130, 0.94]]

    # 存储目标的语义特征的列表
    semantic_features = []

    # 加载图像（假设图像存储在image变量中）
    image = Image.open('E:/image_02/0000/000146.png')

    # 创建SemanticExtractor实例
    semantic_extractor = SemanticExtractor()

    # 对每个目标进行特征提取
    for detection in detections:
        label, left, top, right, bottom, score = detection
        t1 = time()
        # 裁剪图像并预处理
        cropped_image = semantic_extractor.crop(image, [(left, top, right, bottom)])
        t2 = time()
        # 提取语义特征
        feature_vector = semantic_extractor.semantic_extraction(cropped_image)
        t3 = time()
        # 将特征添加到列表中
        semantic_features.append(feature_vector)
        print((t2-t1)*1000, (t3-t2)*1000)

    sim = semantic_extractor.cosine_similarity(semantic_features[0],semantic_features[1])
    print(sim)
    # semantic_features现在包含了每个目标的语义特征
    print(len(semantic_features[0]))
    print(semantic_features[0])
