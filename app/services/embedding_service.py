import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        base = models.resnet50(pretrained=False)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # remove FC
        self.fc = nn.Linear(2048, 256)

    def forward(self, x):
        x = self.backbone(x)            # [B, 2048, 1, 1]
        x = x.view(x.size(0), -1)       # [B, 2048]
        x = self.fc(x)                  # [B, 256]
        return F.normalize(x, p=2, dim=1)

class EmbeddingModel:
    def __init__(self, model_path: str = "models/resnet50_embedding.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EmbeddingNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def encode(self, image: Image.Image):
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(image_tensor)
        return features.squeeze(0).cpu().numpy()
