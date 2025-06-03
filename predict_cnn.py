import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

classes = ['lo_vi_song', 'may_giat', 'may_hut_bui', 'noi_com_dien', 'tivi', 'tu_lanh']
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("C:\project_root\cnn\model_cnn_inventory.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        label = classes[predicted.item()]
        confidence = torch.softmax(outputs, 1)[0][predicted.item()].item()
    return label, confidence
