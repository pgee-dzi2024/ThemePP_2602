

from PIL import Image
import torch
from torchvision import models, transforms

# 1) Зареди предобучения модел ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()  # режим на инференс

# 2) Дефинирай трансформациите за вход
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# 3) Ако имаш списък с ImageNet класове (1000 класове), може да го зарешеш
# Как да получиш labels: може да използваш файл с класовете от ImageNet
# Тук предполагаме, че labels е списък от 1000 стринга, напр.:
# labels = [ "tench", "goldfish", ..., "work table" ]
# За демонстрация ще използваме декодиране чрез `torchvision.datasets.ImageNet`-подобен подход
# Но най-лесно е да имаш локално файлче 'imagenet_class_index.json' или similar.

# Ако имаш json с mapping
# import json
# with open('imagenet_class_index.json') as f:
#     class_idx = json.load(f)
#     labels = [class_idx[str(k)][1] for k in range(len(class_idx))]

def predict_image(image_path, top_k=5):
    # 4) Зареди изображението
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # добави batch dimension

    # 5) Инференс
    with torch.no_grad():
        output = model(input_batch)

    # 6) Вземи топ-k класове
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, top_k)

    # 7) Преобразуване в Python списък
    results = []
    for prob, idx in zip(top_prob, top_catid):
        # Ако имаш labels:
        # results.append({"label": labels[idx.item()], "probability": float(prob)})
        results.append({"class_id": int(idx.item()), "probability": float(prob.item())})

    return results

# Пример за използване:
if __name__ == "__main__":
    img_path = "path/to/your/image.jpg"  # заменете с реален път
    top = predict_image(img_path, top_k=5)
    print("Top-5 predictions:")
    for r in top:
        print(f"Class ID: {r['class_id']}, Probability: {r['probability']:.4f}")