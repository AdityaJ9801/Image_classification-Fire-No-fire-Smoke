# 🔥 Wildfire Image Classification using CNN (PyTorch)

This project is an **image classification model** built using a **custom dataset** created from multiple Kaggle sources.  
The model classifies images into **three categories**:
- **Fire** 🔥
- **Smoke** 💨
- **No Fire** 🌲

The goal of this project is to help in **early wildfire detection** using computer vision.

---

## 📂 Dataset
- Dataset was created by **combining images from several Kaggle wildfire datasets**.
- Classes:
  - `Fire` (images with flames)
  - `Smoke` (images with visible smoke, but no flames)
  - `No Fire` (normal forest/landscape images)

Dataset was preprocessed with:
- Resizing to `128x128`
- Normalization
- Train/validation split

---

## 🧠 Model Architecture
The model is a **Convolutional Neural Network (CNN)** built in **PyTorch**.

```python
class WildfireCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(WildfireCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 16 * 16, 256) 
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
