# FacePose

FacePose is a Python library for **real-time 3D head pose estimation**.  
It combines **SynergyNet** and **6DRepNet360** to predict:

- Head rotation angles (Yaw, Pitch, Roll)  
- Nose landmark coordinates in the image  

It works on single images, video files, or live webcam streams.

---

## ✨ Features

- Accurate **3D head pose estimation**  
- Combines **landmarks + rotation matrices** for robust results  
- Supports **three inference modes**:  
  - **Mode 1**: Uses both SynergyNet and 6DRepNet together  
  - **Mode 2**: SynergyNet only  
  - **Mode 3**: Alternates between both models  
- Built-in **temporal smoothing** using frame weighting  
- Live **visualization with OpenCV**

---

## 📦 Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/FacePose.git
cd FacePose

python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

Download the pretrained model weights and place them in `pretrained/`:

- `best.pth.tar` (SynergyNet)  
- `6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth` (6DRepNet360)  

---

## 🚀 Usage

### 1. Initialize
```python
import cv2
from facepose import FacePose

fp = FacePose(
    synergymodel_path="pretrained/best.pth.tar",
    sixdrepnetmodel_path="pretrained/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth"
)
```

### 2. Run on an Image
```python
img = cv2.imread("face.jpg")
pose = fp.get_pose(img, mode=1)

print("Yaw, Pitch, Roll:", pose[0])
print("Nose coordinates:", pose[1])
```

### 3. Run on a Webcam
```python
cap = cv2.VideoCapture(0)
poses = fp.video_pose(cap, mode=1, show=True, frame_num_max=200)
```

---

## ⚙️ Parameters

- `mode`  
  - `1`: SynergyNet + 6DRepNet  
  - `2`: SynergyNet only  
  - `3`: Alternate between models  

- `frame_weights`:  
  Controls smoothing across frames. Example:  
  ```python
  frame_weights=[0.6, 0.3, 0.1]
  ```

---

## 📂 Project Structure

```
FacePose/
│── pretrained/                       # Model checkpoints
│── utils/                            # Helper functions
│── FaceBoxes/                        # Face detector
│── facepose.py                       # Main implementation
│── requirements.txt
│── README.md
```

---

## 🧠 Models

- **SynergyNet**  
  [SynergyNet: Joint 3D Face Shape and Expression Recovery with Synthetic Data](https://arxiv.org/abs/2107.05275)

- **6DRepNet**  
  [6DRepNet: Category-level 6D Pose Estimation via 6D Rotation Representation](https://arxiv.org/abs/2208.03614)

---

## 📜 Citation

If you use this repository, please cite the original papers:

```bibtex
@article{SynergyNet2021,
  title={SynergyNet: Joint 3D Face Shape and Expression Recovery with Synthetic Data},
  author={Wu, Xiangyu and others},
  year={2021},
  journal={arXiv preprint arXiv:2107.05275}
}

@article{6DRepNet2022,
  title={6DRepNet: Category-level 6D Pose Estimation via 6D Rotation Representation},
  author={He, Shih-Yao and others},
  year={2022},
  journal={arXiv preprint arXiv:2208.03614}
}
```

---

## 📌 To-Do

- [ ] Multi-face support  
- [ ] Optimized inference for edge devices  
- [ ] Jupyter demo notebook  

---

## 🤝 Contributing

Contributions are welcome.  
Open an issue first to discuss changes or improvements.

---

## 📄 License

This project is released under the **MIT License**.
