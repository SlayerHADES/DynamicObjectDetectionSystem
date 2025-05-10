
---

````markdown
# 📦 Dynamic Object Detection System

---

## 🔍 Features

- **Motion Detection:** Utilizes frame differencing to detect movement and define Regions of Interest (ROIs).
- **Image Classification:** Employs a Keras model exported from Teachable Machine to classify detected objects.
- **Real-Time Performance:** Achieves 10–15 FPS on standard CPUs.
- **High Accuracy:** Maintains over 90% detection accuracy with minimal false positives.
- **Resource Efficient:** Operates with low CPU and memory usage—ideal for edge devices like Raspberry Pi.

---

## 🛠️ Technologies Used

- Python 3.8+
- OpenCV
- TensorFlow / Keras
- Teachable Machine

---

## 🚀 How to Run

1. **Clone this repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
````

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add your model files:**

   * `keras_model.h5`
   * `labels.txt`

   *(Exported from [Teachable Machine](https://teachablemachine.withgoogle.com/))*

4. **Run the script:**

   ```bash
   python main.py
   ```

---

## 📂 Project Structure

```
├── main.py              # Main script to run object detection
├── keras_model.h5       # Pre-trained model (Teachable Machine)
├── labels.txt           # Class labels for the model
├── requirements.txt     # Python dependencies
```

---

## 🤖 Example Use Cases

* Smart Surveillance Systems
* Motion-triggered Object Identification
* Lightweight Edge AI Applications (e.g., Raspberry Pi)

---

## 💡 Future Improvements

* Add bounding boxes using detection models (YOLO/SSD)
* Deploy via Flask web interface
* Add object tracking for persistent IDs

---

## 🙌 Acknowledgements

* [Teachable Machine](https://teachablemachine.withgoogle.com/)
* [OpenCV](https://opencv.org/)
* [TensorFlow](https://www.tensorflow.org/)

---

## 📬 Contact

Feel free to connect or suggest improvements!

**Kshitij Jaiswal**
📧 [kshitijjaiswal0205@gmail.com](mailto:kshitijjaiswal0205@gmail.com)


