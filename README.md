# EmotionDetection

This project aims to detect emotions from facial expressions using a Convolutional Neural Network (CNN) model.

## Dataset

The dataset used for training and testing the model can be found [here](https://www.kaggle.com/datasets/msambare/fer2013).

## Requirements

- Python 3.9
- OpenCV
- MediaPipe
- PyTorch
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/EmotionDetection.git
    cd EmotionDetection
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

To train the model, run the Jupyter notebook `Model_Training.ipynb`:
```bash
jupyter notebook Model_Training.ipynb
```

### Running the Application

To run the emotion detection application, execute the `App.py` script:
```bash
python App.py
```

## Model Architecture

The CNN model consists of three convolutional layers followed by max-pooling layers, a fully connected layer, and a dropout layer. The model is trained using the CrossEntropyLoss and the Adam optimizer.

## Results

The training and validation accuracy and loss are plotted over epochs to visualize the model's performance.

## License

This project is licensed under the MIT License.
