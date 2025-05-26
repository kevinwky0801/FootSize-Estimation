from sklearn import svm
import os
import cv2

class ImageClassifier:
    def __init__(self, retrain=False):
        self.dict_label = {
            'normal_img': 0,
            'white_background_img': 1,
            'skin_background_img': 2,
            'pattern_background_img': 3,
            'curve_background_img': 4
        }
        self.reverse_dict_label = {v: k for k, v in self.dict_label.items()}
        self.model = None
        if retrain or not os.path.exists("trained_model.pkl"):
            self.train()
        else:
            self.train()  # Optional: load pre-trained model if you implement persistence

    def train(self):
        X, y = [], []
        for folder in os.listdir('image_classified'):
            for img_path in os.listdir(f'image_classified/{folder}'):
                img = cv2.imread(f'image_classified/{folder}/{img_path}')
                img = cv2.resize(img, (128, 128))
                img = img / 255.0
                img = img.flatten()
                X.append(img)
                y.append(self.dict_label[folder])
        self.model = svm.SVC(gamma=0.001, C=100)
        self.model.fit(X, y)

    def predict(self, img_array):
        return self.model.predict([img_array])[0]
