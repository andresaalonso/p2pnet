import torch
import torchvision.transforms as standard_transforms
import numpy as np
import cv2
import os
from P2PNetDetector.models import build_model, vgg_
import urllib.request

class P2PNet:
    def __init__(self, backbone="vgg16_bn", row=2, weight_path=os.getcwd()+"/P2PNetDetector/weights/SHTechA.pth", gpu_id=0, output_dir="./logs/", threshold=0.5):
        if backbone not in ["vgg16_bn"]:
            raise NotImplementedError("Backbone not admitted. Available: vgg16_bn")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.backbone_name = backbone
        self.args = self._create_args(backbone, row, weight_path, gpu_id)
        self.model = self._load_model()
        self.transform = self._get_transform()

    def _create_args(self, backbone, row, weight_path, gpu_id):
        class Args:
            pass
        args = Args()
        args.backbone = backbone
        args.row = row
        args.output_dir = self.output_dir
        args.weight_path = weight_path
        args.gpu_id = gpu_id
        args.line = 2
        return args

    def _load_model(self):
        if not os.path.exists(vgg_.model_paths[self.backbone_name]):
            print(f"Backbone not found in '{vgg_.model_paths[self.backbone_name]}'. Downloading from '{vgg_.model_urls[self.backbone_name]}'...")

            os.makedirs(os.path.dirname(vgg_.model_paths[self.backbone_name]), exist_ok=True)
            try:
                urllib.request.urlretrieve(vgg_.model_urls[self.backbone_name], vgg_.model_paths[self.backbone_name])
                print(f"Download completed and saved in '{vgg_.model_paths[self.backbone_name]}'")
            except Exception as e:
                print(f"Error during download: {e}")

        model = build_model(self.args).to(self.device)
        checkpoint = torch.load(self.args.weight_path, map_location=self.device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        return model

    def _get_transform(self):
        return standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        new_width = (width + 127) // 128 * 128
        new_height = (height + 127) // 128 * 128
        frame_resized = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        img = self.transform(frame_resized).unsqueeze(0).to(self.device)
        return frame_resized, img

    def predict(self, img):
        outputs = self.model(img)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]
        
        points = outputs_points[outputs_scores > self.threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > self.threshold).sum())
        return points, predict_cnt

    def draw_predictions(self, frame, points):
        for p in points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
        # output_path = os.path.join(self.output_dir, f'pred{predict_cnt}.jpg')
        # cv2.imwrite(output_path, frame)
        return frame

    def run(self, frame, draw_predictions=False):
        """
        Main function to run the P2PNet detector on a given frame.

        Args:
            frame (numpy.ndarray): Input frame in RGB format.

        Returns:
            tuple: (numpy.ndarray, list): Output frame with predictions drawn, and a list of detected points.
        
        1. Preprocess the frame using the preprocess_frame method
        2. Predict the points using the predict method
        3. Draw the predictions on the frame using the draw_predictions method
        4. Return the modified frame and the list of detected points
        # Example:
        # frame_resized, img = self.preprocess_frame(frame)
        # points, predict_cnt = self.predict(img)
        # frame = self.draw_predictions(frame_resized, points, predict_cnt)
        """
        frame_resized, img = self.preprocess_frame(frame)
        points, predict_cnt = self.predict(img)
        if draw_predictions:
            frame = self.draw_predictions(frame_resized, points, predict_cnt)
        return frame, points

    @staticmethod
    def transform_to_plot(puntos, img_size):
        """
        Transforma las coordenadas de imagen a coordenadas de Matplotlib.
        
        Args:
        - puntos: Lista de tuplas (x, y) con coordenadas originales.
        - altura_img: Altura total de la imagen en píxeles.
        
        Returns:
        - Lista de tuplas (x, y') con el eje Y invertido.
        """
        return [(img_size[1] - x, img_size[0] - y) for x, y in puntos]



# Ejemplo de uso
if __name__ == '__main__':
    detector = P2PNetDetector()
    frame = cv2.imread("./vis/IMG_2.jpg")
    output = detector.run(frame)
    print(f"Resultado guardado en: {output}")
