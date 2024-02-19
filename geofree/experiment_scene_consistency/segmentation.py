import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import torchvision.transforms as transforms
import torch
from semseg.models import *

class Segmentation():
    def __init__(self) -> None:
        self.model = eval('SegFormer')(
        backbone='MiT-B3',
        num_classes=150
        )
        self.PALETTE = np.array([
        [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
        [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7], [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
        [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3], [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
        [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220], [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
        [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255], [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
        [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153], [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
        [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0], [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
        [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255], [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
        [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0], [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
        [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255], [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
        [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20], [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
        [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255], [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
        [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0], [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
        [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255], [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
        [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160], [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
        [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0], [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
        [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255], [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
        [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255], [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
        [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194], [102, 255, 0], [92, 0, 255]
        ])
        try:
            self.model.load_state_dict(torch.load('../../semantic-segmentation/checkpoints/pretrained/segformer/segformer.b3.ade.pth', map_location='cpu'))
        except:
            print("Download a pretrained model's weights from the result table.")
        self.model.eval()
    
        print('Loaded Model')

    def decode_segmap(self,image, nc=150):
        label_colors = self.PALETTE
        r = image.copy()
        g = image.copy()
        b = image.copy()
        for l in range(0, nc):
            r[image == l] = label_colors[l, 0]
            g[image == l] = label_colors[l, 1]
            b[image == l] = label_colors[l, 2]
        rgb = np.zeros((image.shape[0], image.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0

        rgb = np.array(rgb)
        return rgb

    def get_segmentaion_map(self, content_image):
        content_image = transforms.ToTensor()(content_image).unsqueeze(0)
        with torch.inference_mode():
            seg_map = self.model(content_image)
        seg_map = seg_map.squeeze(0).detach().cpu().numpy().argmax(0)
        return seg_map

    def get_clicked_pixel(self,image_np):
        height, width, _ = image_np.shape

        root = tk.Tk()
        root.title("Image Clicker")
        root.geometry(f"{width}x{height}")

        result = {'x': None, 'y': None}  # To store the result

        def on_click(event):
            result['x'], result['y'] = event.x, event.y
            root.destroy()  # Close the window when clicked

        canvas = tk.Canvas(root)
        canvas.pack(fill=tk.BOTH, expand=True)
        image_np = image_np * 255
        image_np = image_np.astype(np.uint8)
        image = Image.fromarray(image_np)
        tk_image = ImageTk.PhotoImage(image)

        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)

        canvas.bind("<Button-1>", on_click)

        root.mainloop()

        return result['x'], result['y']