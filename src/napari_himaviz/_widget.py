"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

from napari.types import ImageData
from magicgui import magic_factory
import numpy as np
from napari.utils.notifications import show_info
import cv2
from skimage.transform import resize
from skimage import img_as_float
import tensorflow as tf
from tensorflow.keras import backend as K

class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")

# the coefficient takes values in [0, 1], where 0 is the worst score, 1 is the best score
# the dice coefficient of two sets represented as vectors a, b ca be computed as (2 *|a b| / (a^2 + b^2))
def dice_coefficient(y_true, y_pred):
    eps = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + eps)


def do_otsu(layer: ImageData) -> ImageData:
    
    img = cv2.cvtColor(layer, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh1

def do_segmentation(layer: ImageData) -> ImageData:
    if len(layer.shape) !=3:
        layer = np.array([layer]*3)
    X_img_test = (img_as_float(layer)[:,:,:3]*255).astype(np.uint8)
    X_img_test = resize(X_img_test, (256, 256), mode='constant', preserve_range=True).astype(np.uint8)
    print(X_img_test)
    model_path = r"/Users/valentingilet/Downloads/best_model.h5"
    model = tf.keras.models.load_model(model_path ,custom_objects={'dice_coefficient': dice_coefficient})
    preds_test = model.predict(np.expand_dims(X_img_test, axis=0), verbose=1)
    preds_test_t = (preds_test > 0.8).astype(np.uint8)
    return resize(preds_test_t[0,:,:,0], (layer.shape[0], layer.shape[1]), mode='constant', preserve_range=True)

@magic_factory(call_button="Run",radio_option={"widget_type": "RadioButtons",
                        "orientation": "vertical",
                        "choices": [("Otsu",1), ("U-Net",2)]})
def do_model_segmentation(
    layer: ImageData, radio_option=1
    ) -> ImageData:
    if radio_option==1:
        show_info('Succes !')
        return do_otsu(layer)
    if radio_option==2:
        show_info('Succes !')
        return do_segmentation(layer)
