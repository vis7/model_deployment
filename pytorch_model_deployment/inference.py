"""
pytorch code for making classification using pre-trained model
"""

import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import json

# loading densenet 121 model
model = models.densenet121(pretrained=True)
# Since we are using our model only for inference, switch to `eval` mode:
model.eval()

imagenet_class_index = json.load(open('static/imagenet_class_index.json'))

# transforming image
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]

# testing image_net_class file
with open("static/img/pop_ww.jpg", 'rb') as f:
    image_bytes = f.read()
    print(get_prediction(image_bytes=image_bytes))

