import numpy as np
import torch.optim
import torch
import torchvision
from dataset import WFLW

from utils import get_color
import cv2 as cv

from model import FaceLandmark
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
model = FaceLandmark()
model = model.to(device)
model.load_state_dict(torch.load('models/weights-1', map_location=device, weights_only=True))

transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5,), (0.5,)),
])

landmark_transform = None

test_set = WFLW(
    'data/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt',
    'data/WFLW_images', transform=transform, target_transform=landmark_transform)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=None, shuffle=False,
                                          generator=torch.Generator(device=device))


def test_model():
    model.eval()
    running_test_loss = 0.

    with torch.no_grad():

        image = torchvision.io.decode_image('data/test-me.jpg')

        image = image[:, 150:750, 400:1000]
        original = cv.cvtColor(image.permute(1, 2, 0).numpy(), cv.COLOR_RGB2BGR)
        image = transform(image)
        landmarks = model(image.unsqueeze(0)).squeeze(0)

        for i in range(0 * 2, 98 * 2, 2):
            p = (landmarks[i:i + 2].numpy()).astype(np.int32)
            p[0] *= original.shape[0] / 224
            p[1] *= original.shape[1] / 224
            original = cv.drawMarker(original, p, get_color(i // 2), markerSize=20, thickness=2)

        cv.imshow('Face landmark estimation', original)
        cv.waitKey(0)

        for index, test_data in enumerate(test_loader):
            test_inputs, _, _, _ = test_data

            test_inputs = test_inputs.to(device)
            test_outputs = model(test_inputs.unsqueeze(0))

            image = torch.permute(test_inputs, (1, 2, 0)).numpy()
            image = ((image + 1) * 127.5).clip(0, 255).astype(np.uint8).copy()
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

            for i in range(0 * 2, 98 * 2, 2):
                p = (test_outputs[0][i:i + 2].numpy()).astype(np.int32)
                image = cv.drawMarker(image, p, get_color(i // 2), markerSize=10, thickness=1)

            cv.imshow('Face landmark estimation', image)
            cv.waitKey(0)


test_model()
