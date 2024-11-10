import torch.optim
import torch
from dataset import WFLW
from wing_loss import WingLoss

from torch.utils.data import random_split

from model import FaceLandmark
from torchvision import transforms
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
EPOCHS = 200
model = FaceLandmark()
model = model.to(device)
loss_fn = WingLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.LRScheduler(optimizer) TODO: test how lr scheduling could improve results

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/face_landmark_estimation_trainer_{}'.format(timestamp))
epoch_number = 0

best_val_loss = 1_000_000.

transform = transforms.Compose([
    #transforms.ToTensor(), # not required anymore
    transforms.ConvertImageDtype(torch.float32),
    transforms.Resize((224, 224)),
    transforms.Normalize((0.5,), (0.5,)),
])

landmark_transform = None

dataset = WFLW(
    'data/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt',
    'data/WFLW_images', transform=transform, target_transform=landmark_transform)
test_set = WFLW(
    'data/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt',
    'data/WFLW_images', transform=transform, target_transform=landmark_transform)

train_set, val_set = random_split(dataset, [0.8, 0.2], generator=torch.Generator(device=device))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True,
                                           generator=torch.Generator(device=device))
val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True,
                                         generator=torch.Generator(device=device))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False,
                                          generator=torch.Generator(device=device))


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    for index, data in enumerate(train_loader):
        inputs, landmarks, bounding_box, attributes = data
        inputs = inputs.to(device)
        landmarks = landmarks.to(device)

        '''
        foo = torch.permute(inputs[0], (1, 2, 0)).numpy()
        foo = ((foo + 1) * 127.5).clip(0, 255).astype(np.uint8).copy()

        for i in range(0 * 2, 98 * 2, 2):
            p = (landmarks[0][i:i + 2].numpy()).astype(np.int32)
            foo = cv.drawMarker(foo, p, (0, 255, 0), markerSize=10, thickness=1)

        cv.imshow('foo', foo)
        cv.waitKey(0)
        '''

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, landmarks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if index == len(train_loader) - 1:
            last_loss = running_loss / len(train_loader)
            print('  batch {} loss: {}'.format(index + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + index + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def test_model():
    model.eval()
    running_test_loss = 0.

    with torch.no_grad():
        for index, test_data in enumerate(test_loader):
            test_inputs, test_landmarks, test_bounding_box, test_attributes = test_data

            test_inputs = test_inputs.to(device)
            test_landmarks = test_landmarks.to(device)

            test_outputs = model(test_inputs)
            test_loss = loss_fn(test_outputs, test_landmarks)
            running_test_loss += test_loss

    avg_test_loss = running_test_loss / len(test_loader)
    print('LOSS test {} '.format(avg_test_loss))

    writer.add_scalars('Test Loss',
                       {'Test': avg_test_loss},
                       epoch_number + 1)
    writer.flush()


for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_val_loss = 0.0
    model.eval()

    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            val_inputs, val_landmarks, val_bounding_box, val_attributes = val_data
            val_inputs = val_inputs.to(device)
            val_landmarks = val_landmarks.to(device)

            val_outputs = model(val_inputs)
            val_loss = loss_fn(val_outputs, val_landmarks)
            running_val_loss += val_loss

    avg_val_loss = running_val_loss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_val_loss))

    writer.add_scalars('Training vs. Validation Loss',
                       {'Training': avg_loss, 'Validation': avg_val_loss},
                       epoch_number + 1)
    writer.flush()

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

test_model()
