from PIL import ImageGrab
from torchvision import transforms
from models import *
from utils.utils import *
import time

SIZE = 512
im_test = ImageGrab.grab()
SCREEN_W, SCREEN_H = im_test.width, im_test.height


def grab():  # works on macOS(RGBA) and Windows(RGB) only
    im = ImageGrab.grab()  # What region to copy. Default is the entire screen
    im = im.resize((SIZE, SIZE))
    return im.convert('RGB')


if __name__ == '__main__':
    loader = transforms.Compose([transforms.ToTensor()])
    # start = 100
    # while True:
    #     grab(str(start))
    #     start += 1
    #     time.sleep(1)
    model = Darknet("./config/yolov3-custom.cfg", img_size=SIZE).to('cpu')
    model.load_state_dict(torch.load('./checkpoints/yolov3_ckpt_99.pth', map_location='cpu'))
    model.eval()

    test_data = []
    num, width_mean, height_min = 0, 0, 999
    # position = []
    # height = []
    for i in range(5):
        tensor = loader(grab())
        tensor = tensor.unsqueeze(0)
        predict = non_max_suppression(model(tensor), 0.8, 0.2)
        if predict[0] is not None:
            data = predict[0][0]
            w = data[2] - data[0]
            h = data[3] - data[1]
            num += 1
            # position.append([data[0]*SCREEN_W/SIZE, data[1]*SCREEN_H/SIZE])
            # height.append(data[3])
            width_mean += w
            if h < height_min:
                height_min = h

    width = int(width_mean / num)
    print("START, width: %f, height: %f." % (width * SCREEN_W / SIZE, height_min * SCREEN_H / SIZE))

    while True:
        im = ImageGrab.grab()
        img = im.resize((SIZE, SIZE))
        img.convert('RGB')

        img_crop = []
        tensor = loader(img)
        tensor = tensor.unsqueeze(0)
        predict = non_max_suppression(model(tensor), 0.8, 0.2)
        if predict[0] is not None:
            data = predict[0][0]
            w = data[2] - data[0]
            h = data[3] - data[1]

            num = math.ceil(h / height_min) if math.floor(h / height_min) > 1 else 1
            delta = h / num
            delta = int(delta) * SCREEN_H / SIZE
            pos = [data[0] * SCREEN_W / SIZE, data[1] * SCREEN_H / SIZE, data[2] * SCREEN_W / SIZE,
                   data[3] * SCREEN_H / SIZE]

            box = (int(pos[0]), int(pos[1]), int(pos[2]), int(pos[3]))
            im = im.crop(box)
            wd = im.width - width * SCREEN_W / SIZE

            for i in range(num):
                inf_box = (wd / 2, i * delta, width * SCREEN_W / SIZE + wd / 2, (i + 1) * delta)
                inf = im.crop(inf_box)
                inf.save(str(time.time()) + '.png')
                # img_crop.append(im.crop(box))

