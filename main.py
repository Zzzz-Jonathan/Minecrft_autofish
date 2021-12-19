from PIL import ImageGrab
from torchvision import transforms
from models import *
from utils.utils import *
from fish import Fish
import time

SIZE = 512
SCREEN_W = 0
SCREEN_H = 0
CHECK_TIME = 0


class TextPos:
    def __init__(self):
        self.loader = transforms.Compose([transforms.ToTensor()])
        self.model = Darknet("yolov3-custom.cfg", img_size=SIZE).to('cpu')
        self.model.load_state_dict(torch.load('weights.pth', map_location='cpu'))
        self.model.eval()

    def check_position(self):
        tensor = self.loader(grab(SIZE))
        tensor = tensor.unsqueeze(0)
        predict = non_max_suppression(self.model(tensor), 0.8, 0.1)
        if predict[0] is not None:
            predict = list(predict[0][0])

            # w = 2 * (predict[2] - predict[0])
            # h = 2 * (predict[3] - predict[1])
            # x = (predict[0] - 0.25 * w) * SCREEN_W / SIZE
            # y = (predict[1] - 0.25 * h) * SCREEN_H / SIZE
            #
            # if x + w >= SCREEN_W:
            #     w = SCREEN_W - x - 10
            # if y + h > SCREEN_H:
            #     h = SCREEN_H - y - 10
            x1, y1 = int(predict[0]) * SCREEN_W / SIZE, int(predict[1]) * SCREEN_H / SIZE
            x2, y2 = int(predict[2]) * SCREEN_W / SIZE, int(predict[3]) * SCREEN_H / SIZE
            w = x2 - x1
            h = max(y2 - y1, 150)

            return max(x1-w, 0), max(y1-h, 0), min(x2+w, SCREEN_W), min(y2+h, SCREEN_H)
        return 0, 0, 0, 0


def grab(size=None):  # works on macOS(RGBA) and Windows(RGB) only
    im = ImageGrab.grab()  # What region to copy. Default is the entire screen
    if size is not None:
        im = im.resize((size, size))

    return im.convert('RGB')


if __name__ == '__main__':
    SCREEN_W = grab().width
    SCREEN_H = grab().height

    pos = TextPos()
    fish = Fish()

    px1, py1, px2, py2 = 0, 0, 0, 0
    while (px1, py1, px2, py2) == (0, 0, 0, 0):
        px1, py1, px2, py2 = pos.check_position()
        time.sleep(5)
    CHECK_TIME = time.time()

    while True:
        if CHECK_TIME - time.time() > 300:
            px1, py1, px2, py2 = pos.check_position()
            CHECK_TIME = time.time()

        im = grab().crop((px1, py1, px2, py2))
        # im.save('./img/'+str(time.time())+'.jpg')
        img = np.array(im)
        fish.fish(img)
