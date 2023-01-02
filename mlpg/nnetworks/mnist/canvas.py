import cv2
import numpy as np
import torch

from model import MNISTModel

CANVAS_SIZE = 500
IMG_SIZE = 28
OFFSET = 100
CELL_SIZE = 10
BLACK = 0
WHITE = 255


class Canvas(np.ndarray):
    @property
    def image(self):
        return self[OFFSET : OFFSET + IMG_SIZE * CELL_SIZE, OFFSET : OFFSET + IMG_SIZE * CELL_SIZE]

    @image.setter
    def image(self, value):
        self[OFFSET : OFFSET + IMG_SIZE * CELL_SIZE, OFFSET : OFFSET + IMG_SIZE * CELL_SIZE] = value

    @classmethod
    def new(self):
        canvas = Canvas((CANVAS_SIZE, CANVAS_SIZE), dtype='uint8')
        canvas.fill(WHITE)
        canvas.image = BLACK
        return canvas

    def show_text(self, text):
        self[0:60, OFFSET : OFFSET + 250] = WHITE
        cv2.putText(self, text, (OFFSET, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


canvas = Canvas.new()
start_point = None
end_point = None
is_drawing = False


def draw_line(img, start_at, end_at):
    cv2.line(img, start_at, end_at, WHITE, 15)


def on_mouse_events(event, x, y, flags, params):
    global start_point
    global end_point
    global canvas
    global is_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        start_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing and start_point:
            end_point = (x, y)
            draw_line(canvas, start_point, end_point)
            start_point = end_point
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False
        start_point = None


def main():
    global is_drawing, canvas

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MNISTModel().to(device)
    model.load_state_dict(torch.load('model.pth'))

    cv2.namedWindow('MNIST Test')
    cv2.setMouseCallback('MNIST Test', on_mouse_events)

    while True:
        cv2.imshow('MNIST Test', canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas.image = BLACK
            is_drawing = False
        elif key == ord('p'):
            is_drawing = False
            image = torch.zeros((IMG_SIZE, IMG_SIZE)).to(device).float()
            for i in range(IMG_SIZE):
                for j in range(IMG_SIZE):
                    image[i, j] = canvas.image[i * CELL_SIZE, j * CELL_SIZE]
            pred = model(image.view(1, 1, 28, 28))
            canvas.show_text(f'PREDICTION : {pred.argmax(1).item()}')

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
