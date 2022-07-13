from argparse import ArgumentParser
import matplotlib
matplotlib.use("tkagg")

from equilib import equi2pers
#from py360convert import e2p as equirec2perspective
import cv2 as cv
import numpy as np

WINDOW_NAME = "Viewer"
WIDTH = 800
HEIGHT = 800
FOV = 90.0

def mouse_handler(event, x, y, flags, data):
    if event == cv.EVENT_LBUTTONUP and data["btn_down"]:
        data["btn_down"] = False
    elif event == cv.EVENT_MOUSEMOVE and data["btn_down"]:
        last_x = data["last_x"]
        last_y = data["last_y"]
        dx = x - last_x
        dy = y - last_y
        data["rots"]["yaw"] += np.deg2rad(dx)
        data["rots"]["pitch"] -= np.deg2rad(dy)
        data["last_x"] = x
        data["last_y"] = y
    elif event == cv.EVENT_LBUTTONDOWN:
        data["btn_down"] = True
        data["last_x"] = x
        data["last_y"] = y
    #pers_img = equirec2perspective(np.transpose(data["image"], (2, 0, 1)), (FOV,FOV), 
    #                               data["rots"]["yaw"], data["rots"]["pitch"], 
    #                               (HEIGHT,WIDTH), in_rot_deg=data["rots"]["roll"], mode='bilinear')
    pers_img = equi2pers(np.transpose(data["image"], (2, 0, 1)), data["rots"], HEIGHT, WIDTH, FOV)
    cv.imshow(WINDOW_NAME, np.transpose(pers_img, (1, 2, 0)))


def main():
    global WIDTH, HEIGHT, FOV
    parser = ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--width", default=WIDTH, type=int)
    parser.add_argument("--height", default=HEIGHT, type=int)
    parser.add_argument("--fov", default=FOV, type=float)
    args = parser.parse_args()

    print("""Welcome to the 360 degree viewer
    +-----+---------------+
    | key | Action        |
    +-----+---------------+
    | q   | Quit          |
    | a   | Rotate Left   |
    | w   | Rotate Up     |
    | s   | Rotate Down   |
    | d   | Rotate Right  |
    | r   | Zoom In       |
    | f   | Zoom Out      |
    | 1   | Face Forward  |
    | 2   | Face Right    |
    | 3   | Face Back     |
    | 4   | Face Left     |
    | 5   | Face Down     |
    | 6   | Face Up       |
    +-----+---------------+""")

    WIDTH = args.width
    HEIGHT = args.height
    FOV = args.fov

    rots = {
        "roll": 0,
        "pitch": 0,
        "yaw": 0
    }

    image = cv.imread(args.image)
    data = {
        "image": image,
        "last_x": 0,
        "last_y": 0,
        "rots": rots,
        "btn_down": False
    }

    cv.namedWindow(WINDOW_NAME)
    cv.setMouseCallback(WINDOW_NAME, mouse_handler, data)


    while True:
        
        #pers_img = equirec2perspective(np.transpose(image, (2, 0, 1)), (FOV,FOV), 
        #                       rots["yaw"], rots["pitch"], 
        #                       (HEIGHT,WIDTH), in_rot_deg=rots["roll"], mode='bilinear')
        pers_img = equi2pers(np.transpose(image, (2, 0, 1)), rots, HEIGHT, WIDTH, FOV)
        cv.imshow(WINDOW_NAME, np.transpose(pers_img, (1, 2, 0)))

        key = cv.waitKey()
        if key & 0xFF == ord("q"):
            break
        elif key & 0xFF in (ord("a"), 81):
            rots["yaw"] = rots["yaw"] + np.pi / 32
            if rots["yaw"] > 2*np.pi:
                rots["yaw"] -= 2 * np.pi
        elif key & 0xFF in (ord("d"), 83):
            rots["yaw"] = rots["yaw"] - np.pi / 32
            if rots["yaw"] < 0:
                rots["yaw"] += 2 * np.pi
        elif key & 0xFF == ord("s"):
            rots["pitch"] = rots["pitch"] + np.pi / 32
            if rots["pitch"] > 2 * np.pi:
                rots["pitch"] -= 2 * np.pi
        elif key & 0xFF == ord("w"):
            rots["pitch"] = rots["pitch"] - np.pi / 32
            if rots["pitch"] < 0:
                rots["pitch"] += 2 * np.pi
        elif key & 0xFF in (ord('r'), 82):
            FOV = max(FOV - 1, 0)
        elif key & 0xFF in (ord('f'), 84):
            FOV = min(FOV + 1, 135)
        elif key & 0xFF in (ord('1'),):
            rots["yaw"] = 0
            rots["pitch"] = 0
        elif key & 0xFF in (ord('2'),):
            rots["yaw"] = 3 * np.pi / 2
            rots["pitch"] = 0
        elif key & 0xFF in (ord('3'),):
            rots["yaw"] = np.pi
            rots["pitch"] = 0
        elif key & 0xFF in (ord('4'),):
            rots["yaw"] = np.pi / 2
            rots["pitch"] = 0
        elif key & 0xFF in (ord('5'),):
            rots["yaw"] = 0
            rots["pitch"] = np.pi/2
        elif key & 0xFF in (ord('6'),):
            rots["yaw"] = 0
            rots["pitch"] = 3 * np.pi / 2


if __name__ == "__main__":
    main()
