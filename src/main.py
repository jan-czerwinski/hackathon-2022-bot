from typing import Tuple, Any

import numpy as np
import cv2
from mss import mss
from PIL import Image
from pynput.keyboard import Key, Controller, Listener
import math


class Vector2D:
    """A two-dimensional vector with Cartesian coordinates."""

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        """Human-readable string representation of the vector."""
        return '{:g}i + {:g}j'.format(self.x, self.y)

    def __repr__(self):
        """Unambiguous string representation of the vector."""
        return repr((self.x, self.y))

    def dot(self, other):
        """The scalar (dot) product of self and other. Both must be vectors."""

        if not isinstance(other, Vector2D):
            raise TypeError(
                'Can only take dot product of two Vector2D objects')
        return self.x * other.x + self.y * other.y

    # Alias the __matmul__ method to dot so we can use a @ b as well as a.dot(b).
    __matmul__ = dot

    def __sub__(self, other):
        """Vector subtraction."""
        return Vector2D(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        """Vector addition."""
        return Vector2D(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar):
        """Multiplication of a vector by a scalar."""

        if isinstance(scalar, int) or isinstance(scalar, float):
            return Vector2D(self.x * scalar, self.y * scalar)
        raise NotImplementedError('Can only multiply Vector2D by a scalar')

    def __rmul__(self, scalar):
        """Reflected multiplication so vector * scalar also works."""
        return self.__mul__(scalar)

    def __neg__(self):
        """Negation of the vector (invert through origin.)"""
        return Vector2D(-self.x, -self.y)

    def __truediv__(self, scalar):
        """True division of the vector by a scalar."""
        return Vector2D(self.x / scalar, self.y / scalar)

    def __mod__(self, scalar):
        """One way to implement modulus operation: for each component."""
        return Vector2D(self.x % scalar, self.y % scalar)

    def __abs__(self):
        """Absolute value (magnitude) of the vector."""
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def distance_to(self, other):
        """The distance between vectors self and other."""
        return abs(self - other)

    def to_polar(self):
        """Return the vector's components in polar coordinates."""
        return self.__abs__(), math.atan2(self.y, self.x)


class GameObject:
    def __init__(self, position: Vector2D):
        self.direction = 1
        self.position = position

    def updatePosition(self, new_pos: Vector2D) -> Vector2D:
        self.direction = 1 if new_pos.x > self.position.x else -1
        self.position = new_pos
        return self.position

    def getPosition(self) -> Vector2D:
        return self.position

    def getPositionTuple(self) -> tuple[Any, Any]:
        return int(self.position.x), int(self.position.y)

    def getHeight(self):
        pass  # TODO return height from the player

    def getDirection(self):
        return self.direction


class Enemy(GameObject):
    def __init__(self, position: Vector2D):
        super().__init__(position)


class Bullet(GameObject):
    def __init__(self, position: Vector2D):
        super().__init__(position)


class Player(GameObject):
    def __init__(self, position: Vector2D):
        super().__init__(position)

    # TODO move player


class Game:
    keyboard = None
    sct = None
    bounding_box = {'top': 8, 'left': 8, 'width': 960, 'height': 540}

    def __init__(self):
        self.keyboard = Controller()
        self.sct = mss()
        self.grabbedFrame = None
        self.detectedContours = None

        self.bullets = []
        self.enemies = []
        self.player = Player(Vector2D(0, 0))

    def move_player(self, move=None):
        if move is None:
            return

        if move == 'left':
            self.keyboard.press(Key.left)
            self.keyboard.release(Key.left)

        elif move == 'right':
            self.keyboard.press(Key.right)
            self.keyboard.release(Key.right)

        elif move == 'shoot':
            self.keyboard.press(Key.space)
            self.keyboard.release(Key.space)

    def showDebugImage(self):
        debug_img = self.grabbedFrame
        for cnt in self.detectedContours:
            (x, y, w, h) = cnt
            cv2.rectangle(debug_img,
                          (x, y), (x + w, y + h),
                          (0, 255, 0),
                          2)
        font = cv2.FONT_HERSHEY_SIMPLEX


        font_scale = 0.7
        color = (255, 0, 255)
        cv2.putText(debug_img, 'pl', self.player.getPositionTuple(), font,
                            font_scale, color, 1, cv2.LINE_AA)

        for bullet in self.bullets:
            cv2.putText(debug_img, 'bul', bullet.getPositionTuple(), font,
                        font_scale, color, 1, cv2.LINE_AA)

        for enemy in self.enemies:
            cv2.putText(debug_img, 'enm', enemy.getPositionTuple(), font,
                        font_scale, color, 1, cv2.LINE_AA)
        cv2.imshow("debug image", debug_img)
        
    def detect_contours(self):
        gray = cv2.cvtColor(self.grabbedFrame, cv2.COLOR_BGR2GRAY)

        ret, thresh1 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

        # Find the contours
        contours, hierarchy = cv2.findContours(
            thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        self.detectedContours = [cv2.boundingRect(contour) for contour in contours]

    def updatePositions(self):
        enemies_positions = [ob for ob in self.detectedContours if 50 < ob[2] < 70]  # get big objects in x dir
        bullets_positions = [ob for ob in self.detectedContours if 6 < ob[2] < 15]  # get smol objects in x dir

        player_idx, player_contour = max(enumerate(enemies_positions), key=lambda x: x[1][1])
        enemies_positions.pop(player_idx)
        self.enemies = [Enemy(Vector2D(cont[0] + cont[2] / 2, cont[1] + cont[3] / 2)) for cont in enemies_positions]
        self.bullets = [Bullet(Vector2D(cont[0] + cont[2] / 2, cont[1] + cont[3] / 2)) for cont in bullets_positions]

        # if len(player_contour) > 1:
        #     print("DETECTED MORE THEN 1 PLAYER")
        # print(player_contour)
        self.player = Player(Vector2D(player_contour[0], player_contour[1]))

    def getFrame(self):
        frame_raw = self.sct.grab(self.bounding_box)
        # self.grabbedFrame = np.array(frame_raw)
        self.grabbedFrame = cv2.imread("sample.png")
        cv2.rectangle(self.grabbedFrame, (0, 0), (80, 50), (0,0,0), -1)

    def main(self):
        self.getFrame()

        while True:
            self.getFrame()
            self.detect_contours()
            # move_player(keyboard, 'shoot')
            self.updatePositions()
            self.showDebugImage()

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    game = Game()
    game.main()
