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

    def updatePosition(self, new_pos: Vector2D):
        self.direction = 1 if new_pos.x > self.position.x else -1
        self.position = new_pos
        return self.position

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
    def __init__(self):
        self.keyboard = Controller()
        self.sct = mss()
        
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


    def debugImage(img, contours):
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img,
                        (x, y), (x + w, y + h),
                        (0, 255, 0),
                        2)

    def detect_rectangles(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, thresh1 = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

        # Find the contours
        contours, hierarchy = cv2.findContours(
            thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # for cnt in contours:
        #     x, y, w, h = cv2.boundingRect(cnt)
        #     cv2.rectangle(img,
        #                   (x, y), (x + w, y + h),
        #                   (0, 255, 0),
        #                   2)

        return [cv2.boundingRect(contour) for contour in contours], img
    
    def labelObjects(self, contours):
        self.enemiesPositions = [ob for ob in contours if 50 <
                        ob[2] < 70]  # get big objects in x dir
        bulletsPositions = [ob for ob in contours if 4 <
                        ob[2] < 8]  # get smol objects in x dir

        playerPosition = max(contours, key=lambda x: x[1])

        return playerPosition, enemiesPositions, bulletsPositions

    def updateEnemiesPositions(self, contours):
        return [Enemy(Vector2D(cont[0] + cont[2]/2, cont[1] + cont[3]/2)) for cont in contours]


    def updateBulletsPositions(self, contours):
        return [Bullet(Vector2D(cont[0] + cont[2]/2, cont[1] + cont[3]/2)) for cont in contours]


    def updatePlayerPosition(self, contour):
        if len(contour) > 1:
            print("DETECTED MORE THEN 1 PLAYER")
        return Player(Vector2D(contour[0], contour[1]))


    def updatePositions(self, contours):
        player_position, enemies_positions, bullets_positions = labelObjects(
            contours)
        print(player_position, bullets_positions, enemies_positions)

        enemies = updateEnemiesPositions(enemies_positions)
        bullets = updateBulletsPositions(bullets_positions)
        player = updatePlayerPosition(player_position)
        return player, enemies, bullets

    def main(self):

        bounding_box = {'top': 8, 'left': 8, 'width': 960, 'height': 540}

        frame_raw = sct.grab(bounding_box)
        frame = np.array(frame_raw)

        while True:
            frame_raw = sct.grab(bounding_box)
            frame = np.array(frame_raw)
            detect_rectangles(frame)

            # move_player(keyboard, 'shoot')
            frame = cv2.imread("sample.png")
            contours, img = detect_rectangles(frame)
            cv2.imshow('debug image', img)
            updatePositions(contours)

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                cv2.destroyAllWindows()
                break






# [(283, 537, 8, 3), (741, 536, 8, 4), (362, 520, 60, 20), (768, 479, 8, 21), (665, 429, 8, 20),
# (622, 405, 8, 21), (389, 337, 8, 20), (213, 334, 8, 21), (807, 252, 60, 19), (153, 241, 60, 21),
# (395, 226, 60, 21), (532, 225, 60, 21), (537, 134, 60, 20), (618, 119, 60, 21), (46, 37, 3, 3), (46, 31, 3, 3),
# (54, 28, 9, 13), (56, 30, 5, 9), (17, 28, 3, 2), (9, 28, 36, 13), (39, 36, 4, 3), (31, 36, 4, 3), (36, 33, 5, 5),
# (39, 32, 4, 3), (31, 32, 4, 3), (51, 17, 3, 3), (51, 11, 3, 3), (19, 11, 31, 10), (43, 16, 5, 3), (44, 12, 3, 3),
# (28, 12, 6, 7), (20, 12, 8, 7), (59, 8, 9, 13), (60, 10, 6, 9), (8, 7, 10, 14), (960, 0, 1, 540)]








if __name__ == '__main__':
    main()
