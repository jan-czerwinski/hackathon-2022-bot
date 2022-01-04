from time import sleep, time
from typing import Tuple, Any

import numpy as np
import cv2
from mss import mss
from PIL import Image
from pynput.keyboard import Key, Controller, Listener
import math
import copy


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

    # Alias the __matmul__ method to dot so  can use a @ b as well as a.dot(b).
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


BULLET_WIDTH = 8
PLAYER_WIDTH = 60


class GameObject:
    def __init__(self, position: Vector2D):
        self.direction = None
        self.position = position

    def updatePosition(self, new_pos: Vector2D) -> Vector2D:
        self.position = new_pos
        return self.position

    def setDirection(self, old_objects):
        old_obj = min(old_objects, key=lambda x: abs(
            self.position - x.getPosition()))
        old_pos = old_obj.getPosition()

        if old_pos.x != self.position.x:
            self.direction = 1 if old_pos.x < self.position.x else -1
        else:
            self.direction = old_obj.getDirection()

    def getId(self) -> int:
        return id(self)

    def getPosition(self) -> Vector2D:
        return self.position

    def getDirection(self) -> Vector2D:
        return self.direction

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
    bounding_box = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

    def __init__(self):
        self.keyboard = Controller()
        self.sct = mss()
        self.grabbedFrame = None
        self.detectedContours = None

        self.moving = None

        self.bullets = []
        self.enemies = []
        self.player = Player(Vector2D(0, 0))

    def ai(self):
        playerPos = self.player.getPosition()

        distances = [x.getPosition() - playerPos for x in self.bullets]


        #bullets directly above and close
        bullets_above = list(filter(lambda dist: abs(dist.x) < (PLAYER_WIDTH/2 +20) and abs(dist.y) < 100, distances))

        weight = 0
        for bullet_vec in bullets_above:
            weight += abs(bullet_vec) * \
                (1 if bullet_vec.x < 0 else -1)


        if weight > 0:
            self.moving = "right"
        elif weight == 0:
            self.moving = None
        else:
            self.moving = "left"

        print(self.moving)

        # print(bullets_above)






    def move_player(self):
        if self.moving is None:
            self.keyboard.release(Key.left)
            self.keyboard.release(Key.right)
            return

        if self.moving == 'left':
            self.keyboard.press(Key.left)
            self.keyboard.release(Key.right)

        elif self.moving == 'right':
            self.keyboard.press(Key.right)
            self.keyboard.release(Key.left)

    def shoot(self):
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
            cv2.putText(debug_img, f'enm {enemy.getDirection()}', enemy.getPositionTuple(), font,
                        font_scale, color, 1, cv2.LINE_AA)

        name = 'debug image'
        cv2.namedWindow(name)
        cv2.moveWindow(name, 400, 900)
        cv2.imshow(name, debug_img)

    def detect_contours(self):
        gray = cv2.cvtColor(self.grabbedFrame, cv2.COLOR_BGR2GRAY)

        ret, thresh1 = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # Find the contours
        contours, hierarchy = cv2.findContours(
            thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        self.detectedContours = [cv2.boundingRect(
            contour) for contour in contours]

    def updatePositions(self):

        # get big objects in x dir
        enemies_positions = [ob for ob in self.detectedContours if 50 < ob[2]]
        # get smol objects in x dir
        bullets_positions = [
            ob for ob in self.detectedContours if 6 < ob[2] < 15]

        enemies_positions_vec = [Vector2D(
            pos[0] + pos[2] / 2, pos[1] + pos[3] / 2) for pos in enemies_positions]
        bullets_positions_vec = [Vector2D(
            pos[0] + pos[2] / 2, pos[1] + pos[3] / 2) for pos in bullets_positions]
        if enemies_positions:
            player_idx, player_contour = max(
                enumerate(enemies_positions), key=lambda x: x[1][1])
            enemies_positions.pop(player_idx)
            self.player.updatePosition(
                Vector2D(player_contour[0], player_contour[1]))

        prev_enemies_obj = copy.deepcopy(self.enemies)
        self.enemies = [Enemy(Vector2D(
            cont[0] + cont[2] / 2, cont[1] + cont[3] / 2)) for cont in enemies_positions]
        self.bullets = [Bullet(Vector2D(
            cont[0] + cont[2] / 2, cont[1] + cont[3] / 2)) for cont in bullets_positions]

        if prev_enemies_obj:
            for enemy in self.enemies:
                enemy.setDirection(prev_enemies_obj)

    def getFrame(self):
        frame_raw = self.sct.grab(self.bounding_box)
        self.grabbedFrame = np.array(frame_raw)
        temp_frame = self.grabbedFrame.copy()

        gray = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2GRAY)

        ret, thresh1 = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        thresh1 = (255 - thresh1)
        contours, hierarchy = cv2.findContours(
            thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        c = max(contours, key=cv2.contourArea)
        cnt = cv2.boundingRect(c)
        (x, y, w, h) = cnt
        self.grabbedFrame = self.grabbedFrame[y:h + y, x:w + x]
        cv2.rectangle(self.grabbedFrame, (0, 0), (80, 40), (0, 0, 0), -1)

    def willBulletHitEnemy(self):
        for enemy in self.enemies:
            if enemy.getDirection() is not None:

                enemySpeed  = 3
                bulletSpeed = 8

                enemy_vec = self.player.getPosition() - enemy.getPosition()

                timeBulletHit = enemy_vec.y / bulletSpeed

                enemyNewX = enemy_vec.x + enemy.getDirection() * enemySpeed * timeBulletHit

                screenWidth = self.bounding_box['width']
                if enemyNewX > screenWidth-PLAYER_WIDTH/2:
                    enemyNewX - screenWidth
                enemyNewX = abs(enemyNewX)


                if enemyNewX-PLAYER_WIDTH/2 < self.player.getPosition().x < enemyNewX+PLAYER_WIDTH/2:
                    self.shoot()

    def main(self):
        self.getFrame()

        FPS = 50
        while True:
            sleep((FPS - time() % FPS) / 1000)
            self.getFrame()
            self.detect_contours()
            self.updatePositions()
            self.showDebugImage()

            self.ai()
            self.move_player()
            self.willBulletHitEnemy()

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    game = Game()
    game.main()

'''
OG game has 50 fps
bullet moves 8 px/frame
player also movees 8px/frame
enemies move randommly (all 2px or 3px) /frame=
'''
