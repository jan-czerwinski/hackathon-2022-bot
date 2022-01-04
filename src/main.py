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
        self.direction = 0
        self.position = position

    def updatePosition(self, new_pos: Vector2D) -> Vector2D:
        self.position = new_pos
        return self.position

    def getId(self) -> int:
        return id(self)

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

    def setDirection(self, old_objects):
        old_obj = min(old_objects, key=lambda x: abs(
            self.position - x.getPosition()))
        old_pos = old_obj.getPosition()

        if old_pos.x != self.position.x:
            self.direction = 1 if old_pos.x < self.position.x else -1
        else:
            self.direction = old_obj.getDirection()


class Bullet(GameObject):
    def __init__(self, position: Vector2D):
        super().__init__(position)

    def setDirection(self, old_objects):
        old_obj = min(old_objects, key=lambda x: abs(
            self.position - x.getPosition()))
        old_pos = old_obj.getPosition()

        if old_pos.y != self.position.y:
            self.direction = 1 if old_pos.y < self.position.y else -1
        else:
            self.direction = old_obj.getDirection()


class Player(GameObject):
    keyboard = None
    def __init__(self, position: Vector2D):
        super().__init__(position)
        self.keyboard = Controller()

        self.last_shot_time = time()
        self.moving = None
        
    def setMoving(self, moving):
        self.moving = moving


    def shoot(self):
        if time() - self.last_shot_time > 1.1:
            self.last_shot_time = time()
            self.keyboard.press(Key.space)
            self.keyboard.release(Key.space)
            
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
        
        
    # def checkShootAbility(self):
    # if time() - self.last_shoot_time > 1:
    # self.can_shoot = True


class Game:
    sct = None
    bounding_box = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

    def __init__(self):
        self.sct = mss()
        self.grabbedFrame = None
        self.detectedContours = None
        self.gameDimensions = (0, 0)
        self.startTime = time()


        self.bullets = []
        self.enemies = []
        self.player = Player(Vector2D(0, 0))

    def findUnavoidableBullets(self):
        player_pos = self.player.getPosition()
        bullets_horizontal = list(
            filter(lambda dist: abs(dist.getPosition().x) < (PLAYER_WIDTH * 2.5) and abs(dist.getPosition().y) < 70,
                   self.bullets))
        if bullets_horizontal:
            print(bullets_horizontal)

    def playerMovementAi(self):
        player_pos = self.player.getPosition()
        # bullets directly above and close

        bullets_around = [bul for bul in self.bullets if abs(bul.getPosition() - player_pos) < 200]

        distances = [x.getPosition() - player_pos for x in self.bullets]
        bullets_above = list(
            filter(lambda dist: abs(dist.x) < (PLAYER_WIDTH * 1.2) and abs(dist.y) < 200, distances))

        weight = 0
        bullets_around_poss = [bul.getPosition() for bul in bullets_around]

        # print(bullets_above_poss)
        # for bullet_vec in bullets_around_poss:
        #     weight += abs(bullet_vec - player_pos) ** 2

        # if bullets_above_poss:
        #     weight /= len(bullets_above_poss)

        for bullet_vec in bullets_above:
            weight += 1/abs(bullet_vec) * \
                      (1 if bullet_vec.x < 0 else -1)
        weight *= 1400

        distance_from_center = self.gameDimensions[0] / 2 - player_pos.x


        center_sign = 1 if distance_from_center >= 0 else -1
        weight += abs(distance_from_center / 800) ** 2 * center_sign

        # print(weight)

        weight = weight
        # print(weight)

        # - left + right
        self.findUnavoidableBullets()
        if weight > 0:
            self.player.setMoving("right")
        elif weight == 0:
            self.player.setMoving("None")
        else:
            self.player.setMoving("left")

        # print(bullets_above)





    def showDebugImage(self, enemyX):
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
            cv2.putText(debug_img, f'bul {bullet.getDirection()}', bullet.getPositionTuple(), font,
                        font_scale, color, 1, cv2.LINE_AA)

        for enemy in self.enemies:
            cv2.putText(debug_img, f'enm', enemy.getPositionTuple(), font,
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
        bullets_positions = [ob for ob in self.detectedContours if 6 < ob[2] < 15]

        if enemies_positions:
            player_idx, player_contour = max(
                enumerate(enemies_positions), key=lambda x: x[1][1])
            enemies_positions.pop(player_idx)
            self.player.updatePosition(
                Vector2D(player_contour[0] + player_contour[2] / 2, player_contour[1] + player_contour[3] / 2))

        prev_enemies_obj = copy.deepcopy(self.enemies)
        prev_bullets_obj = copy.deepcopy(self.bullets)
        self.enemies = [Enemy(Vector2D(
            cont[0] + cont[2] / 2, cont[1] + cont[3] / 2)) for cont in enemies_positions]
        self.bullets = [Bullet(Vector2D(
            cont[0] + cont[2] / 2, cont[1] + cont[3] / 2)) for cont in bullets_positions]

        if prev_enemies_obj:
            for enemy in self.enemies:
                enemy.setDirection(prev_enemies_obj)

        if prev_bullets_obj:
            for bullet in self.bullets:
                bullet.setDirection(prev_bullets_obj)

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
        return w, h

    def willBulletHitEnemy(self):
        for enemy in self.enemies:
            if enemy.getDirection() is not None:

                enemySpeed = 2
                bulletSpeed = 8

                enemy_vec = self.player.getPosition() - enemy.getPosition()

                timeBulletHit = abs(enemy_vec.y / bulletSpeed)
                # print(timeBulletHit)

                enemyNewX = enemy.getPosition().x + enemy.getDirection() * enemySpeed * timeBulletHit

                if enemyNewX < PLAYER_WIDTH / 2 or enemyNewX > self.gameDimensions[0] - (PLAYER_WIDTH / 2):
                    return 0

                if enemyNewX - PLAYER_WIDTH / 4 < self.player.getPosition().x < enemyNewX + PLAYER_WIDTH / 4:
                    self.player.shoot()
                    print(enemyNewX)
                    return enemyNewX
        return 0 

    def main(self):
        self.gameDimensions = self.getFrame()

        FPS = 50
        # time_sleep = (FPS - time() % FPS) / 1000
        while True:
            # start_time = time()

            self.getFrame()
            self.detect_contours()
            self.updatePositions()
            self.willBulletHitEnemy()
            self.showDebugImage()

            
            self.playerMovementAi()
            self.player.move()
            self.willBulletHitEnemy()

            # will_sleep = 0 if time_sleep - (time() - start_time) < 0 else time_sleep - (time() - start_time)
            # sleep(will_sleep)
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
enemies move randommly (2px or 3px) /frame=
'''
