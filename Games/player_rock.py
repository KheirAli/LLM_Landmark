from .entity import Entity
from .actions import GetoutActions
from .entityEncoding import EntityID
from .resource_loader import flip_horizontal
from .actions import coin_jump_actions_from_unified


class Player(Entity):

    def __init__(self, level, x, y, resource_loader=None):
        super().__init__(level, EntityID.PLAYER, x, y)
        self._noop_action = [GetoutActions.NOOP]
        self.action = self._noop_action

        self.move_speed = 0.2
        self.jump_strength = 0.95

        self.facing_left = False

        self.checks_collision = True

        self.is_powered_up = False
        self.coin_mine = 0
        self.collision_x = -1  # x position of coin collisions
        self.collisions = [False, False, False, False]  # key=0, door=1, enemy=2, coin = 3

        self._resource_loader = resource_loader
        self._load_sprites(resource_loader)

    def _load_sprites(self, resource_loader, color="Green"):
        if resource_loader is None:
            self.sprites = None
            return

        sprites = {}
        player_crop = (0, 100, 128, 255)

        player_path = f'Players/128x256/{color}/alien{color}_'

        sprites["standing_right"] = [
            resource_loader.get_sprite(f'player_{color}_standing1_r', player_path + 'stand.png', player_crop)
        ]
        sprites["standing_left"] = [
            resource_loader.get_sprite(f'player_{color}_standing1_l', player_path + 'stand.png', player_crop,
                                       flip_horizontal)
        ]
        sprites["walking_right"] = [
            resource_loader.get_sprite(f'player_{color}_walking1_r', player_path + 'walk1.png', player_crop),
            resource_loader.get_sprite(f'player_{color}_walking2_r', player_path + 'walk2.png', player_crop)
        ]
        sprites["walking_left"] = [
            resource_loader.get_sprite(f'player_{color}_walking1_l', player_path + 'walk1.png', player_crop,
                                       flip_horizontal),
            resource_loader.get_sprite(f'player_{color}_walking2_l', player_path + 'walk2.png', player_crop,
                                       flip_horizontal)
        ]
        sprites["jumping_right"] = [
            resource_loader.get_sprite(f'player_{color}_jumping1_r', player_path + 'jump.png', player_crop)
        ]
        sprites["jumping_left"] = [
            resource_loader.get_sprite(f'player_{color}_jumping1_l', player_path + 'jump.png', player_crop,
                                       flip_horizontal)
        ]

        self.sprites = sprites

    def set_action(self, action):
        if type(action) is int:
            self.action = coin_jump_actions_from_unified(action)
        else:
            self.action = action

    def step(self):
        self.ax = 0
        self.ay = 0
        self.use_support = True
        # change here to choose  mode
        # Fixme: change here to choose  mode
        # print('action', GetoutActions.MOVE_LEFT, self.action, self.action, GetoutActions.MOVE_RIGHT)
        if len(self.action) == 1:
            self.action = self.action[0]
        if 'left'in self.action:
            self.ax -= self.move_speed
        if 'right' in self.action:
            # print('right')
            self.ax += self.move_speed
        if 'jump' in self.action:
            if self.floating or self.flying:
                self.ay += self.move_speed
            else:
                self.ay += self.jump_strength
        # else:
            # if GetoutActions.MOVE_DOWN in self.action:
            #     self.use_support = False
            #     if self.floating or self.flying:
            #         self.ay -= self.move_speed
        # print('action', GetoutActions.MOVE_LEFT, self.action, self.action, GetoutActions.MOVE_RIGHT)
        # if GetoutActions.MOVE_LEFT in self.action:
        #     self.ax -= self.move_speed
        # if GetoutActions.MOVE_RIGHT in self.action:
        #     self.ax += self.move_speed
        # if GetoutActions.MOVE_UP in self.action:
        #     if self.floating or self.flying:
        #         self.ay += self.move_speed
        #     else:
        #         self.ay += self.jump_strength
        # else:
        #     if GetoutActions.MOVE_DOWN in self.action:
        #         self.use_support = False
        #         if self.floating or self.flying:
        #             self.ay -= self.move_speed

        # our facing is determined by the direction we are 'trying' to walk
        # not the direction we are actually moving.
        if self.ax != 0:
            self.facing_left = self.ax < 0

        super(Player, self).step()

        # return "standing_"+"left"
        self.action = self._noop_action

    def _check_collisions(self):
        self.collisions = [False, False, False, False]  # key=0, door=1, enemy=2
        for entity in self.level.entities:
            if entity == self:
                continue
            if self.bounding_box.intersects(entity.bounding_box):
                self._handle_entity_collision(entity)

    def _handle_entity_collision(self, entity):
        if entity.is_enemy:
            self.collisions[2] = True
            # we jump 'on' an enemy if we are moving down (good old mario logic...)
            if self.is_powered_up or (self.vy < 0 and entity.can_be_killed_by_jump):
                # jumped on enemy:
                self.level.entities.remove(entity)
                self.level.take_reward(self.level.reward_values['enemy']) #reward_values_coin replaced
            # else: 
                # touched enemy -> end game
                # self.level.terminate(lost=True)
                

        if entity._entity_id == EntityID.KEY:
            self.collisions[0] = True
            self.level.entities.remove(entity)
            self.level.add_key(1)
            self.level.take_reward(self.level.reward_values['key'])

        if entity._entity_id == EntityID.DOOR:
            self.collisions[1] = True
            if self.level.get_key() > 0:
                if self.level.get_coins()>1:
                    if self.level.get_flag()>0:
                        self.level.take_reward(self.level.reward_values['door'])
                self.level.terminate(lost=False)

        if entity._entity_id == EntityID.COIN:
            self.collision_x = entity.x
            self.collisions[3] = True
            self.coin_mine += 1
            if self.coin_mine > 0.9:
                self.level.entities.remove(entity)
                # self.level.terminate(lost=False)
                self.level.add_coins(1)
            self.level.take_reward(self.level.reward_values['coin'])

        if entity._entity_id == EntityID.COIN2:
            self.collision_x = entity.x
            self.collisions[3] = True
            self.coin_mine += 1
            if self.coin_mine > 0.9:
                self.level.entities.remove(entity)
                # self.level.terminate(lost=False)
                self.level.add_coins(1)
            self.level.take_reward(self.level.reward_values['coin'])

        if entity._entity_id == EntityID.FLAG:
            self.collision_x = entity.x
            self.collisions[3] = True
            self.level.entities.remove(entity)
            self.level.add_flag(1)

            # self.coin_mine += 1
            # if self.coin_mine > 0.9:
            # self.level.entities.remove(entity)
                # self.level.terminate(lost=False)
                # self.level.add_coins(1)
            # self.level.take_reward(self.level.reward_values['key'])

        if entity._entity_id == EntityID.KEY2:
            self.collision_x = entity.x
            self.collisions[0] = True
            self.level.entities.remove(entity)
            self.level.take_reward(self.level.reward_values['coin'])
            # self.level.add_key(1)
            # self.level.take_reward(self.level.reward_values['key'])

    def _get_parameterization(self):
        return [self.is_powered_up, 0, 0, 0]

    def render(self, camera, frame):
        sprite_sequence = self.sprites[self._get_state_string()]
        seq_id = frame // 2 % len(sprite_sequence)
        camera.paint_sprite(self.x - self.size[0] / 2, self.y, self.size, sprite_sequence[seq_id])
        # camera.paint_sprite(self.x - 0.5, self.y, self.size, self.sprite_sequence[seq_id])

    def _get_state_string(self):
        if self.vy == 0:
            if self.vx == 0:
                return f"standing_{'left' if self.facing_left else 'right'}"
            else:
                return f"walking_{'left' if self.facing_left else 'right'}"
        else:
            return f"jumping_{'left' if self.facing_left else 'right'}"
