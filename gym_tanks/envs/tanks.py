
import numpy as np
import os, pygame, time, random, uuid, sys
import matplotlib.pyplot as plt
import multiprocessing
import queue
from queue import Empty


import heapq
import math


import gymnasium

from skimage.transform import rescale
from collections import deque


'''
===============================================================================================================================
															UTILITIES
===============================================================================================================================
'''

#图像灰度化
def rgb_to_grayscale(rgb_array):
	
	weights = np.array([0.2989, 0.5870, 0.1140])
	
	grayscale_array = np.dot(rgb_array[...,:3], weights)
	
	grayscale_array_noinfobar = grayscale_array[:, :416]
	grayscale_array_downscaled = rescale(grayscale_array_noinfobar, 1/2.0, anti_aliasing=True, mode='reflect')
	grayscale_array_rounded = np.round(grayscale_array_downscaled).astype(np.uint8)
	
	return grayscale_array_rounded

#计算两点间的曼哈顿距离
def Vmanhattan_distance(a, b):
	x1, y1 = a
	x2, y2 = b
	return abs(x1 - x2) + abs(y1 - y2)

#欧几里得距离
def Veuclidean_distance(a, b):
	x1, y1 = a
	x2, y2 = b
	return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

#判断玩家与敌人是否位于同一直线
def Vinline_with_enemy(player_rect, enemy_rect):
	
	if enemy_rect.left <= player_rect.centerx <= enemy_rect.right and abs(player_rect.top - enemy_rect.bottom) <= 151:
		
		if enemy_rect.bottom <= player_rect.top:
			
			return 0
		
		elif player_rect.bottom <= enemy_rect.top:
			
			return 2
	
	if enemy_rect.top <= player_rect.centery <= enemy_rect.bottom and abs(player_rect.left - enemy_rect.right) <= 151:
		
		if enemy_rect.right <= player_rect.left:
			
			return 3
		
		elif player_rect.right <= enemy_rect.left:
			
			return 1
	return 4

#根据子弹位置预测最佳躲避方向
def Vbullet_avoidance(player_rect_out, bullet_info_list): 
	obs_bullet_avoidance_direction = 4

	
	player_rect = player_rect_out

	
	sorted_bullet_info_list = sorted(bullet_info_list, key=lambda x: Veuclidean_distance((x[0].left, x[0].top), (player_rect.centerx, player_rect.centery)))

	
	if sorted_bullet_info_list:
		min_dist_with_bullet = Veuclidean_distance((sorted_bullet_info_list[0][0].left, sorted_bullet_info_list[0][0].top), (player_rect.centerx, player_rect.centery))
	else:
		min_dist_with_bullet = float(1e30000)

	
	if min_dist_with_bullet <= 120:
		
		bullet_rect = sorted_bullet_info_list[0][0]
		bullet_direction = sorted_bullet_info_list[0][1]
		
		if abs(bullet_rect.centerx - player_rect.centerx) <= 25:
			
			if abs(bullet_rect.centerx - player_rect.centerx) <= 5:
				
				if bullet_direction == 0 and bullet_rect.top > player_rect.top:
					
					obs_bullet_avoidance_direction = 2 
					
				
				if bullet_direction == 2 and bullet_rect.top < player_rect.top:
					
					obs_bullet_avoidance_direction = 0 
					
			
		
		elif abs(bullet_rect.centery - player_rect.centery) <= 25:
			
			if abs(bullet_rect.centery - player_rect.centery) <= 5:
				
				if bullet_direction == 1 and bullet_rect.left < player_rect.left:
					
					obs_bullet_avoidance_direction = 3 
					
				
				if bullet_direction == 3 and bullet_rect.left > player_rect.left:
					
					obs_bullet_avoidance_direction = 1 
					
		
	return obs_bullet_avoidance_direction

#阻止自杀
def antiStupidBlock(player_direction, player_rect, base_rect):
	
	if base_rect.left <= player_rect.centerx <= base_rect.right:
		
		if base_rect.bottom <= player_rect.top and player_direction == 0:
			
			return 1
		
		elif player_rect.bottom <= base_rect.top and player_direction == 2:
			
			return 1
	
	if base_rect.top <= player_rect.centery <= base_rect.bottom:
		
		if base_rect.right <= player_rect.left and player_direction == 3:
			
			return 1
		
		elif player_rect.right <= base_rect.left and player_direction == 1:
			
			return 1
	return 0

'''
===============================================================================================================================
															AI BOT CODE
===============================================================================================================================
'''
#管理队列，包括初始化、检查是否空值、添加与获取元素等等有关队列的操作
class PriorityQueue:
	def __init__(self):
		self.elements = []

	def empty(self):
		return len(self.elements) == 0

	def put(self, item, priority):
		heapq.heappush(self.elements, (priority, item))

	def get(self):
		return heapq.heappop(self.elements)[1]
class ai_agent():
	mapinfo = []
	#预设基地位置
	castle_rect = pygame.Rect(12 * 16, 24 * 16, 32, 32)
	#初始化AI智能体
	def __init__(self):
		self.mapinfo = []#储存地图信息
	#通过循环来决定坦克的行为策略
	def operations(self, p_mapinfo, c_control):
		
		while True:
			self.Get_mapInfo(p_mapinfo)  
			player_rect = self.mapinfo[3][0][0]  
            
			
			sorted_enemy_with_distance_to_castle = sorted(
                self.mapinfo[1],
                key=lambda x: self.manhattan_distance(x[0].center, self.castle_rect.center))
            
			sorted_enemy_with_distance_to_player = sorted(
                self.mapinfo[1],
                key=lambda x: self.manhattan_distance(x[0].center, player_rect.center))

            
			default_pos_rect = pygame.Rect(195, 3, 26, 26)
			if sorted_enemy_with_distance_to_castle:
                
				if self.manhattan_distance(sorted_enemy_with_distance_to_castle[0][0].topleft, self.castle_rect.topleft) < 150:
					enemy_rect = sorted_enemy_with_distance_to_castle[0][0]
					enemy_direction = sorted_enemy_with_distance_to_castle[0][1]
				else:  
					enemy_rect = sorted_enemy_with_distance_to_player[0][0]
					enemy_direction = sorted_enemy_with_distance_to_player[0][1]

                
				inline_direction = self.inline_with_enemy(player_rect, enemy_rect)
                
				astar_direction = self.a_star(player_rect, enemy_rect, 6)
                
				shoot, direction = self.bullet_avoidance(self.mapinfo[3][0], 6, self.mapinfo[0], astar_direction, inline_direction)
                
				self.Update_Strategy(c_control, shoot, direction)
				time.sleep(0.005)
			else:
                
				astar_direction = self.a_star(player_rect, default_pos_rect, 6)
				if astar_direction is not None:
					self.Update_Strategy(c_control, 0, astar_direction)
				else:
					self.Update_Strategy(c_control, 0, 0)
	#从队列中获取地图信息
	def Get_mapInfo(self, p_mapinfo):
		if p_mapinfo.empty() != True:
			try:
				self.mapinfo = p_mapinfo.get(False)
			except queue.empty:
				skip_this = True
	#更新策略，包括是否射击、移动方向等等	
	def Update_Strategy(self, c_control, shoot, move_dir):
		
		if c_control.empty() == True:
			c_control.put([shoot, move_dir])
	#通过敌我坦克的位置、大小，判断是否应该开火
	def should_fire(self, player_rect, enemy_rect_info_list):
		for enemy_rect_info in enemy_rect_info_list:
			if self.inline_with_enemy(player_rect, enemy_rect_info[0]) is not False:
				return True
	#使用优先队列和启发式算法寻找最优路径来实现寻路
	def a_star(self, start_rect, goal_rect, speed):
		
		start = (start_rect.left, start_rect.top)
		goal = (goal_rect.left, goal_rect.top)

		
		frontier = PriorityQueue()
		came_from = {} 
		cost_so_far = {} 

		frontier.put(start, 0) 
		came_from[start] = None
		cost_so_far[start] = 0

		while not frontier.empty():
			current_left, current_top = frontier.get()
			current = (current_left, current_top)

			
			temp_rect = pygame.Rect(current_left, current_top, 26, 26)
			if self.is_goal(temp_rect, goal_rect):
				break

			 
			for next in self.find_neighbour(current_top, current_left, speed, goal_rect):
				
				new_cost = cost_so_far[current] + speed
				if next not in cost_so_far or new_cost < cost_so_far[next]:
					cost_so_far[next] = new_cost
					priority = new_cost + self.heuristic(goal, next)
					frontier.put(next, priority)
					came_from[next] = current

		
		next = None
		dir_cmd = None
		while current != start:
			next = current
			current = came_from[current]

		if next:
			next_left, next_top = next
			current_left, current_top = current
			if current_top > next_top:
				dir_cmd = 0 
			elif current_top < next_top:
				dir_cmd = 2 
			elif current_left > next_left:
				dir_cmd = 3 
			elif current_left < next_left:
				dir_cmd = 1 
		return dir_cmd
	#曼哈顿距离
	def manhattan_distance(self, a, b):
		x1, y1 = a
		x2, y2 = b
		return abs(x1 - x2) + abs(y1 - y2)
	#欧几里得距离
	def euclidean_distance(self, a, b):
		x1, y1 = a
		x2, y2 = b
		return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
	#上述a*star函数的启发式函数	
	def heuristic(self, a, b):
		return self.manhattan_distance(a, b)
	#判断是否到达目标
	def is_goal(self, rect1, rect2):
		center_x1, center_y1 = rect1.center
		center_x2, center_y2 = rect2.center
		return abs(center_x1 - center_x2) <= 7 and abs(center_y1 - center_y2) <= 7
	#寻找可以行走的相邻位置
	def find_neighbour(self, top, left, speed, goal_rect):
		global obs_flag_top_occupied, obs_flag_bottom_occupied, obs_flag_left_occupied, obs_flag_right_occupied
		obs_flag_top_occupied = False
		obs_flag_bottom_occupied = False
		obs_flag_left_occupied = False
		obs_flag_right_occupied = False

		
		allowable_move = []

		
		new_top = top - speed
		new_left = left
		if not (new_top < 0): 
			move_up = True
			temp_rect = pygame.Rect(new_left, new_top, 26, 26)

			
			for enemy in self.mapinfo[1]:
				if enemy[0] is not goal_rect:
					if temp_rect.colliderect(enemy[0]):
						move_up = False
						break

			
			if move_up:
				for tile in self.mapinfo[2]:
					
					if tile[1] != 4:
						if temp_rect.colliderect(tile[0]):
							move_up = False
							break

			if move_up:
				allowable_move.append((new_left, new_top))

		
		new_top = top
		new_left = left + speed
		if not (new_left > (416 - 26)): 
			move_right = True
			temp_rect = pygame.Rect(new_left, new_top, 26, 26)

			
			for enemy in self.mapinfo[1]:
				if enemy[0] is not goal_rect:
					if temp_rect.colliderect(enemy[0]):
						move_right = False
						break

			
			if move_right:
				for tile in self.mapinfo[2]:
					if tile[1] != 4:
						if temp_rect.colliderect(tile[0]):
							move_right = False
							break

			if move_right:
				allowable_move.append((new_left, new_top))

		
		new_top = top + speed
		new_left = left
		if not (new_top > (416 - 26)):
			move_down = True
			temp_rect = pygame.Rect(new_left, new_top, 26, 26)

			
			for enemy in self.mapinfo[1]:
				if enemy[0] is not goal_rect:
					if temp_rect.colliderect(enemy[0]):
						move_down = False
						break

			
			if move_down:
				for tile in self.mapinfo[2]:
					if tile[1] != 4:
						if temp_rect.colliderect(tile[0]):
							move_down = False
							break
			if move_down:
				allowable_move.append((new_left, new_top))

		
		new_top = top
		new_left = left - speed
		if not (new_left < 0): 
			move_left = True
			temp_rect = pygame.Rect(new_left, new_top, 26, 26)

			
			for enemy in self.mapinfo[1]:
				if enemy[0] is not goal_rect:
					if temp_rect.colliderect(enemy[0]):
						move_left = False
						break
			
			
			if move_left:
				for tile in self.mapinfo[2]:
					if tile[1] != 4:
						if temp_rect.colliderect(tile[0]):
							move_left = False
							break

			if move_left:
				allowable_move.append((new_left, new_top))

		return allowable_move
	#判断敌我坦克在同一行或者同一列
	def inline_with_enemy(self, player_rect, enemy_rect):
		
		if enemy_rect.left <= player_rect.centerx <= enemy_rect.right and abs(player_rect.top - enemy_rect.bottom) <= 151:
			
			if enemy_rect.bottom <= player_rect.top:
				
				return 0
			
			elif player_rect.bottom <= enemy_rect.top:
				
				return 2
		
		if enemy_rect.top <= player_rect.centery <= enemy_rect.bottom and abs(player_rect.left - enemy_rect.right) <= 151:
			
			if enemy_rect.right <= player_rect.left:
				
				return 3
			
			elif player_rect.right <= enemy_rect.left:
				
				return 1
		
		return False
	#通过子弹信息，坦克移速，a*star算法，判断子弹是否会击中，返回下一步移动的方向
	def bullet_avoidance(self, player_info, speed, bullet_info_list, direction_from_astar, inlined_with_enemy):
 		
		directions = []

        
		player_rect = player_info[0]

        
		sorted_bullet_info_list = sorted(bullet_info_list, key=lambda x: self.euclidean_distance((x[0].left, x[0].top), (player_rect.centerx, player_rect.centery)))

        
		shoot = 0
		min_dist_with_bullet = float(1e30000)  

        
		if sorted_bullet_info_list:
            
			min_dist_with_bullet = self.euclidean_distance((sorted_bullet_info_list[0][0].left, sorted_bullet_info_list[0][0].top), (player_rect.centerx, player_rect.centery))

        
		if min_dist_with_bullet <= 120:
            
			bullet_rect = sorted_bullet_info_list[0][0]
			bullet_direction = sorted_bullet_info_list[0][1]

            
			if abs(bullet_rect.centerx - player_rect.centerx) <= 30:
                
				if abs(bullet_rect.centerx - player_rect.centerx) <= 5:
                    
					if bullet_direction == 0 and bullet_rect.top > player_rect.top:
						directions.append(2)
						shoot = 1
						
                    
					if bullet_direction == 2 and bullet_rect.top < player_rect.top:
						directions.append(0)
						shoot = 1
						
                
				else:
                    
					if bullet_rect.left > player_rect.centerx:
						directions.append(3)
						
                    
					else:
						directions.append(1)
						

            
			elif abs(bullet_rect.centery - player_rect.centery) <= 30:
                
				if abs(bullet_rect.centery - player_rect.centery) <= 5:
					
					if bullet_direction == 1 and bullet_rect.left < player_rect.left:
						directions.append(3)
						shoot = 1
						
                    
					if bullet_direction == 3 and bullet_rect.left > player_rect.left:
						directions.append(1)
						shoot = 1
						
                
				else:
                    
					if bullet_rect.top > player_rect.centery:
						directions.append(0)
						directions.append(2)
						
					else:
						directions.append(2)
						directions.append(0)
						

            
			else:
				if inlined_with_enemy == direction_from_astar:
					shoot = 1
				directions.append(direction_from_astar)

                
				if bullet_direction == 0 or bullet_direction == 2:  
					if bullet_rect.left > player_rect.left:
						if 1 in directions:
							directions.remove(1)
							
						else:
							if 3 in directions:
								directions.remove(3)
							
					
				if bullet_direction == 1 or bullet_direction == 3:  
					if bullet_rect.top > player_rect.top:
						if 2 in directions:
							directions.remove(2)
						
					else:
						if 0 in directions:
							directions.remove(0)
						

        
		else:
            
			if inlined_with_enemy == direction_from_astar:
				shoot = 1
			directions.append(direction_from_astar)
		
		if directions:
			for direction in directions:
				new_left, new_top = self.calculate_new_position(player_rect, direction, speed)
				temp_rect = pygame.Rect(new_left, new_top, 26, 26)
				
				if self.is_valid_position(new_top, new_left):
					if not self.is_collision(temp_rect):
						if not self.will_hit_base_or_obstacles(player_rect, direction):
							return shoot, direction
				else:
					opposite_direction = self.get_opposite_direction(direction)
					new_left, new_top = self.calculate_new_position(player_rect, opposite_direction, speed)
					temp_rect = pygame.Rect(new_left, new_top, 26, 26)
					if self.is_valid_position(new_top, new_left) and not self.is_collision(temp_rect):
						if not self.will_hit_base_or_obstacles(player_rect, opposite_direction):
							return shoot, opposite_direction

        
		else:
			return shoot, 4

        
		return shoot, direction_from_astar
	#计算移动的新位置
	def calculate_new_position(self, player_rect, direction, speed):
		
		if direction == 0:  
			new_left = player_rect.left
			new_top = player_rect.top - speed
		elif direction == 1:  
			new_left = player_rect.left + speed
			new_top = player_rect.top
		elif direction == 2:  
			new_left = player_rect.left
			new_top = player_rect.top + speed
		elif direction == 3:  
			new_left = player_rect.left - speed
			new_top = player_rect.top
		else:  
			new_top = player_rect.top
			new_left = player_rect.left
		return new_left, new_top
	#判断位置是否有效
	def is_valid_position(self, top, left):
		
		return 0 <= top <= 416 - 26 and 0 <= left <= 416 - 26
	#判断位置是否会碰撞
	def is_collision(self, temp_rect):
		
		for tile_info in self.mapinfo[2]:
			tile_rect = tile_info[0]
			tile_type = tile_info[1]
			if tile_type != 4:  
				if temp_rect.colliderect(tile_rect):
					return True
		return False
	#获取方向的相反方向
	def get_opposite_direction(self, direction):
		
		return (direction + 2) % 4
	#判断子弹是否会击中基地或者障碍物
	def will_hit_base_or_obstacles(self, player_rect, direction):
		
		bullet_path = self.simulate_bullet_path(player_rect, direction)
		for obstacle in self.mapinfo[2]:  
			if bullet_path.colliderect(obstacle[0]):
				return True  
		if bullet_path.colliderect(self.castle_rect):
			return True  
		return False
	#模拟子弹运动路径
	def simulate_bullet_path(self, player_rect, direction):
		
		if direction in [0, 2]:  
			return pygame.Rect(player_rect.centerx - 2, 0, 4, 416)
		if direction in [1, 3]:  
			return pygame.Rect(0, player_rect.centery - 2, 416, 4)
		return pygame.Rect(0, 0, 0, 0)  





'''
===============================================================================================================================
													MAIN TANK BATTALION GAME
===============================================================================================================================
'''


#标识矩形块类型
class myRect(pygame.Rect):
	""" Add type property """
	def __init__(self, left, top, width, height, type):
		pygame.Rect.__init__(self, left, top, width, height)
		self.type = type
#添加、销毁、更新计时器，实现动画效果
class Timer(object):
	def __init__(self):
		self.timers = []

	def add(self, interval, f, repeat = -1):
		options = {
			"interval"	: interval,
			"callback"	: f,
			"repeat"		: repeat,
			"times"			: 0,
			"time"			: 0,
			"uuid"			: uuid.uuid4()
		}
  
		self.timers.append(options)

		return options["uuid"]

	def destroy(self, uuid_nr):
		for timer in self.timers:
			if timer["uuid"] == uuid_nr:
				self.timers.remove(timer)
				return

	def update(self, time_passed):
		for timer in self.timers:
			timer["time"] += time_passed
			if timer["time"] > timer["interval"]:
				timer["time"] -= timer["interval"]
				timer["times"] += 1
				if timer["repeat"] > -1 and timer["times"] == timer["repeat"]:
					self.timers.remove(timer)
				try:
					timer["callback"]()
				except:
					try:
						self.timers.remove(timer)
					except:
						pass
#城堡状态，包括绘制与重建逻辑，若其被摧毁则游戏结束
class Castle():
	""" Player's castle/fortress """

	(STATE_STANDING, STATE_DESTROYED, STATE_EXPLODING) = range(3)

	def __init__(self):

		global sprites

		
		self.img_undamaged = sprites.subsurface(0, 15*2, 16*2, 16*2)
		self.img_destroyed = sprites.subsurface(16*2, 15*2, 16*2, 16*2)

		
		self.rect = pygame.Rect(12*16, 24*16, 32, 32)

		
		self.rebuild()

	def draw(self):
		""" Draw castle """
		global screen

		screen.blit(self.image, self.rect.topleft)

		if self.state == self.STATE_EXPLODING:
			if not self.explosion.active:
				self.state = self.STATE_DESTROYED
				del self.explosion
			else:
				self.explosion.draw()

	def rebuild(self):
		""" Reset castle """
		self.state = self.STATE_STANDING
		self.image = self.img_undamaged
		self.active = True

	def destroy(self):
		""" Destroy castle """
		self.state = self.STATE_EXPLODING
		self.explosion = Explosion(self.rect.topleft)
		self.image = self.img_destroyed
		self.active = False
#管理游戏道具，包括手榴弹、头盔、铲子、星星等等，设计随机生成与效果
class Bonus():
	""" Various power-ups
	When bonus is spawned, it begins flashing and after some time dissapears

	Available bonusses:
		grenade	: Picking up the grenade power up instantly wipes out ever enemy presently on the screen, including Armor Tanks regardless of how many times you've hit them. You do not, however, get credit for destroying them during the end-stage bonus points.
		helmet	: The helmet power up grants you a temporary force field that makes you invulnerable to enemy shots, just like the one you begin every stage with.
		shovel	: The shovel power up turns the walls around your fortress from brick to stone. This makes it impossible for the enemy to penetrate the wall and destroy your fortress, ending the game prematurely. The effect, however, is only temporary, and will wear off eventually.
		star		: The star power up grants your tank with new offensive power each time you pick one up, up to three times. The first star allows you to fire your bullets as fast as the power tanks can. The second star allows you to fire up to two bullets on the screen at one time. And the third star allows your bullets to destroy the otherwise unbreakable steel walls. You carry this power with you to each new stage until you lose a life.
		tank		: The tank power up grants you one extra life. The only other way to get an extra life is to score 20000 points.
		timer		: The timer power up temporarily freezes time, allowing you to harmlessly approach every tank and destroy them until the time freeze wears off.
	"""

	
	(BONUS_GRENADE, BONUS_HELMET, BONUS_SHOVEL, BONUS_STAR, BONUS_TANK, BONUS_TIMER) = range(6)

	def __init__(self, level):

		global sprites

		
		self.level = level

		
		self.active = True

		
		self.visible = True

		self.rect = pygame.Rect(random.randint(0, 416-32), random.randint(0, 416-32), 32, 32)

		self.bonus = random.choice([
			self.BONUS_GRENADE,
			self.BONUS_HELMET,
			self.BONUS_SHOVEL,
			self.BONUS_STAR,
			self.BONUS_TANK,
			self.BONUS_TIMER
		])

		self.image = sprites.subsurface(16*2*self.bonus, 32*2, 16*2, 15*2)

	def draw(self):
		""" draw bonus """
		global screen
		if self.visible:
			screen.blit(self.image, self.rect.topleft)

	def toggleVisibility(self):
		""" Toggle bonus visibility """
		self.visible = not self.visible
#处理子弹速度、方向等数据，判断是否摧毁坦克或方块
class Bullet():
	
	(DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT) = range(4)

	
	(STATE_REMOVED, STATE_ACTIVE, STATE_EXPLODING) = range(3)

	(OWNER_PLAYER, OWNER_ENEMY) = range(2)

	def __init__(self, level, position, direction, damage = 100, speed = 5):

		global sprites

		self.level = level
		self.direction = direction
		self.damage = damage
		self.owner = None
		self.owner_class = None

		
		
		self.power = 1

		self.image = sprites.subsurface(75*2, 74*2, 3*2, 4*2)

		
		
		if direction == self.DIR_UP:
			self.rect = pygame.Rect(position[0] + 11, position[1] - 8, 6, 8)
		elif direction == self.DIR_RIGHT:
			self.image = pygame.transform.rotate(self.image, 270)
			self.rect = pygame.Rect(position[0] + 26, position[1] + 11, 8, 6)
		elif direction == self.DIR_DOWN:
			self.image = pygame.transform.rotate(self.image, 180)
			self.rect = pygame.Rect(position[0] + 11, position[1] + 26, 6, 8)
		elif direction == self.DIR_LEFT:
			self.image = pygame.transform.rotate(self.image, 90)
			self.rect = pygame.Rect(position[0] - 8 , position[1] + 11, 8, 6)

		self.explosion_images = [
			sprites.subsurface(0, 80*2, 32*2, 32*2),
			sprites.subsurface(32*2, 80*2, 32*2, 32*2),
		]

		self.speed = speed

		self.state = self.STATE_ACTIVE

	def draw(self):
		""" draw bullet """
		global screen
		if self.state == self.STATE_ACTIVE:
			screen.blit(self.image, self.rect.topleft)
		elif self.state == self.STATE_EXPLODING:
			self.explosion.draw()

	def update(self):
		global castle, players, enemies, bullets

		if self.state == self.STATE_EXPLODING:
			if not self.explosion.active:
				self.destroy()
				del self.explosion

		if self.state != self.STATE_ACTIVE:
			return

		""" move bullet """
		if self.direction == self.DIR_UP:
			self.rect.topleft = [self.rect.left, self.rect.top - self.speed]
			if self.rect.top < 0:
				if play_sounds and self.owner == self.OWNER_PLAYER:
					sounds["steel"].play()
				self.explode()
				return
		elif self.direction == self.DIR_RIGHT:
			self.rect.topleft = [self.rect.left + self.speed, self.rect.top]
			if self.rect.left > (416 - self.rect.width):
				if play_sounds and self.owner == self.OWNER_PLAYER:
					sounds["steel"].play()
				self.explode()
				return
		elif self.direction == self.DIR_DOWN:
			self.rect.topleft = [self.rect.left, self.rect.top + self.speed]
			if self.rect.top > (416 - self.rect.height):
				if play_sounds and self.owner == self.OWNER_PLAYER:
					sounds["steel"].play()
				self.explode()
				return
		elif self.direction == self.DIR_LEFT:
			self.rect.topleft = [self.rect.left - self.speed, self.rect.top]
			if self.rect.left < 0:
				if play_sounds and self.owner == self.OWNER_PLAYER:
					sounds["steel"].play()
				self.explode()
				return

		has_collided = False

		
		
		rects = self.level.obstacle_rects
		collisions = self.rect.collidelistall(rects)
		if collisions != []:
			for i in collisions:
				if self.level.hitTile(rects[i].topleft, self.power, self.owner == self.OWNER_PLAYER):
					has_collided = True
		if has_collided:
			self.explode()
			return

		
		for bullet in bullets:
			if self.state == self.STATE_ACTIVE and bullet.owner != self.owner and bullet != self and self.rect.colliderect(bullet.rect):
				self.destroy()
				self.explode()
				return

		
		for player in players:
			if player.state == player.STATE_ALIVE and self.rect.colliderect(player.rect):
				if player.bulletImpact(self.owner == self.OWNER_PLAYER, self.damage, self.owner_class):
					if self.owner == self.OWNER_ENEMY:
						self.destroy()
					return

		
		for enemy in enemies:
			if enemy.state == enemy.STATE_ALIVE and self.rect.colliderect(enemy.rect):
				if enemy.bulletImpact(self.owner == self.OWNER_ENEMY, self.damage, self.owner_class):
					self.destroy()
					return

		
		if castle.active and self.rect.colliderect(castle.rect):
			castle.destroy()
			self.destroy()
			return

	def explode(self):
		""" start bullets's explosion """
		global screen
		if self.state != self.STATE_REMOVED:
			self.state = self.STATE_EXPLODING
			self.explosion = Explosion([self.rect.left-13, self.rect.top-13], None, self.explosion_images)

	def destroy(self):
		self.state = self.STATE_REMOVED
#文本标签，展示分数等数值
class Label():
	def __init__(self, position, text = "", duration = None):

		self.position = position

		self.active = True

		self.text = text

		self.font = pygame.font.SysFont("Arial", 13)

		if duration != None:
			gtimer.add(duration, lambda :self.destroy(), 1)

	def draw(self):
		""" draw label """
		
		

	def destroy(self):
		self.active = False
#爆炸效果，管理动画的播放
class Explosion():
	def __init__(self, position, interval = None, images = None):

		global sprites

		self.position = [position[0]-16, position[1]-16]
		self.active = True

		if interval == None:
			interval = 1

		if images == None:
			images = [
				sprites.subsurface(0, 80*2, 32*2, 32*2),
				sprites.subsurface(32*2, 80*2, 32*2, 32*2),
				sprites.subsurface(64*2, 80*2, 32*2, 32*2)
			]

		images.reverse()

		self.images = [] + images

		self.image = self.images.pop()

		gtimer.add(interval, lambda :self.update(), len(self.images) + 1)

	def draw(self):
		global screen
		""" draw current explosion frame """
		screen.blit(self.image, self.position)

	def update(self):
		""" Advace to the next image """
		if len(self.images) > 0:
			self.image = self.images.pop()
		else:
			self.active = False
#游戏关卡，由levels文件中的符号串来生成地图
class Level():

	
	(TILE_EMPTY, TILE_BRICK, TILE_STEEL, TILE_WATER, TILE_GRASS, TILE_FROZE) = range(6)

	
	TILE_SIZE = 16

	def __init__(self, level_nr = None):
		""" There are total 35 different levels. If level_nr is larger than 35, loop over
		to next according level so, for example, if level_nr ir 37, then load level 2 """

		global sprites

		
		self.max_active_enemies = 4

		tile_images = [
			pygame.Surface((8*2, 8*2)),
			sprites.subsurface(48*2, 64*2, 8*2, 8*2),
			sprites.subsurface(48*2, 72*2, 8*2, 8*2),
			sprites.subsurface(56*2, 72*2, 8*2, 8*2),
			sprites.subsurface(64*2, 64*2, 8*2, 8*2),
			sprites.subsurface(64*2, 64*2, 8*2, 8*2),
			sprites.subsurface(72*2, 64*2, 8*2, 8*2),
			sprites.subsurface(64*2, 72*2, 8*2, 8*2)
		]
		self.tile_empty = tile_images[0]
		self.tile_brick = tile_images[1]
		self.tile_steel = tile_images[2]
		self.tile_grass = tile_images[3]
		self.tile_water = tile_images[4]
		self.tile_water1= tile_images[4]
		self.tile_water2= tile_images[5]
		self.tile_froze = tile_images[6]

		self.obstacle_rects = []

		
		
		

		self.loadLevel(level_nr)

		
		self.obstacle_rects = []

		
		self.updateObstacleRects()

		gtimer.add(400, lambda :self.toggleWaves())

	def hitTile(self, pos, power = 1, sound = False):
		"""
			Hit the tile
			@param pos Tile's x, y in px
			@return True if bullet was stopped, False otherwise
		"""

		global play_sounds, sounds

		for tile in self.mapr:
			if tile.topleft == pos:
				if tile.type == self.TILE_BRICK:
					if play_sounds and sound:
						sounds["brick"].play()
					self.mapr.remove(tile)
					self.updateObstacleRects()
					return True
				elif tile.type == self.TILE_STEEL:
					if play_sounds and sound:
						sounds["steel"].play()
					if power == 2:
						self.mapr.remove(tile)
						self.updateObstacleRects()
					return True
				else:
					return False

	def toggleWaves(self):
		""" Toggle water image """
		if self.tile_water == self.tile_water1:
			self.tile_water = self.tile_water2
		else:
			self.tile_water = self.tile_water1


	def loadLevel(self, level_nr = 1):
		""" Load specified level
		@return boolean Whether level was loaded
		"""
		filename = "levels/gameplay/"+str(level_nr)
		if (not os.path.isfile(filename)):
			return False
		level = []
		f = open(filename, "r")
		data = f.read().split("\n")
		self.mapr = []
		x, y = 0, 0
		for row in data:
			for ch in row:
				if ch == "#":
					self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_BRICK))
				elif ch == "@":
					self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_STEEL))
				elif ch == "~":
					self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_WATER))
				elif ch == "%":
					self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_GRASS))
				elif ch == "-":
					self.mapr.append(myRect(x, y, self.TILE_SIZE, self.TILE_SIZE, self.TILE_FROZE))
				x += self.TILE_SIZE
			x = 0
			y += self.TILE_SIZE
		return True


	def draw(self, tiles = None):
		""" Draw specified map on top of existing surface """

		global screen
		
		
		pygame.draw.line(screen, (255, 0, 0), (64, 416), (64, 208), 3)
		pygame.draw.line(screen, (255, 0, 0), (64, 208), (352, 208), 3)
		pygame.draw.line(screen, (255, 0, 0), (352, 208), (352, 416), 3)

		if tiles == None:
			tiles = [TILE_BRICK, TILE_STEEL, TILE_WATER, TILE_GRASS, TILE_FROZE]

		for tile in self.mapr:
			if tile.type in tiles:
				if tile.type == self.TILE_BRICK:
					screen.blit(self.tile_brick, tile.topleft)
				elif tile.type == self.TILE_STEEL:
					screen.blit(self.tile_steel, tile.topleft)
				elif tile.type == self.TILE_WATER:
					screen.blit(self.tile_water, tile.topleft)
				elif tile.type == self.TILE_FROZE:
					screen.blit(self.tile_froze, tile.topleft)
				elif tile.type == self.TILE_GRASS:
					screen.blit(self.tile_grass, tile.topleft)

	def updateObstacleRects(self):
		""" Set self.obstacle_rects to all tiles' rects that players can destroy
		with bullets """

		global castle

		self.obstacle_rects = [castle.rect]

		for tile in self.mapr:
			if tile.type in (self.TILE_BRICK, self.TILE_STEEL, self.TILE_WATER):
				self.obstacle_rects.append(tile)

	def buildFortress(self, tile):
		""" Build walls around castle made from tile """

		positions = [
			(11*self.TILE_SIZE, 23*self.TILE_SIZE),
			(11*self.TILE_SIZE, 24*self.TILE_SIZE),
			(11*self.TILE_SIZE, 25*self.TILE_SIZE),
			(14*self.TILE_SIZE, 23*self.TILE_SIZE),
			(14*self.TILE_SIZE, 24*self.TILE_SIZE),
			(14*self.TILE_SIZE, 25*self.TILE_SIZE),
			(12*self.TILE_SIZE, 23*self.TILE_SIZE),
			(13*self.TILE_SIZE, 23*self.TILE_SIZE)
		]

		obsolete = []

		for i, rect in enumerate(self.mapr):
			if rect.topleft in positions:
				obsolete.append(rect)
		for rect in obsolete:
			self.mapr.remove(rect)

		for pos in positions:
			self.mapr.append(myRect(pos[0], pos[1], self.TILE_SIZE, self.TILE_SIZE, tile))

		self.updateObstacleRects()
#管理坦克的数据状态与基本功能
class Tank():

	
	(DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT) = range(4)

	
	(STATE_SPAWNING, STATE_DEAD, STATE_ALIVE, STATE_EXPLODING) = range(4)

	
	(SIDE_PLAYER, SIDE_ENEMY) = range(2)

	def __init__(self, level, side, position = None, direction = None, filename = None):

		global sprites

		
		self.health = 100

		
		self.paralised = False

		
		self.paused = False

		
		self.shielded = False

		
		self.speed = 5*2

		
		self.max_active_bullets = 1

		
		self.side = side

		
		self.flash = 0

		
		
		
		
		self.superpowers = 0

		
		self.bonus = None

		
		self.controls = [pygame.K_SPACE, pygame.K_UP, pygame.K_RIGHT, pygame.K_DOWN, pygame.K_LEFT]

		
		self.pressed = [False] * 4

		self.shield_images = [
			sprites.subsurface(0, 48*2, 16*2, 16*2),
			sprites.subsurface(16*2, 48*2, 16*2, 16*2)
		]
		self.shield_image = self.shield_images[0]
		self.shield_index = 0

		self.spawn_images = [
			sprites.subsurface(32*2, 48*2, 16*2, 16*2),
			sprites.subsurface(48*2, 48*2, 16*2, 16*2)
		]
		self.spawn_image = self.spawn_images[0]
		self.spawn_index = 0

		self.level = level

		if position != None:
			self.rect = pygame.Rect(position, (26, 26))
		else:
			self.rect = pygame.Rect(0, 0, 26, 26)

		if direction == None:
			self.direction = random.choice([self.DIR_RIGHT, self.DIR_DOWN, self.DIR_LEFT])
		else:
			self.direction = direction

		self.state = self.STATE_SPAWNING

		
		self.timer_uuid_spawn = gtimer.add(100, lambda :self.toggleSpawnImage())

		
		self.timer_uuid_spawn_end = gtimer.add(1000, lambda :self.endSpawning())

	def endSpawning(self):
		""" End spawning
		Player becomes operational
		"""
		self.state = self.STATE_ALIVE
		gtimer.destroy(self.timer_uuid_spawn_end)


	def toggleSpawnImage(self):
		""" advance to the next spawn image """
		if self.state != self.STATE_SPAWNING:
			gtimer.destroy(self.timer_uuid_spawn)
			return
		self.spawn_index += 1
		if self.spawn_index >= len(self.spawn_images):
			self.spawn_index = 0
		self.spawn_image = self.spawn_images[self.spawn_index]

	def toggleShieldImage(self):
		""" advance to the next shield image """
		if self.state != self.STATE_ALIVE:
			gtimer.destroy(self.timer_uuid_shield)
			return
		if self.shielded:
			self.shield_index += 1
			if self.shield_index >= len(self.shield_images):
				self.shield_index = 0
			self.shield_image = self.shield_images[self.shield_index]


	def draw(self):
		""" draw tank """
		global screen
		if self.state == self.STATE_ALIVE:
			screen.blit(self.image, self.rect.topleft)
			if self.shielded:
				screen.blit(self.shield_image, [self.rect.left-3, self.rect.top-3])
		elif self.state == self.STATE_EXPLODING:
			self.explosion.draw()
		elif self.state == self.STATE_SPAWNING:
			screen.blit(self.spawn_image, self.rect.topleft)

	def explode(self):
		""" start tanks's explosion """
		if self.state != self.STATE_DEAD:
			self.state = self.STATE_EXPLODING
			self.explosion = Explosion(self.rect.topleft)

			if self.bonus:
				pass
				

	def fire(self, forced = False):
		global obs_flag_bullet_fired
		""" Shoot a bullet
		@param boolean forced. If false, check whether tank has exceeded his bullet quota. Default: True
		@return boolean True if bullet was fired, false otherwise
		"""

		global bullets, labels

		if self.state != self.STATE_ALIVE:
			gtimer.destroy(self.timer_uuid_fire)
			return False

		if self.paused:
			return False

		if not forced:
			active_bullets = 0
			for bullet in bullets:
				if bullet.owner_class == self and bullet.state == bullet.STATE_ACTIVE:
					active_bullets += 1
			if active_bullets >= self.max_active_bullets:
				return False

		bullet = Bullet(self.level, self.rect.topleft, self.direction)

		
		if self.superpowers > 0:
			bullet.speed = 5*8

		
		if self.superpowers > 2:
			bullet.power = 2

		if self.side == self.SIDE_PLAYER:
			bullet.owner = self.SIDE_PLAYER
		else:
			bullet.owner = self.SIDE_ENEMY
			self.bullet_queued = False

		bullet.owner_class = self
		bullets.append(bullet)
		return True

	def rotate(self, direction, fix_position = True):
		""" Rotate tank
		rotate, update image and correct position
		"""
		self.direction = direction

		if direction == self.DIR_UP:
			self.image = self.image_up
		elif direction == self.DIR_RIGHT:
			self.image = self.image_right
		elif direction == self.DIR_DOWN:
			self.image = self.image_down
		elif direction == self.DIR_LEFT:
			self.image = self.image_left

		if fix_position:
			new_x = self.nearest(self.rect.left, 8) + 3
			new_y = self.nearest(self.rect.top, 8) + 3

			if (abs(self.rect.left - new_x) < 5):
				self.rect.left = new_x

			if (abs(self.rect.top - new_y) < 5):
				self.rect.top = new_y

	def turnAround(self):
		""" Turn tank into opposite direction """
		if self.direction in (self.DIR_UP, self.DIR_RIGHT):
			self.rotate(self.direction + 2, False)
		else:
			self.rotate(self.direction - 2, False)

	def update(self, time_passed):
		""" Update timer and explosion (if any) """
		if self.state == self.STATE_EXPLODING:
			if not self.explosion.active:
				self.state = self.STATE_DEAD
				del self.explosion

	def nearest(self, num, base):
		""" Round number to nearest divisible """
		return int(round(num / (base * 1.0)) * base)


	def bulletImpact(self, friendly_fire = False, damage = 100, tank = None):
		""" Bullet impact
		Return True if bullet should be destroyed on impact. Only enemy friendly-fire
		doesn't trigger bullet explosion
		"""

		global play_sounds, sounds

		if self.shielded:
			return True

		if not friendly_fire:
			self.health -= damage
			if self.health < 1:
				if self.side == self.SIDE_ENEMY:
					tank.trophies["enemy"+str(self.type)] += 1
					points = (self.type+1) * 100
					tank.score += points
					if play_sounds:
						sounds["explosion"].play()

					labels.append(Label(self.rect.topleft, str(points), 500))

				self.explode()
			return True

		if self.side == self.SIDE_ENEMY:
			return False
		elif self.side == self.SIDE_PLAYER:
			
			
			
			return False

	def setParalised(self, paralised = True):
		""" set tank paralise state
		@param boolean paralised
		@return None
		"""
		if self.state != self.STATE_ALIVE:
			gtimer.destroy(self.timer_uuid_paralise)
			return
		self.paralised = paralised
#管理敌人的AI，继承Tank类，设计其路径，行为与数值
class Enemy(Tank):

	(TYPE_BASIC, TYPE_FAST, TYPE_POWER, TYPE_ARMOR) = range(4)

	def __init__(self, level, type, position = None, direction = None, filename = None):

		Tank.__init__(self, level, type, position = None, direction = None, filename = None)

		global enemies, sprites

		
		self.bullet_queued = False

		
		if len(level.enemies_left) > 0:
			self.type = level.enemies_left.pop()
		else:
			self.state = self.STATE_DEAD
			return

		if self.type == self.TYPE_BASIC:
			self.speed = 5*1
		elif self.type == self.TYPE_FAST:
			self.speed = 5*3
		elif self.type == self.TYPE_POWER:
			self.superpowers = 5*1
		elif self.type == self.TYPE_ARMOR:
			self.health = 400

		
		if random.randint(1, 5) == 1:
			self.bonus = True
			for enemy in enemies:
				if enemy.bonus:
					self.bonus = False
					break

		images = [
			sprites.subsurface(32*2, 0, 13*2, 15*2),
			sprites.subsurface(48*2, 0, 13*2, 15*2),
			sprites.subsurface(64*2, 0, 13*2, 15*2),
			sprites.subsurface(80*2, 0, 13*2, 15*2),
			sprites.subsurface(32*2, 16*2, 13*2, 15*2),
			sprites.subsurface(48*2, 16*2, 13*2, 15*2),
			sprites.subsurface(64*2, 16*2, 13*2, 15*2),
			sprites.subsurface(80*2, 16*2, 13*2, 15*2)
		]

		self.image = images[self.type+0]

		self.image_up = self.image
		self.image_left = pygame.transform.rotate(self.image, 90)
		self.image_down = pygame.transform.rotate(self.image, 180)
		self.image_right = pygame.transform.rotate(self.image, 270)

		if self.bonus:
			self.image1_up = self.image_up
			self.image1_left = self.image_left
			self.image1_down = self.image_down
			self.image1_right = self.image_right

			self.image2 = images[self.type+4]
			self.image2_up = self.image2
			self.image2_left = pygame.transform.rotate(self.image2, 90)
			self.image2_down = pygame.transform.rotate(self.image2, 180)
			self.image2_right = pygame.transform.rotate(self.image2, 270)

		self.rotate(self.direction, False)

		if position == None:
			self.rect.topleft = self.getFreeSpawningPosition()
			if not self.rect.topleft:
				self.state = self.STATE_DEAD
				return

		
		self.path = self.generatePath(self.direction)

		
		self.timer_uuid_fire = gtimer.add(1000, lambda :self.fire())

		
		if self.bonus:
			self.timer_uuid_flash = gtimer.add(200, lambda :self.toggleFlash())

	def toggleFlash(self):
		""" Toggle flash state """
		if self.state not in (self.STATE_ALIVE, self.STATE_SPAWNING):
			gtimer.destroy(self.timer_uuid_flash)
			return
		self.flash = not self.flash
		if self.flash:
			self.image_up = self.image2_up
			self.image_right = self.image2_right
			self.image_down = self.image2_down
			self.image_left = self.image2_left
		else:
			self.image_up = self.image1_up
			self.image_right = self.image1_right
			self.image_down = self.image1_down
			self.image_left = self.image1_left
		self.rotate(self.direction, False)

	def spawnBonus(self):
		""" Create new bonus if needed """

		global bonuses

		if len(bonuses) > 0:
			return
		bonus = Bonus(self.level)
		bonuses.append(bonus)
		gtimer.add(500, lambda :bonus.toggleVisibility())
		gtimer.add(10000, lambda :bonuses.remove(bonus), 1)


	def getFreeSpawningPosition(self):

		global players, enemies

		available_positions = [
			[(self.level.TILE_SIZE * 2 - self.rect.width) / 2, (self.level.TILE_SIZE * 2 - self.rect.height) / 2],
			[12 * self.level.TILE_SIZE + (self.level.TILE_SIZE * 2 - self.rect.width) / 2, (self.level.TILE_SIZE * 2 - self.rect.height) / 2],
			[24 * self.level.TILE_SIZE + (self.level.TILE_SIZE * 2 - self.rect.width) / 2,  (self.level.TILE_SIZE * 2 - self.rect.height) / 2],
			
			
		]

		random.shuffle(available_positions)

		for pos in available_positions:

			enemy_rect = pygame.Rect(pos, [26, 26])

			
			collision = False
			for enemy in enemies:
				if enemy_rect.colliderect(enemy.rect):
					collision = True
					continue

			if collision:
				continue

			
			collision = False
			for player in players:
				if enemy_rect.colliderect(player.rect):
					collision = True
					continue

			if collision:
				continue

			return pos
		return False

	def move(self):
		""" move enemy if possible """

		global players, enemies, bonuses

		if self.state != self.STATE_ALIVE or self.paused or self.paralised:
			return

		if self.path == []:
			self.path = self.generatePath(None, True)

		new_position = self.path.pop(0)

		
		if self.direction == self.DIR_UP:
			if new_position[1] < 0:
				self.path = self.generatePath(self.direction, True)
				return
		elif self.direction == self.DIR_RIGHT:
			if new_position[0] > (416 - 26):
				self.path = self.generatePath(self.direction, True)
				return
		elif self.direction == self.DIR_DOWN:
			if new_position[1] > (416 - 26):
				self.path = self.generatePath(self.direction, True)
				return
		elif self.direction == self.DIR_LEFT:
			if new_position[0] < 0:
				self.path = self.generatePath(self.direction, True)
				return

		new_rect = pygame.Rect(new_position, [26, 26])

		
		if new_rect.collidelist(self.level.obstacle_rects) != -1:
			self.path = self.generatePath(self.direction, True)
			return

		
		for enemy in enemies:
			if enemy != self and new_rect.colliderect(enemy.rect):
				self.turnAround()
				self.path = self.generatePath(self.direction)
				return

		
		for player in players:
			if new_rect.colliderect(player.rect):
				self.turnAround()
				self.path = self.generatePath(self.direction)
				return

		
		for bonus in bonuses:
			if new_rect.colliderect(bonus.rect):
				bonuses.remove(bonus)

		
		self.rect.topleft = new_rect.topleft


	def update(self, time_passed):
		Tank.update(self, time_passed)
		if self.state == self.STATE_ALIVE and not self.paused:
			self.move()

	def generatePath(self, direction = None, fix_direction = False):
		""" If direction is specified, try continue that way, otherwise choose at random
		"""

		all_directions = [self.DIR_UP, self.DIR_RIGHT, self.DIR_DOWN, self.DIR_LEFT]

		if direction == None:
			if self.direction in [self.DIR_UP, self.DIR_RIGHT]:
				opposite_direction = self.direction + 2
			else:
				opposite_direction = self.direction - 2
			directions = all_directions
			random.shuffle(directions)
			directions.remove(opposite_direction)
			directions.append(opposite_direction)
		else:
			if direction in [self.DIR_UP, self.DIR_RIGHT]:
				opposite_direction = direction + 2
			else:
				opposite_direction = direction - 2

			if direction in [self.DIR_UP, self.DIR_RIGHT]:
				opposite_direction = direction + 2
			else:
				opposite_direction = direction - 2
			directions = all_directions
			random.shuffle(directions)
			directions.remove(opposite_direction)
			directions.remove(direction)
			directions.insert(0, direction)
			directions.append(opposite_direction)

		
		x = int(round(self.rect.left / 16))
		y = int(round(self.rect.top / 16))

		new_direction = None

		for direction in directions:
			if direction == self.DIR_UP and y > 1:
				new_pos_rect = self.rect.move(0, -8)
				if new_pos_rect.collidelist(self.level.obstacle_rects) == -1:
					new_direction = direction
					break
			elif direction == self.DIR_RIGHT and x < 24:
				new_pos_rect = self.rect.move(8, 0)
				if new_pos_rect.collidelist(self.level.obstacle_rects) == -1:
					new_direction = direction
					break
			elif direction == self.DIR_DOWN and y < 24:
				new_pos_rect = self.rect.move(0, 8)
				if new_pos_rect.collidelist(self.level.obstacle_rects) == -1:
					new_direction = direction
					break
			elif direction == self.DIR_LEFT and x > 1:
				new_pos_rect = self.rect.move(-8, 0)
				if new_pos_rect.collidelist(self.level.obstacle_rects) == -1:
					new_direction = direction
					break

		
		if new_direction == None:
			new_direction = opposite_direction

		
		if fix_direction and new_direction == self.direction:
			fix_direction = False

		self.rotate(new_direction, fix_direction)

		positions = []

		x = self.rect.left
		y = self.rect.top

		if new_direction in (self.DIR_RIGHT, self.DIR_LEFT):
			axis_fix = self.nearest(y, 16) - y
		else:
			axis_fix = self.nearest(x, 16) - x
		axis_fix = 0

		pixels = self.nearest(random.randint(1, 12) * 32, 32) + axis_fix + 3

		if new_direction == self.DIR_UP:
			for px in range(0, pixels, self.speed):
				positions.append([x, y-px])
		elif new_direction == self.DIR_RIGHT:
			for px in range(0, pixels, self.speed):
				positions.append([x+px, y])
		elif new_direction == self.DIR_DOWN:
			for px in range(0, pixels, self.speed):
				positions.append([x, y+px])
		elif new_direction == self.DIR_LEFT:
			for px in range(0, pixels, self.speed):
				positions.append([x-px, y])

		return positions
#管理玩家的得分，道具收集情况与数值，处理玩家输入
class Player(Tank):

	def __init__(self, level, type, position = None, direction = None, filename = None):

		Tank.__init__(self, level, type, position = None, direction = None, filename = None)

		global sprites

		if filename == None:
			filename = (0, 0, 16*2, 16*2)

		self.start_position = position
		self.start_direction = direction

		self.lives = 3 

		
		self.score = 0

		
		self.trophies = {
			"bonus" : 0,
			"enemy0" : 0,
			"enemy1" : 0,
			"enemy2" : 0,
			"enemy3" : 0
		}

		self.image = sprites.subsurface(filename)
		self.image_up = self.image
		self.image_left = pygame.transform.rotate(self.image, 90)
		self.image_down = pygame.transform.rotate(self.image, 180)
		self.image_right = pygame.transform.rotate(self.image, 270)

		if direction == None:
			self.rotate(self.DIR_UP, False)
		else:
			self.rotate(direction, False)

	def move(self, direction):
		global obs_flag_player_collision
		""" move player if possible """

		global players, enemies, bonuses

		if self.state == self.STATE_EXPLODING:
			if not self.explosion.active:
				self.state = self.STATE_DEAD
				del self.explosion

		if self.state != self.STATE_ALIVE:
			return

		
		if self.direction != direction:
			self.rotate(direction)

		if self.paralised:
			return

		
		if direction == self.DIR_UP:
			new_position = [self.rect.left, self.rect.top - self.speed]
			if new_position[1] < 0 + 416 // 2: 
				obs_flag_player_collision = 1
				return
		elif direction == self.DIR_RIGHT:
			new_position = [self.rect.left + self.speed, self.rect.top]
			if new_position[0] > (416 - 26) - 32*2: 
				obs_flag_player_collision = 1
				return
		elif direction == self.DIR_DOWN:
			new_position = [self.rect.left, self.rect.top + self.speed]
			if new_position[1] > (416 - 26):
				obs_flag_player_collision = 1
				return
		elif direction == self.DIR_LEFT:
			new_position = [self.rect.left - self.speed, self.rect.top]
			if new_position[0] < 0 + 32*2: 
				obs_flag_player_collision = 1
				return

		player_rect = pygame.Rect(new_position, [26, 26])

		
		if player_rect.collidelist(self.level.obstacle_rects) != -1:
			obs_flag_player_collision = 1
			return

		
		for player in players:
			if player != self and player.state == player.STATE_ALIVE and player_rect.colliderect(player.rect) == True:
				obs_flag_player_collision = 1
				return

		
		for enemy in enemies:
			if player_rect.colliderect(enemy.rect) == True:
				obs_flag_player_collision = 1
				return

		
		for bonus in bonuses:
			if player_rect.colliderect(bonus.rect) == True:
				self.bonus = bonus

		
		self.rect.topleft = (new_position[0], new_position[1])

	def reset(self, pos):
		""" reset player """
		self.start_position = pos
		self.start_direction = random.randint(0, 3)
		self.rotate(self.start_direction, False)
		self.rect.topleft = self.start_position
		self.superpowers = 0
		self.max_active_bullets = 1
		self.health = 100
		self.paralised = False
		self.paused = False
		self.pressed = [False] * 4
		self.state = self.STATE_ALIVE
#管理游戏整体流程和状态处理游戏初始化、菜单、关卡切换、得分计算等，协调所有其他类的交互，实现游戏主循环
class Game():
	
	(DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT) = range(4)

	TILE_SIZE = 16

	def __init__(self):

		global screen, sprites, play_sounds, sounds

		
		os.environ['SDL_VIDEO_WINDOW_POS'] = 'center'

		if play_sounds:
			pygame.mixer.pre_init(44100, -16, 1, 512)

		pygame.init()


		pygame.display.set_caption("Battle City")

		size = width, height = 480, 416

		if "-f" in sys.argv[1:]:
			screen = pygame.display.set_mode(size, pygame.FULLSCREEN)
		else:
			
			screen = pygame.display.set_mode(size)
			
			

		self.clock = pygame.time.Clock()

		sprites = pygame.transform.scale(pygame.image.load("images/sprites.gif"), [192, 224])

		pygame.display.set_icon(sprites.subsurface(0, 0, 13*2, 13*2))

		
		if play_sounds:
			pygame.mixer.init(44100, -16, 1, 512)

			sounds["start"] = pygame.mixer.Sound("sounds/gamestart.ogg")
			sounds["end"] = pygame.mixer.Sound("sounds/gameover.ogg")
			sounds["score"] = pygame.mixer.Sound("sounds/score.ogg")
			sounds["bg"] = pygame.mixer.Sound("sounds/background.ogg")
			sounds["fire"] = pygame.mixer.Sound("sounds/fire.ogg")
			sounds["bonus"] = pygame.mixer.Sound("sounds/bonus.ogg")
			sounds["explosion"] = pygame.mixer.Sound("sounds/explosion.ogg")
			sounds["brick"] = pygame.mixer.Sound("sounds/brick.ogg")
			sounds["steel"] = pygame.mixer.Sound("sounds/steel.ogg")

		self.enemy_life_image = sprites.subsurface(81*2, 57*2, 7*2, 7*2)
		self.player_life_image = sprites.subsurface(89*2, 56*2, 7*2, 8*2)
		self.flag_image = sprites.subsurface(64*2, 49*2, 16*2, 15*2)

		
		self.player_image = pygame.transform.rotate(sprites.subsurface(0, 0, 13*2, 13*2), 270)

		
		self.timefreeze = False

		
		self.font = pygame.font.Font("fonts/prstart.ttf", 16)

		
		self.im_game_over = pygame.Surface((64, 40))
		self.im_game_over.set_colorkey((0,0,0))
		self.im_game_over.blit(self.font.render("GAME", False, (127, 64, 64)), [0, 0])
		self.im_game_over.blit(self.font.render("OVER", False, (127, 64, 64)), [0, 20])
		self.game_over_y = 416+40

		
		self.nr_of_players = 1
		self.available_positions = []

		del players[:]
		del bullets[:]
		del enemies[:]
		del bonuses[:]


	def triggerBonus(self, bonus, player):
		""" Execute bonus powers """

		global enemies, labels, play_sounds, sounds

		if play_sounds:
			sounds["bonus"].play()

		player.trophies["bonus"] += 1
		player.score += 500

		if bonus.bonus == bonus.BONUS_GRENADE:
			for enemy in enemies:
				enemy.explode()
		elif bonus.bonus == bonus.BONUS_HELMET:
			self.shieldPlayer(player, True, 10000)
		elif bonus.bonus == bonus.BONUS_SHOVEL:
			self.level.buildFortress(self.level.TILE_STEEL)
			gtimer.add(10000, lambda :self.level.buildFortress(self.level.TILE_BRICK), 1)
		elif bonus.bonus == bonus.BONUS_STAR:
			player.superpowers += 1
			if player.superpowers == 2:
				player.max_active_bullets = 2
		elif bonus.bonus == bonus.BONUS_TANK:
			player.lives += 1
		elif bonus.bonus == bonus.BONUS_TIMER:
			self.toggleEnemyFreeze(True)
			gtimer.add(10000, lambda :self.toggleEnemyFreeze(False), 1)
		bonuses.remove(bonus)

		labels.append(Label(bonus.rect.topleft, "500", 500))

	def shieldPlayer(self, player, shield = True, duration = None):
		""" Add/remove shield
		player: player (not enemy)
		shield: true/false
		duration: in ms. if none, do not remove shield automatically
		"""
		player.shielded = shield
		if shield:
			player.timer_uuid_shield = gtimer.add(100, lambda :player.toggleShieldImage())
		else:
			gtimer.destroy(player.timer_uuid_shield)

		if shield and duration != None:
			gtimer.add(duration, lambda :self.shieldPlayer(player, False), 1)


	def spawnEnemy(self):
		""" Spawn new enemy if needed
		Only add enemy if:
			- there are at least one in queue
			- map capacity hasn't exceeded its quota
			- now isn't timefreeze
		"""

		global enemies

		if len(enemies) >= self.level.max_active_enemies:
			return
		if len(self.level.enemies_left) < 1 or self.timefreeze:
			return
		enemy = Enemy(self.level, 1)

		enemies.append(enemy)


	def respawnPlayer(self, player, clear_scores = False):
		""" Respawn player """
		n = random.randint(0, len(self.available_positions) - 1)
		[kx, ky] = self.available_positions[n]
		x = kx * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2
		y = ky * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2
		pos = [x, y]

		player.reset(pos)

		if clear_scores:
			player.trophies = {
				"bonus" : 0, "enemy0" : 0, "enemy1" : 0, "enemy2" : 0, "enemy3" : 0
			}

		self.shieldPlayer(player, True, 4000)

	def gameOver(self):
		""" End game and return to menu """
		
		global play_sounds, sounds

		for player in players:
			player.lives = 3 

		
		if play_sounds:
			for sound in sounds:
				sounds[sound].stop()
			sounds["end"].play()

		self.game_over_y = 416+40

		self.game_over = True
		
		if self.game_over:
			self.stage = 0
			self.nextLevel()
			
		else:
			self.nextLevel()

	def gameOverScreen(self):
		""" Show game over screen """

		global screen

		
		self.running = False

		screen.fill([0, 0, 0])

		self.writeInBricks("game", [125, 140])
		self.writeInBricks("over", [125, 220])
		pygame.display.flip()

		while 1:
			time_passed = self.clock.tick(50)
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					quit()
				elif event.type == pygame.KEYDOWN:
					if event.key == pygame.K_RETURN:
						self.showMenu()
						return

	def showMenu(self):
		""" Show game menu
		Redraw screen only when up or down key is pressed. When enter is pressed,
		exit from this screen and start the game with selected number of players
		"""

		global players, screen, gtimer

		
		self.running = False

		
		del gtimer.timers[:]

		
		self.stage = 0
		self.nr_of_players = 1
		del players[:]
		self.nextLevel()
		

	def reloadPlayers(self):
		""" Init players
		If players already exist, just reset them
		"""

		global players
  
		if len(players) == 0:

			x = 8 * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2
			y = 24 * self.TILE_SIZE + (self.TILE_SIZE * 2 - 26) / 2

			player = Player(
				self.level, 0, [x, y], self.DIR_UP, (0, 0, 13*2, 13*2)
			)
			players.append(player)
			
		for player in players:
			player.level = self.level
			self.respawnPlayer(player, True)

	def showScores(self):
		""" Show level scores """

		global screen, sprites, players, play_sounds, sounds

		
		self.running = False

		
		del gtimer.timers[:]

		if play_sounds:
			for sound in sounds:
				sounds[sound].stop()

		hiscore = self.loadHiscore()

		
		if players[0].score > hiscore:
			hiscore = players[0].score
			self.saveHiscore(hiscore)
		if self.nr_of_players == 2 and players[1].score > hiscore:
			hiscore = players[1].score
			self.saveHiscore(hiscore)

		img_tanks = [
			sprites.subsurface(32*2, 0, 13*2, 15*2),
			sprites.subsurface(48*2, 0, 13*2, 15*2),
			sprites.subsurface(64*2, 0, 13*2, 15*2),
			sprites.subsurface(80*2, 0, 13*2, 15*2)
		]

		img_arrows = [
			sprites.subsurface(81*2, 48*2, 7*2, 7*2),
			sprites.subsurface(88*2, 48*2, 7*2, 7*2)
		]

		screen.fill([0, 0, 0])

		
		black = pygame.Color("black")
		white = pygame.Color("white")
		purple = pygame.Color(127, 64, 64)
		pink = pygame.Color(191, 160, 128)

		screen.blit(self.font.render("HI-SCORE", False, purple), [105, 35])
		screen.blit(self.font.render(str(hiscore), False, pink), [295, 35])

		screen.blit(self.font.render("STAGE"+str(self.stage).rjust(3), False, white), [170, 65])

		screen.blit(self.font.render("I-PLAYER", False, purple), [25, 95])

		
		screen.blit(self.font.render(str(players[0].score).rjust(8), False, pink), [25, 125])

		if self.nr_of_players == 2:
			screen.blit(self.font.render("II-PLAYER", False, purple), [310, 95])

			
			screen.blit(self.font.render(str(players[1].score).rjust(8), False, pink), [325, 125])

		
		for i in range(4):
			screen.blit(img_tanks[i], [226, 160+(i*45)])
			screen.blit(img_arrows[0], [206, 168+(i*45)])
			if self.nr_of_players == 2:
				screen.blit(img_arrows[1], [258, 168+(i*45)])

		screen.blit(self.font.render("TOTAL", False, white), [70, 335])

		
		pygame.draw.line(screen, white, [170, 330], [307, 330], 4)

		pygame.display.flip()

		self.clock.tick(2)

		interval = 5

		
		for i in range(4):

			
			tanks = players[0].trophies["enemy"+str(i)]

			for n in range(tanks+1):
				if n > 0 and play_sounds:
					sounds["score"].play()

				
				screen.blit(self.font.render(str(n-1).rjust(2), False, black), [170, 168+(i*45)])
				
				screen.blit(self.font.render(str(n).rjust(2), False, white), [170, 168+(i*45)])
				
				screen.blit(self.font.render(str((n-1) * (i+1) * 100).rjust(4)+" PTS", False, black), [25, 168+(i*45)])
				
				screen.blit(self.font.render(str(n * (i+1) * 100).rjust(4)+" PTS", False, white), [25, 168+(i*45)])
				pygame.display.flip()
				self.clock.tick(interval)

			if self.nr_of_players == 2:
				tanks = players[1].trophies["enemy"+str(i)]

				for n in range(tanks+1):

					if n > 0 and play_sounds:
						sounds["score"].play()

					screen.blit(self.font.render(str(n-1).rjust(2), False, black), [277, 168+(i*45)])
					screen.blit(self.font.render(str(n).rjust(2), False, white), [277, 168+(i*45)])

					screen.blit(self.font.render(str((n-1) * (i+1) * 100).rjust(4)+" PTS", False, black), [325, 168+(i*45)])
					screen.blit(self.font.render(str(n * (i+1) * 100).rjust(4)+" PTS", False, white), [325, 168+(i*45)])

					pygame.display.flip()
					self.clock.tick(interval)

			self.clock.tick(interval)

		
		tanks = sum([i for i in players[0].trophies.values()]) - players[0].trophies["bonus"]
		screen.blit(self.font.render(str(tanks).rjust(2), False, white), [170, 335])
		if self.nr_of_players == 2:
			tanks = sum([i for i in players[1].trophies.values()]) - players[1].trophies["bonus"]
			screen.blit(self.font.render(str(tanks).rjust(2), False, white), [277, 335])

		pygame.display.flip()

		
		self.clock.tick(1)
		self.clock.tick(1)

		if self.game_over:
			self.gameOverScreen()
		else:
			self.nextLevel()


	def draw(self):
		global screen, castle, players, enemies, bullets, bonuses

		screen.fill([0, 0, 0])

		self.level.draw([self.level.TILE_EMPTY, self.level.TILE_BRICK, self.level.TILE_STEEL, self.level.TILE_FROZE, self.level.TILE_WATER])

		castle.draw()

		for enemy in enemies:
			enemy.draw()

		for player in players:
			player.draw()

		for bullet in bullets:
			bullet.draw()

		for bonus in bonuses:
			bonus.draw()

		self.level.draw([self.level.TILE_GRASS])

		if self.game_over:
			if self.game_over_y > 188:
				self.game_over_y -= 4
			screen.blit(self.im_game_over, [176, self.game_over_y]) 

		self.drawSidebar() 

		pygame.display.flip()

	def drawSidebar(self):

		global screen, players, enemies

		x = 416
		y = 0
		screen.fill([100, 100, 100], pygame.Rect([416, 0], [64, 416]))

		xpos = x + 16
		ypos = y + 16

		
		for n in range(len(self.level.enemies_left) + len(enemies)):
			screen.blit(self.enemy_life_image, [xpos, ypos])
			if n % 2 == 1:
				xpos = x + 16
				ypos+= 17
			else:
				xpos += 17

		
		if pygame.font.get_init():
			text_color = pygame.Color('black')
			for n in range(len(players)):
				if n == 0:
					screen.blit(self.font.render(str(n+1)+"P", False, text_color), [x+16, y+200])
					screen.blit(self.font.render(str(players[n].lives), False, text_color), [x+31, y+215])
					screen.blit(self.player_life_image, [x+17, y+215])
				else:
					screen.blit(self.font.render(str(n+1)+"P", False, text_color), [x+16, y+240])
					screen.blit(self.font.render(str(players[n].lives), False, text_color), [x+31, y+255])
					screen.blit(self.player_life_image, [x+17, y+255])

			screen.blit(self.flag_image, [x+17, y+280])
			screen.blit(self.font.render(str(self.stage), False, text_color), [x+17, y+312])


	def drawIntroScreen(self, put_on_surface = True):
		""" Draw intro (menu) screen
		@param boolean put_on_surface If True, flip display after drawing
		@return None
		"""

		global screen

		screen.fill([0, 0, 0])

		if pygame.font.get_init():

			hiscore = self.loadHiscore()

			screen.blit(self.font.render("HI- "+str(hiscore), True, pygame.Color('white')), [170, 35])

			screen.blit(self.font.render("1 PLAYER", True, pygame.Color('white')), [165, 250])
			screen.blit(self.font.render("2 PLAYERS", True, pygame.Color('white')), [165, 275])

			screen.blit(self.font.render("(c) 1980 1985 NAMCO LTD.", True, pygame.Color('white')), [50, 350])
			screen.blit(self.font.render("ALL RIGHTS RESERVED", True, pygame.Color('white')), [85, 380])


		if self.nr_of_players == 1:
			screen.blit(self.player_image, [125, 245])
		elif self.nr_of_players == 2:
			screen.blit(self.player_image, [125, 270])

		self.writeInBricks("battle", [65, 80])
		self.writeInBricks("city", [129, 160])

		if put_on_surface:
			pygame.display.flip()

	def animateIntroScreen(self):
		""" Slide intro (menu) screen from bottom to top
		If Enter key is pressed, finish animation immediately
		@return None
		"""

		global screen

		self.drawIntroScreen(False)
		screen_cp = screen.copy()

		screen.fill([0, 0, 0])

		y = 416
		while (y > 0):
			time_passed = self.clock.tick(50)
			for event in pygame.event.get():
				if event.type == pygame.KEYDOWN:
					if event.key == pygame.K_RETURN:
						y = 0
						break

			screen.blit(screen_cp, [0, y])
			pygame.display.flip()
			y -= 5

		screen.blit(screen_cp, [0, 0])
		pygame.display.flip()


	def chunks(self, l, n):
		""" Split text string in chunks of specified size
		@param string l Input string
		@param int n Size (number of characters) of each chunk
		@return list
		"""
		return [l[i:i+n] for i in range(0, len(l), n)]

	def writeInBricks(self, text, pos):
		""" Write specified text in "brick font"
		Only those letters are available that form words "Battle City" and "Game Over"
		Both lowercase and uppercase are valid input, but output is always uppercase
		Each letter consists of 7x7 bricks which is converted into 49 character long string
		of 1's and 0's which in turn is then converted into hex to save some bytes
		@return None
		"""

		global screen, sprites

		bricks = sprites.subsurface(56*2, 64*2, 8*2, 8*2)
		brick1 = bricks.subsurface((0, 0, 8, 8))
		brick2 = bricks.subsurface((8, 0, 8, 8))
		brick3 = bricks.subsurface((8, 8, 8, 8))
		brick4 = bricks.subsurface((0, 8, 8, 8))

		alphabet = {
			"a" : "0071b63c7ff1e3",
			"b" : "01fb1e3fd8f1fe",
			"c" : "00799e0c18199e",
			"e" : "01fb060f98307e",
			"g" : "007d860cf8d99f",
			"i" : "01f8c183060c7e",
			"l" : "0183060c18307e",
			"m" : "018fbffffaf1e3",
			"o" : "00fb1e3c78f1be",
			"r" : "01fb1e3cff3767",
			"t" : "01f8c183060c18",
			"v" : "018f1e3eef8e08",
			"y" : "019b3667860c18"
		}

		abs_x, abs_y = pos

		for letter in text.lower():

			binstr = ""
			for h in self.chunks(alphabet[letter], 2):
				binstr += str(bin(int(h, 16)))[2:].rjust(8, "0")
			binstr = binstr[7:]

			x, y = 0, 0
			letter_w = 0
			surf_letter = pygame.Surface((56, 56))
			for j, row in enumerate(self.chunks(binstr, 7)):
				for i, bit in enumerate(row):
					if bit == "1":
						if i%2 == 0 and j%2 == 0:
							surf_letter.blit(brick1, [x, y])
						elif i%2 == 1 and j%2 == 0:
							surf_letter.blit(brick2, [x, y])
						elif i%2 == 1 and j%2 == 1:
							surf_letter.blit(brick3, [x, y])
						elif i%2 == 0 and j%2 == 1:
							surf_letter.blit(brick4, [x, y])
						if x > letter_w:
							letter_w = x
					x += 8
				x = 0
				y += 8
			screen.blit(surf_letter, [abs_x, abs_y])
			abs_x += letter_w + 16

	def toggleEnemyFreeze(self, freeze = True):
		""" Freeze/defreeze all enemies """

		global enemies

		for enemy in enemies:
			enemy.paused = freeze
		self.timefreeze = freeze


	def loadHiscore(self):
		""" Load hiscore
		Really primitive version =] If for some reason hiscore cannot be loaded, return 20000
		@return int
		"""
		filename = ".hiscore"
		if (not os.path.isfile(filename)):
			return 20000

		f = open(filename, "r")
		hiscore = int(f.read())

		if hiscore > 19999 and hiscore < 1000000:
			return hiscore
		else:
			
			return 20000

	def saveHiscore(self, hiscore):
		""" Save hiscore
		@return boolean
		"""
		try:
			f = open(".hiscore", "w")
		except:
			
			return False
		f.write(str(hiscore))
		f.close()
		return True


	def finishLevel(self):
		""" Finish current level
		Show earned scores and advance to the next stage
		"""

		global play_sounds, sounds

		for player in players:
			player.lives = 3 
			
		if play_sounds:
			sounds["bg"].stop()

		self.active = False
		if self.game_over:
			game.showMenu()
		else:
			self.nextLevel()		
		print("Stage "+str(self.stage)+" completed")

	def nextLevel(self):
		""" Start next level """

		global castle, players, bullets, bonuses, play_sounds, sounds, screen_array, screen_array_grayscale

		del bullets[:]
		del enemies[:]
		del bonuses[:]
		castle.rebuild()
		del gtimer.timers[:]

		
		self.stage = random.randint(1, 2)
		self.level = Level(self.stage)
		self.timefreeze = False
 
		
		
		self.level.enemies_left = [0] * 20  

		if play_sounds:
			sounds["start"].play()
			gtimer.add(4330, lambda :sounds["bg"].play(-1), 1)

		
		self.available_positions = []
		filename = "levels/gameplay/" + str(self.stage)
		f = open(filename, "r")
		data = f.read().split("\n")
		f.close()
		for y in range(len(data) - 1):
			row = data[y]
			for x in range(len(row) - 1): 
				if row[x] == "." and row[x + 1] == "." and data[y + 1][x] == "." and data[y+1][x+1] == "." and (not (x == 12 and y == 24)) and y > 12 and 3 < x < 21:
					self.available_positions.append([x, y])
		random.shuffle(self.available_positions)	
		

		self.reloadPlayers()

		gtimer.add(2500, lambda :self.spawnEnemy()) 

		
		self.game_over = False 

		
		self.running = True

		
		self.active = True

		self.draw() 

		screen_array = pygame.surfarray.array3d(screen)
		screen_array = np.transpose(screen_array, (1, 0, 2))
		screen_array_grayscale = rgb_to_grayscale(screen_array)

		
		self.agent = ai_agent()
		self.p_mapinfo = multiprocessing.Queue()
		self.c_control = multiprocessing.Queue()
  
		mapinfo = self.get_mapinfo()
		self.agent.mapinfo = mapinfo
		if self.p_mapinfo.empty() == True:
			self.p_mapinfo.put(mapinfo)

		self.ai_bot_actions = [0, 4]
		self.p = multiprocessing.Process(target = self.agent.operations, args = (self.p_mapinfo, self.c_control))
		self.p.start()

  
	def get_mapinfo(self):
		global players, bullets
		mapinfo=[]
		mapinfo.append([])
		mapinfo.append([])
		mapinfo.append([])
		mapinfo.append([])
		for bullet in bullets:
			if bullet.owner == bullet.OWNER_ENEMY:
				nrect=bullet.rect.copy()
				mapinfo[0].append([nrect,bullet.direction,bullet.speed])
		for enemy in enemies:
			nrect=enemy.rect.copy()
			mapinfo[1].append([nrect,enemy.direction,enemy.speed,enemy.type])
		for tile in game.level.mapr:
			nrect=pygame.Rect(tile.left, tile.top, 16, 16)
			mapinfo[2].append([nrect,tile.type])
		for player in players:
			nrect=player.rect.copy()
			mapinfo[3].append([nrect,player.direction,player.speed,player.shielded])
		return mapinfo

'''
===============================================================================================================================
														RL TRAINING ENVIRONMENT
===============================================================================================================================
'''


class TanksEnv(gymnasium.Env):
	metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
	#初始化游戏环境
	def __init__(self, render_mode=None):
		global gtimer, sprites, screen, screen_array, screen_array_grayscale, players, enemies, bullets, bonuses, labels, play_sounds, sounds, game, castle
		global obs_flag_castle_danger, obs_flag_stupid, obs_flag_player_collision, obs_flag_hot, obs_flag_bullet_fired
		obs_flag_castle_danger = 0
		obs_flag_stupid = 0
		obs_flag_player_collision = 0
		obs_flag_hot = 0
		obs_flag_bullet_fired = 0
		self.enemy_in_line = 4
		self.width = 208   
		self.height = 208  
		self.paso = 0

		
		self.heat_map = np.zeros((13, 13))
		self.grid_size = 32
		self.grid_position = [0, 0, 0, 0, 0, 0, 0]
		self.heat_decay_rate = 0.02  
		self.heat_base_penalty = 0.01 

		
		self.enemy_positions = np.full((4,7), 0)
		self.bullet_positions = np.full((6,7), 0)

		self.frame_stack = deque(maxlen=4)
		empty_frame = np.zeros((self.width, self.height), dtype=np.uint8)
		for _ in range(4):
			self.frame_stack.append(empty_frame)

		self.bullet_avoidance_dir = 4

		self.observation_space = gymnasium.spaces.Dict(
			{
				"obs_frames": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(4, self.width, self.height), dtype=np.float64),

				"player_position": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float64),  
				"enemy1_position": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float64),
				"enemy2_position": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float64),
				"enemy3_position": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float64),
				"enemy4_position": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float64),

				"bullet1_position": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float64),
				"bullet2_position": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float64),
				"bullet3_position": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float64),
				"bullet4_position": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float64),
				"bullet5_position": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float64),
				"bullet6_position": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float64),

				"prev_action": gymnasium.spaces.MultiDiscrete([2, 5]), 
				"ai_bot_actions": gymnasium.spaces.MultiDiscrete([2, 5]), 
				"flags": gymnasium.spaces.MultiBinary(5),
				"enemies_left": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64),
				"heatmap": gymnasium.spaces.Box(low=0.0, high=1.0, shape=(13, 13), dtype=np.float64),
				 

				

			}
		)

		
		self.action_space = gymnasium.spaces.MultiDiscrete([2, 5])

		
		gtimer = Timer()

		sprites = None
		screen = None
		screen_array = None
		screen_array_grayscale = empty_frame
		players = []
		enemies = []
		bullets = []
		bonuses = []
		labels = []

		play_sounds = False
		sounds = {}

		game = Game()
		castle = Castle()
		game.showMenu()
	#获取全部所需游戏状态
	def _get_obs(self):
		global gtimer, sprites, screen, screen_array, screen_array_grayscale, players, enemies, bullets, bonuses, labels, play_sounds, sounds, game, castle
		global obs_flag_castle_danger, obs_flag_stupid, obs_flag_player_collision, obs_flag_hot, obs_flag_bullet_fired



		return {
			"obs_frames": np.array(self.obs_frames) / 255.0,

			"player_position": np.array(self.grid_position) / np.array([29*16, 29*16, 4, 4, 59*16, 59*16, 4]),
			"enemy1_position": np.array(self.enemy_positions[0]) / np.array([29*16, 29*16, 4, 1, 59*16, 59*16, 4]),
			"enemy2_position": np.array(self.enemy_positions[1]) / np.array([29*16, 29*16, 4, 1, 59*16, 59*16, 4]),
			"enemy3_position": np.array(self.enemy_positions[2]) / np.array([29*16, 29*16, 4, 1, 59*16, 59*16, 4]),
			"enemy4_position": np.array(self.enemy_positions[3]) / np.array([29*16, 29*16, 4, 1, 59*16, 59*16, 4]),

			"bullet1_position": np.array(self.bullet_positions[0]) / np.array([29*16, 29*16, 4, 1, 59*16, 59*16, 4]),
			"bullet2_position": np.array(self.bullet_positions[1]) / np.array([29*16, 29*16, 4, 1, 59*16, 59*16, 4]),
			"bullet3_position": np.array(self.bullet_positions[2]) / np.array([29*16, 29*16, 4, 1, 59*16, 59*16, 4]),
			"bullet4_position": np.array(self.bullet_positions[3]) / np.array([29*16, 29*16, 4, 1, 59*16, 59*16, 4]),
			"bullet5_position": np.array(self.bullet_positions[4]) / np.array([29*16, 29*16, 4, 1, 59*16, 59*16, 4]),
			"bullet6_position": np.array(self.bullet_positions[5]) / np.array([29*16, 29*16, 4, 1, 59*16, 59*16, 4]),

			"ai_bot_actions": np.array(game.ai_bot_actions),
			"prev_action": np.array(self.prev_action),

			"flags": np.array([obs_flag_castle_danger, obs_flag_stupid, obs_flag_player_collision, obs_flag_hot, obs_flag_bullet_fired]),
			"enemies_left": np.array([len(enemies) / 20]),
			"heatmap": np.array(self.heat_map) / 25,
			
		}
	#获取游戏信息
	def _get_info(self):
		return {"Info": 0}
	#终止AI进程
	def kill_ai_process(self, p):
		os.kill(p.pid, 9)
	#清空进程间通信队列
	def clear_queue(self, queue):
		
		while not queue.empty():
			try:
				queue.get(False)
			except Empty:  
				break  
	#重置游戏环境到初始状态
	def reset(self, seed=None, options=None):
		global gtimer, sprites, screen, screen_array, screen_array_grayscale, players, enemies, bullets, bonuses, labels, play_sounds, sounds, game, castle
		global obs_flag_castle_danger, obs_flag_stupid, obs_flag_player_collision, obs_flag_hot, obs_flag_bullet_fired
		
		
		
		self.reward = 0
		self.paso = 0
		players[0].lives = 3
		self.prev_action = np.array([0, 0])
		

		self.bullet_avoidance_dir = 4
		obs_flag_castle_danger = 0
		obs_flag_stupid = 0
		obs_flag_player_collision = 0
		obs_flag_hot = 0
		obs_flag_bullet_fired = 0

		
		empty_frame = np.zeros((self.width, self.height), dtype=np.uint8)
		
		self.frame_stack.clear()  
		for _ in range(4):
			self.frame_stack.append(empty_frame)
			
		self.obs_frames = np.stack([empty_frame, empty_frame, empty_frame, empty_frame], axis=0)
		self.heat_map = np.zeros((13, 13))

	


		for i in range(4):
			self.enemy_positions[i] = [29*16, 29*16, 4, 0, 59*16, 59*16, 4] 

		for i in range(6):
			self.bullet_positions[i] = [29*16, 29*16, 4, 1, 59*16, 59*16, 4] 

		
		game.nextLevel()
		
		game.ai_bot_actions = [0 if x is None else x for x in game.ai_bot_actions]
		observation = self._get_obs()
		info = self._get_info()

		return observation, info
	#执行一个动作并返回环境状态
	def step(self, action):
		global gtimer, sprites, screen, screen_array, screen_array_grayscale, players, enemies, bullets, bonuses, labels, play_sounds, sounds, game, castle
		global obs_flag_stupid, obs_flag_player_collision, obs_flag_hot, obs_flag_castle_danger, obs_flag_bullet_fired
		self.reward = 0
		
		obs_flag_stupid = 0
		obs_flag_player_collision = 0
		obs_flag_castle_danger = 0
		obs_flag_bullet_fired = 0
		self.bullet_avoidance_dir = 4
		time_passed = 20
		self.paso += 1
		


		for i, enemy in enumerate(enemies[:4]):
			if enemy.state != enemy.STATE_DEAD:
				
				grid_x, grid_y = enemy.rect.centerx, enemy.rect.centery
				direction = enemy.direction
				status = 1  
				distance_to_castle = Vmanhattan_distance(enemy.rect.topleft, castle.rect.topleft)
				distance_to_player = Vmanhattan_distance(enemy.rect.topleft, players[0].rect.topleft)
				in_line_status = Vinline_with_enemy(players[0].rect, enemy.rect)
			else:
				grid_x, grid_y, direction, status, distance_to_castle, distance_to_player, in_line_status = 29*16, 29*16, 4, 0, 59*16, 59*16, 4  

			self.enemy_positions[i] = [grid_x, grid_y, direction, status, distance_to_castle, distance_to_player, in_line_status]

		
		if len(enemies) < 4:
			for i in range(len(enemies), 4):
				self.enemy_positions[i] = [29*16, 29*16, 4, 0, 59*16, 59*16, 4]  

		for i, bullet in enumerate(bullets[:6]):
			if bullet.state != bullet.STATE_REMOVED:
				
				grid_x, grid_y = bullet.rect.centerx, bullet.rect.centery
				direction = bullet.direction
				owner = bullet.owner
				distance_to_castle = Vmanhattan_distance(bullet.rect.topleft, castle.rect.topleft)
				distance_to_player = Vmanhattan_distance(bullet.rect.topleft, players[0].rect.topleft)
				in_line_status = Vinline_with_enemy(players[0].rect, bullet.rect)
			else:
				grid_x, grid_y, direction, owner, distance_to_castle, distance_to_player, in_line_status = 29*16, 29*16, 4, 0, 59*16, 59*16, 4  

			self.bullet_positions[i] = [grid_x, grid_y, direction, owner, distance_to_castle, distance_to_player, in_line_status]

		
		if len(bullets) < 6:
			for i in range(len(bullets), 4):
				self.bullet_positions[i] = [29*16, 29*16, 4, 0, 59*16, 59*16, 4]  


		
		smallest_distance = 59*16

		for enemy in self.enemy_positions:
			status = enemy[3]
			distance_to_castle = enemy[4]
			distance_to_player = enemy[5]

			
			if status == 1 and distance_to_player < smallest_distance:
				smallest_distance = distance_to_player
			if status == 1 and distance_to_castle < 10*16:
				obs_flag_castle_danger = 1


		if len(bullets) != 0:
			bullets_info=[]
			bullets_info.append([])
			for bullet in bullets:
				if bullet.owner == bullet.OWNER_ENEMY:
					nrect=bullet.rect.copy()
					bullets_info[0].append([nrect,bullet.direction,bullet.speed])
			self.bullet_avoidance_dir = Vbullet_avoidance(players[0].rect, bullets_info[0])


		
		mapinfo = game.get_mapinfo()
		if game.p_mapinfo.empty() == True:
			game.p_mapinfo.put(mapinfo)
		if game.c_control.empty() != True:
			try:
				game.ai_bot_actions = game.c_control.get(False)
			except queue.empty:
				skip_this = True

		
		DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT = range(4)
		pygame.event.pump()
		game.ai_bot_actions = [0.0 if x is None else x for x in game.ai_bot_actions]


		for player in players:
			if player.state == player.STATE_ALIVE and not game.game_over and game.active:
				if action[0] == 1 and not antiStupidBlock(player.direction, player.rect, castle.rect):
					player.fire()

				if action[0] == 1 and antiStupidBlock(player.direction, player.rect, castle.rect):
					self.reward -= 0.1

				if action[1] == 0:
					player.move(game.DIR_UP)
					if self.prev_action[1] == 0 and obs_flag_player_collision == 0:
						self.reward += 0.05
						pass

				if action[1] == 1:
					player.move(game.DIR_RIGHT)
					if self.prev_action[1] == 1 and obs_flag_player_collision == 0:
						self.reward += 0.05
						pass

				if action[1] == 2:
					player.move(game.DIR_DOWN)
					if self.prev_action[1] == 2 and obs_flag_player_collision == 0:
						self.reward += 0.05
						pass

				if action[1] == 3:
					player.move(game.DIR_LEFT)
					if self.prev_action[1] == 3 and obs_flag_player_collision == 0:
						self.reward += 0.05
						pass
					
			self.prev_action = action
			if action[0] == game.ai_bot_actions[0] and action[0] == 1:
				self.reward += 0.2
				pass

			if action[1] == game.ai_bot_actions[1] and action[1] != 4:
				self.reward += 0.1
				pass

			
			distance_to_castle = Vmanhattan_distance(player.rect.topleft, castle.rect.topleft)
			self.grid_position = (player.rect.centerx, player.rect.centery, player.direction, player.lives, distance_to_castle, smallest_distance, self.bullet_avoidance_dir)
			
			
			if self.heat_map[round(self.grid_position[0]//self.grid_size), round(self.grid_position[1]//self.grid_size)] < 25:
				self.heat_map[round(self.grid_position[0]//self.grid_size), round(self.grid_position[1]//self.grid_size)] += 0.5
			
			if self.heat_map[round(self.grid_position[0]//self.grid_size), round(self.grid_position[1]//self.grid_size)]	> 9:
				obs_flag_hot = 1
			else:
				obs_flag_hot = 0

			
			self.reward -= self.heat_base_penalty * (1.22 ** self.heat_map[round(self.grid_position[0]//self.grid_size), round(self.grid_position[1]//self.grid_size)])

			
			self.heat_map *= (1 - self.heat_decay_rate)
			

			player.update(time_passed)

		for enemy in enemies:

			if enemy.state == enemy.STATE_DEAD and not game.game_over and game.active:
				self.reward += 5 
				if enemy.rect.y > 208:
					self.reward += 10 
				
				
				enemies.remove(enemy)

				if len(game.level.enemies_left) == 0 and len(enemies) == 0:
					self.reward += 50 
					print("You killed all enemy tanks! :)")
					self.kill_ai_process(game.p)
					self.clear_queue(game.p_mapinfo)
					self.clear_queue(game.c_control)
					game.game_over = 1
			else:
				enemy.update(time_passed)

		if not game.game_over and game.active:
			for player in players:
				if player.state == player.STATE_ALIVE:
					if player.bonus != None and player.side == player.SIDE_PLAYER:
						game.triggerBonus(player.bonus, player)
						self.reward += 1 
						player.bonus = None
				elif player.state == player.STATE_DEAD:
					self.reward -= 5 
					
					game.superpowers = 0
					player.lives -= 1
					if player.lives > 0:
						game.respawnPlayer(player)
					else:
						player.lives = 0
						self.reward -= 15
						print("You died! :(")
						self.kill_ai_process(game.p)
						self.clear_queue(game.p_mapinfo)
						self.clear_queue(game.c_control)
						game.game_over = 1

		for bullet in bullets:

			if bullet.state == bullet.STATE_REMOVED:
				bullets.remove(bullet)
			else:
				bullet.update()
				if bullet.state == bullet.STATE_REMOVED:
					bullets.remove(bullet)
				else:
					bullet.update()
					if bullet.state == bullet.STATE_REMOVED:
						bullets.remove(bullet)
					else:
						bullet.update()
						if bullet.state == bullet.STATE_REMOVED:
							bullets.remove(bullet)
						else:
							bullet.update()
							if bullet.state == bullet.STATE_REMOVED:
								bullets.remove(bullet)
							else:
								bullet.update()
		for bullet in bullets:
			if bullet.owner == 0 and bullet.state == bullet.STATE_ACTIVE:
					obs_flag_bullet_fired = 1	
			if bullet.owner == Bullet.OWNER_PLAYER:  
				
				if bullet.direction == DIR_DOWN and bullet.rect.bottom < castle.rect.top and bullet.rect.left <= castle.rect.right and bullet.rect.right >= castle.rect.left:
					obs_flag_stupid = 1
				if bullet.direction == DIR_UP and bullet.rect.top > castle.rect.bottom and bullet.rect.left <= castle.rect.right and bullet.rect.right >= castle.rect.left:
					obs_flag_stupid = 1
				if bullet.direction == DIR_RIGHT and bullet.rect.right < castle.rect.left and bullet.rect.top <= castle.rect.bottom and bullet.rect.bottom >= castle.rect.top:
					obs_flag_stupid = 1
				if bullet.direction == DIR_LEFT and bullet.rect.left > castle.rect.right and bullet.rect.top <= castle.rect.bottom and bullet.rect.bottom >= castle.rect.top:
					obs_flag_stupid = 1	
		if obs_flag_stupid == 1:
			self.reward -= 0.1
			pass
		for bonus in bonuses:
			if bonus.active == False:
				bonuses.remove(bonus)

		for label in labels:
			if not label.active:
				labels.remove(label)

		if not game.game_over:
			if not castle.active:
				self.reward -= 50 
				print("Castle not active!")
				self.kill_ai_process(game.p)
				self.clear_queue(game.p_mapinfo)
				self.clear_queue(game.c_control)
				game.game_over = 1
		gtimer.update(time_passed)	
		game.draw() 

		
		screen_array = pygame.surfarray.array3d(screen)
		screen_array = np.transpose(screen_array, (1, 0, 2))
		screen_array_grayscale = rgb_to_grayscale(screen_array)

		
		self.frame_stack.append(screen_array_grayscale)
		
		self.obs_frames = np.stack([screen_array_grayscale, self.frame_stack[-2], self.frame_stack[-3], self.frame_stack[-4]], axis=0)
		
		observation = self._get_obs()

		
		reward = self.reward / 10
		terminated = game.game_over
		truncated = False
		
		
		info = self._get_info()
		
		return observation, reward, terminated, truncated, info		
	#渲染环境
	def render(self):
		pass
	#渲染单帧
	def _render_frame(self):
		pass
	#关闭环境
	def close(self):
		pass