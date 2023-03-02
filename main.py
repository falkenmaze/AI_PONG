from pong import Game 
import pygame
import neat
import time
import os
import pickle
pygame.init()

class AI:
	def __init__(self,window,width,height):
		self.game = Game(window, width,height)
		self.ball = self.game.ball
		self.left_paddle = self.game.left_paddle
		self.right_paddle = self.game.right_paddle
	def test_ai(self,genome,net):
		clock = pygame.time.Clock()
		run = True 
		while run:
			clock.tick(60)
			game_info = self.game.loop()
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					run = False 
					break
			output = net.activate((self.right_paddle.y, abs(self.right_paddle.x - self.ball.x), self.ball.y))
			decision = output.index(max(output))
			if decision == 0:
				pass
			elif decision == 1:
				self.game.move_paddle(left=False,up=True)
			elif decision == 2:
				self.game.move_paddle(left=False, up=False)
			keys = pygame.key.get_pressed()
			if keys[pygame.K_w]:
				self.game.move_paddle(left=True, up=True)
			elif keys[pygame.K_s]:
				self.game.move_paddle(left=True, up=False)
			self.game.draw(draw_score = True)
			pygame.display.update()



	def train_ai(self,genome1,genome2,config, draw=False):
		run = True
		start_time = time.time()
		nn1 = neat.nn.FeedForwardNetwork.create(genome1,config)
		nn2 = neat.nn.FeedForwardNetwork.create(genome2,config)
		self.genome1 = genome1 
		self.genome2 = genome2 
		max_hits = 50 
		while run:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					return True 
			game_info = self.game.loop()
			self.move_ai_paddles(nn1,nn2)
			if draw:
				self.game.draw(draw_score = True, draw_hits = False)
			pygame.display.update()
			duration = time.time() - start_time
			if game_info.left_score == 1 or game_info.right_score == 1 or game_info.left_hits >= max_hits:
				self.calc_fitness(game_info,duration)
				break
		return False

	def move_ai_paddles(self,nn1,nn2):
		playas = [(self.genome1, nn1, self.left_paddle, True), (self.genome2, nn2, self.right_paddle, False)]
		for (genome, nn, paddle, left) in playas:
			output = nn.activate((paddle.y, abs(paddle.x - self.ball.x), self.ball.y))
			decision = output.index(max(output))
			valid = True 
			if decision == 0:
				genome.fitness -= 0.1
			elif decision == 1:
				valid = self.game.move_paddle(left=left, up=True)
			elif decision == 2:
				valid = self.game.move_paddle(left=left, up=False)
			if not valid:
				genome.fitness -= 1 

	def calc_fitness(self,game_info,duration):
		self.genome1.fitness += game_info.left_hits + duration
		self.genome2.fitness += game_info.right_hits + duration


def eval_genomes(genomes,config):
	width,height = 700,500
	window = pygame.display.set_mode((width,height))
	for i, (genomeid1,genome1) in enumerate(genomes):
		genome1.fitness = 0 
		for (genomeid2, genome2) in genomes[min(i+1, len(genomes) - 1):]:
			genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
			game = AI(window,width,height)
			Quit = game.train_ai(genome1,genome2,config, draw=True)
			if Quit:
				quit()

def run(config):
	p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-9')
	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)
	p.add_reporter(neat.Checkpointer(1))
	winner = p.run(eval_genomes, 5)
	with open("best.pickle", "wb") as best:
		pickle.dump(winner,best)

def best_network(config_path):
	with open("best.pickle", "rb") as best:
		winner = pickle.load(best)
	winner_nn = neat.nn.FeedForwardNetwork.create(winner,config_path)
	width,height = 700,500
	window = pygame.display.set_mode((700,500))
	game = AI(window,width,height)
	game.test_ai(winner,winner_nn)

if __name__ == "__main__":
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, "config.txt")
	config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,neat.DefaultStagnation, config_path)
	run(config)
	#best_network(config)
