import pygame
import sys

from tools import channel_01

pygame.init()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BROWN = (165, 42, 42)
TRANSPARENT_BLACK = (0, 0, 0, 20)  
TRANSPARENT_WHITE = (255, 255, 255, 20) 
BACKGROUND_COLOR = (173, 216, 230) 

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 650
GRID_SIZE = 18
GRID_WIDTH = 600 // GRID_SIZE
GRID_HEIGHT = 600 // GRID_SIZE
LEFT_TOP = 20

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("圍棋")


def draw_board(board):
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            rect = pygame.Rect(LEFT_TOP + x * GRID_WIDTH, LEFT_TOP + y * GRID_HEIGHT,
                                GRID_WIDTH, GRID_HEIGHT)
            pygame.draw.rect(screen, BROWN, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)  
    for i in range(GRID_SIZE + 1):
        for j in range(GRID_SIZE + 1):
            if board[0][0][i][j]:
                pygame.draw.circle(screen, TRANSPARENT_WHITE,
                                   (LEFT_TOP + j * GRID_WIDTH, LEFT_TOP + i * GRID_HEIGHT), radius)
            if board[0][1][i][j]:
                pygame.draw.circle(screen, TRANSPARENT_BLACK,
                                   (LEFT_TOP + j * GRID_WIDTH, LEFT_TOP + i * GRID_HEIGHT), radius)



def get_board_position(mouse_pos):
    grid_y = (mouse_pos[0] - LEFT_TOP + GRID_WIDTH // 2) // GRID_WIDTH
    grid_x = (mouse_pos[1] - LEFT_TOP + GRID_WIDTH // 2) // GRID_HEIGHT 
    return grid_x, grid_y


running = True
turn = 1
board = [[[[0]*(GRID_SIZE + 1) for _ in range(GRID_SIZE + 1)] for _ in range(2)]]
while running:
    for event in pygame.event.get():
        mouse_pos = pygame.mouse.get_pos()
        grid_x, grid_y = get_board_position(mouse_pos)
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if grid_x >= 0 and grid_y >= 0 and grid_x <= GRID_SIZE and grid_y <= GRID_SIZE:
                channel_01(board, 0, grid_x, grid_y, turn)
                turn = 0 if turn else 1
                print((grid_x, grid_y))
                
    screen.fill(BACKGROUND_COLOR)

    draw_board(board)

    if grid_x >= 0 and grid_y >= 0 and grid_x <= GRID_SIZE and grid_y <= GRID_SIZE:
        radius = 10
        if turn:
            pygame.draw.circle(screen, TRANSPARENT_BLACK, mouse_pos, radius)
        else:
            pygame.draw.circle(screen, TRANSPARENT_WHITE, mouse_pos, radius)


    pygame.display.flip()

pygame.quit()
sys.exit()
