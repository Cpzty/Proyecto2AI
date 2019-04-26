# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:48:29 2019

@author: Cristian
"""
import pygame
from random_word import RandomWords

def text_objects(text,font):
    textSurface = font.render(text,True,black)
    return textSurface, textSurface.get_rect()

pygame.init()
screen = pygame.display.set_mode([800,600])
white = [255,255,255]
black = [0,0,0]
red = [200,0,0]
green = [0,200,0]
bright_red = [255,0,0]
bright_green = [0,255,0]


radius = 10

screen.fill(white)
smallText = pygame.font.Font("freesansbold.ttf",20)
textSurf, textRect = text_objects("Clear",smallText)
textRect.center = ( 75,50 )
pygame.draw.rect(screen,red,(0,0,150,100))
pygame.draw.rect(screen,green,(151,0,150,100))
screen.blit(textSurf,textRect)
textSurf, textRect = text_objects("Save",smallText)
textRect.center = ( 225,50 )
screen.blit(textSurf,textRect)

pygame.display.update()
pygame.display.set_caption("Click to draw")

keep_going = True
should_draw = False
#Loop
while(keep_going):
    mouse = pygame.mouse.get_pos()
    #print(mouse)
    
    for event in pygame.event.get():
        if(event.type == pygame.QUIT):
            keep_going = False
        
        if(event.type == pygame.MOUSEMOTION):
            if(should_draw and mouse[1]>120):
                spot = event.pos
                pygame.draw.circle(screen,black,spot,radius)
                pygame.display.update()
        if(event.type == pygame.MOUSEBUTTONDOWN):
            should_draw = True
            if(mouse[0]<150 and mouse[1]<100):
                screen.fill(white)
                smallText = pygame.font.Font("freesansbold.ttf",20)
                textSurf, textRect = text_objects("Clear",smallText)
                textRect.center = ( 75,50 )
                pygame.draw.rect(screen,red,(0,0,150,100))
                pygame.draw.rect(screen,green,(151,0,150,100))
                screen.blit(textSurf,textRect)
                textSurf, textRect = text_objects("Save",smallText)
                textRect.center = ( 225,50 )
                screen.blit(textSurf,textRect)
                pygame.display.update()
            elif(mouse[0]>151 and mouse[0]<300 and mouse[1]<100):
                r = RandomWords()
                save_rect = pygame.Rect(0,100,500,500)
                sub = screen.subsurface(save_rect)
                pygame.image.save(sub, "{}.jpg".format(r.get_random_word()))
        if(event.type == pygame.MOUSEBUTTONUP):
            should_draw = False
pygame.quit()
