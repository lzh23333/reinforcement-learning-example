#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   gui.py
@Time    :   2021/01/10 16:01:57
@Author  :   lzh
@Version :   1.0
@Contact :   robinlin163@163.com
@Desc    :   GUI for cat catch mouse
'''
import time
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from utils import index2pos


class BoardGUI(object):

    def __init__(self, board, init_state, state_history, ms=0.5):
        self.board = board
        self.state_history = state_history
        self.cat = index2pos(init_state[0], board.shape)
        self.mouse = index2pos(init_state[1], board.shape)

        self.root = Tk()
        self.root.title("Cat Catch Mouse")

        # plot board
        square = 800 // max(board.shape)
        self.square = square
        self.canvas = Canvas(self.root, width=800,
                             height=820, background="white")
        self.canvas.pack(side="top", fill="both", anchor="c", expand=True)
        self.ms = ms
        colors = ["white", "gray", "red", "blue"]
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                x = i * square
                y = j * square
                self.canvas.create_rectangle(x, y, x + square, y + square,
                                             fill=colors[int(board[i, j])],
                                             )
        image_size = (square, square)

        # cat mouse images
        self.cat_img = Image.open(
            "./imgs/cat.jpeg").resize(image_size, Image.ANTIALIAS)
        self.cat_img = ImageTk.PhotoImage(self.cat_img)
        self.cat_canvas = self.canvas.create_image(
            self.cat[0] * square, self.cat[1] * square, image=self.cat_img,
            anchor=NW
        )
        self.mouse_img = Image.open(
            "./imgs/mouse.jpeg").resize(image_size, Image.ANTIALIAS)
        self.mouse_img = ImageTk.PhotoImage(self.mouse_img)
        self.mouse_canvas = self.canvas.create_image(
            self.mouse[0] * square, self.mouse[1] * square, image=self.mouse_img,
            anchor=NW
        )
        self.catch = ImageTk.PhotoImage(
            Image.open("./imgs/catch.jpg").resize(image_size, Image.ANTIALIAS)
        )

        # add a button to control animation
        self.button = Button(self.root, text="start", command=self.run)
        self.button.pack(side=BOTTOM)

        # 
        # time.sleep(1)
        # self.root.after(0, self.run)
        self.root.mainloop()

    def run(self):
        if hasattr(self, "catch_canvas"):
            self.canvas.delete(self.catch_canvas)
        for i in range(len(self.state_history)):
            self.display(i)
            time.sleep(self.ms)

    def display(self, i):
        # display one timestamp
        state = self.state_history[i]
        cat_pos = index2pos(state[0], self.board.shape)
        mouse_pos = index2pos(state[1], self.board.shape)
        self.canvas.delete(self.cat_canvas)
        self.canvas.delete(self.mouse_canvas)
        if state[0] != state[1]:

            self.cat_canvas = self.canvas.create_image(
                cat_pos[0] * self.square, cat_pos[1] * self.square,
                image=self.cat_img,
                anchor=NW
            )
            self.mouse_canvas = self.canvas.create_image(
                mouse_pos[0] * self.square, mouse_pos[1] * self.square,
                image=self.mouse_img,
                anchor=NW
            )
        else:
            self.catch_canvas = self.canvas.create_image(
                cat_pos[0] * self.square,
                cat_pos[1] * self.square,
                image=self.catch,
                anchor=NW
            )
        self.canvas.update()
