#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################
# Robot Controller Script              #
# Copyright (c) Takuya Tsukahara, 2019 #
########################################

import argparse
import cv2
import logging
from flask import Flask, render_template, Response
from utils import get_visual, get_visual_m2det
# import RPi.GPIO as GPIO

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/visual_feed')
def visual_feed():
    if args.m2det == True:
        return Response(get_visual_m2det(capture, args.interval), \
        mimetype='multipart/x-mixed-replace; boundary=boundary')
    else:
        return Response(get_visual(capture, args.interval), \
        mimetype='multipart/x-mixed-replace; boundary=boundary')

@app.route('/forward')
def forward():
    '''
    GPIO.output(m11, 1)
    GPIO.output(m12, 0)
    GPIO.output(m21, 1)
    GPIO.output(m22, 0)
    '''
    print('FORWARD')
    return 'forward'

@app.route('/left')
def left():
    '''
    GPIO.output(m11, 0)
    GPIO.output(m12, 0)
    GPIO.output(m21, 1)
    GPIO.output(m22, 1)
    '''
    print('LEFT')
    return 'left'

@app.route('/auto')
def auto():
    print('AUTO')
    return 'auto'

@app.route('/right')
def right():
    '''
    GPIO.output(m11, 1)
    GPIO.output(m12, 1)
    GPIO.output(m21, 0)
    GPIO.output(m22, 0)
    '''
    print('RIGHT')
    return 'right'

@app.route('/back')
def back():
    '''
    GPIO.output(m11, 0)
    GPIO.output(m12, 1)
    GPIO.output(m21, 0)
    GPIO.output(m22, 1)
    '''
    print('BACK')
    return 'back'

@app.route('/stop')
def stop():
    '''
    GPIO.output(m11, 1)
    GPIO.output(m12, 1)
    GPIO.output(m21, 1)
    GPIO.output(m22, 1)
    '''
    print('STOP')
    return 'stop'

def set_moter(m11, m12, m21, m22):
    '''
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(m11, GPIO.OUT)
    GPIO.setup(m12, GPIO.OUT)
    GPIO.setup(m21, GPIO.OUT)
    GPIO.setup(m22, GPIO.OUT)
    GPIO.output(m11, 0)
    GPIO.output(m12, 0)
    GPIO.output(m21, 0)
    GPIO.output(m22, 0)
    '''
    print('==> Motor setting complete\n')
    return m11, m12, m21, m22

def set_camera(webcamid):
    capture = cv2.VideoCapture(webcamid)
    print('==> Camera setup is complete\n')
    return capture


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Robot Controller')
    parser.add_argument('--camera', '-c', default=0, type=int,
                        metavar='<number>', help='Camera id to use')
    parser.add_argument('--interval', '-i', default=0, type=int,
                        metavar='<number>', help='Image save interval')
    parser.add_argument('--m2det', '-m', action='store_true',
                        help='Use M2Det')
    args = parser.parse_args()
    m11, m12, m21, m22 = set_moter(18, 23, 24, 25) # モータの設定
    capture = set_camera(args.camera) # カメラの設定
    # logging.getLogger('werkzeug').disabled = True
    # app.debug = True
    app.threaded = True
    # app.run()
    app.run(host='0.0.0.0', port=9999)
