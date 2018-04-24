# -*- coding: utf-8 -*-  

"""
Created by Wang Han on 2018/4/19 15:43.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2017 Wang Han. SCU. All Rights Reserved.
"""
import sys


class Logger(object):
  def __init__(self, logfile):
    self.terminal = sys.stdout
    self.log = open(logfile, "a")

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)

  def flush(self):
    # this flush method is needed for python 3 compatibility.
    # this handles the flush command by doing nothing.
    # you might want to specify some extra behavior here.
    pass
