#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@DOC : logger 
@Date ：2023/8/11 11:13 
"""
import logging

# 创建日志对象
logger = logging.getLogger()

# 设置日志级别为DEBUG
logger.setLevel(logging.DEBUG)

# 创建日志处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# 创建日志文件处理器
file_handler = logging.FileHandler('log.txt')
file_handler.setLevel(logging.DEBUG)
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s'
# 创建日志格式
formatter = logging.Formatter(LOG_FORMAT)

# 将日志格式应用到处理器中
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 添加处理器到日志对象
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 记录日志
# logger.debug('Debug message')
# logger.info('Info message')
# logger.warning('Warning message')
# logger.error('Error message')
# logger.critical('Critical message')
