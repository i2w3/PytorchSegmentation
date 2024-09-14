import time
import logging
from pathlib import Path
from typing import Optional

class Logger:
    def __init__(self, 
                 LoggerName:str, 
                 LoggingPath:Optional[Path] = None, 
                 ConsoleLogging:bool = True) -> None:
        '''初始化Logger
        
        Args:
            LoggerName(str): Logger的名字
            LoggingPath(Optional[Path]): log日志的保存路径
            ConsoleLogging(bool): 控制台是否输出log，默认为True
        '''
        local_time = time.strftime("%Y-%m-%dT%H%M%S", time.localtime())
        LoggingPath = LoggingPath if LoggingPath is not None else Path(f"./logs/{local_time}/{local_time}.log")
        Path.mkdir(LoggingPath.parent, parents=True, exist_ok=True)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        self.log_path = LoggingPath
        self.logger = logging.getLogger(LoggerName)
        self.logger.setLevel(logging.DEBUG)

        
        fh = logging.FileHandler(self.log_path, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        if ConsoleLogging:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        
        self.logger.info("init logger sucess!")

    def __repr__(self) -> str:
        return f"Logger(log_path={self.log_path})"

    def __call__(self, level:str, note:str) -> None:
        '''快速log

        Args:
            level(str): log的等级
            note(str): log的信息
        '''
        if level == 'debug':
            self.logger.debug(note)
        elif level == 'info':
            self.logger.info(note)
        elif level == 'warning':
            self.logger.warning(note)
        elif level == 'error':
            self.logger.error(note)
        else:
            self.logger.error("didn't use the true level, but log: " + note)

if __name__ == "__main__":
    logger = Logger("test_logger")
    logger('debug','Debug message')
    logger('info','Info message')
    logger('warning','Warning message')
    logger('error','Error message')
    