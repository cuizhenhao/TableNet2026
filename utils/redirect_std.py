import logging
import os
from logging import handlers
def getLogger(filename):
  logger = logging.getLogger(filename)
  fmt = '%(asctime)s : %(message)s'
  format_str = logging.Formatter(fmt)  # 设置日志格式
  logger.setLevel(logging.DEBUG)  # 设置日志级别
  sh = logging.StreamHandler()  # 往屏幕上输出
  sh.setFormatter(format_str)  # 设置屏幕上显示的格式
  th = handlers.RotatingFileHandler(filename=filename, backupCount=30, maxBytes = 20000000, # 20MB日志文件
                                     encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
  th.setFormatter(format_str)  # 设置文件里写入的格式
  logger.addHandler(sh)  # 把对象加到logger里
  logger.addHandler(th)
  return logger

class ReDirectSTD(object):
  """Modified from Tong Xiao's `Logger` in open-reid.
  This class overwrites sys.stdout or sys.stderr, so that console logs can
  also be written to file.
  Args:
    fpath: file path
    console: one of ['stdout', 'stderr']
    immediately_visible: If `False`, the file is opened only once and closed
      after exiting. In this case, the message written to file may not be
      immediately visible (Because the file handle is occupied by the
      program?). If `True`, each writing operation of the console will
      open, write to, and close the file. If your program has tons of writing
      operations, the cost of opening and closing file may be obvious. (?)
  Usage example:
    `ReDirectSTD('stdout.txt', 'stdout', False)`
    `ReDirectSTD('stderr.txt', 'stderr', False)`
  NOTE: File will be deleted if already existing. Log dir and file is created
    lazily -- if no message is written, the dir and file will not be created.
  """

  def __init__(self, fpath=None, console='stdout', immediately_visible=False):
    import sys

    # assert console in ['stdout', 'stderr']
    # self.console = sys.stdout if console == 'stdout' else sys.stderr
    # self.file = fpath
    # self.f = None
    # self.immediately_visible = immediately_visible
    self.logger = getLogger(fpath)
    self.linebuf = ''

    # if fpath is not None:
    #   # Remove existing log file.
    #   if osp.exists(fpath):
    #     os.remove(fpath)

    # Overwrite
    if console == 'stdout | stderr':
      sys.stdout = self
      sys.stderr = self

    if console == 'stdout':
      sys.stdout = self
    else:
      sys.stderr = self

  def __del__(self):
    self.close()

  def __enter__(self):
    pass

  def __exit__(self, *args):
    self.close()

  def write(self, msg):
    temp_linebuf = self.linebuf + msg
    self.linebuf = ''
    for line in temp_linebuf.splitlines(True):
      # From the io.TextIOWrapper docs:
      #   On output, if newline is None, any '\n' characters written
      #   are translated to the system default line separator.
      # By default sys.stdout.write() expects '\n' newlines and then
      # translates them so this is still cross platform.
      if line[-1] == '\n':
        self.logger.info( line.rstrip())
      else:
        self.linebuf += line
    # self.console.write(msg)
    # if self.file is not None:
    #   may_make_dir(os.path.dirname(os.path.abspath(self.file)))
    #   if self.immediately_visible:
    #     with open(self.file, 'a') as f:
    #       f.write(msg)
    #   else:
    #     if self.f is None:
    #       self.f = open(self.file, 'w')
    #     self.f.write(msg)

  def flush(self):
    if self.linebuf != '':
      self.logger.info( self.linebuf.rstrip())
    self.linebuf = ''
    # self.console.flush()
    # if self.f is not None:
    #   self.f.flush()
    #   import os
    #   os.fsync(self.f.fileno())

  def close(self):
    pass
    # self.console.close()
    # if self.f is not None:
    #   self.f.close()


def may_make_dir(path):
  """
  Args:
    path: a dir, or result of `osp.dirname(osp.abspath(file_path))`
  Note:
    `osp.exists('')` returns `False`, while `osp.exists('.')` returns `True`!
  """
  # This clause has mistakes:
  # if path is None or '':

  if path in [None, '']:
    return
  if not os.path.exists(path):
    os.makedirs(path)



