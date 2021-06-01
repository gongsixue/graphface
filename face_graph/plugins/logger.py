# logger.py

import os
import pdb


class Logger:
    def __init__(self, path, filename, enabled=True):
        self.enabled = enabled
        if not self.enabled:
            return
        self.num = 0
        if not os.path.isdir(path):
            os.makedirs(path)
        self.filename = os.path.join(path, filename)
        self.fid = open(self.filename, 'w')
        self.fid.close()

    def register(self, modules):
        if not self.enabled:
            return
        self.num = self.num + len(modules)
        tmpstr = ''
        for tmp in modules:
            tmpstr = tmpstr + tmp + '\t'
        tmpstr = tmpstr + '\n'
        self.fid = open(self.filename, 'a')
        self.fid.write(tmpstr)
        self.fid.close()

    def update(self, modules):
        if not self.enabled:
            return
        tmpstr = ''
        for i,tmp in enumerate(modules):
            if i == 0:
                tmpstr = tmpstr + '%s' % (modules[tmp]) + '\t'
            else:
                tmpstr = tmpstr + '%.6f' % (modules[tmp]) + '\t'
        tmpstr = tmpstr + '\n'
        self.fid = open(self.filename, 'a')
        self.fid.write(tmpstr)
        self.fid.close()
