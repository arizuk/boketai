#-*- encoding: utf-8 -*-
from pyknp import Jumanpp
import sys
import codecs

# JUMAN++をsubprocessモードで使用
jumanpp = Jumanpp()
result = jumanpp.analysis(u"ケーキを食べる")
for mrph in result.mrph_list():
    print(u"見出し:{0}".format(mrph.midasi))
