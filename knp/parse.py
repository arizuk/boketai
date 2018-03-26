#-*- encoding: utf-8 -*-
from pyknp import Jumanpp
import sys
import codecs

jumanpp = Jumanpp()
f = codecs.open(sys.argv[1], 'r', 'utf-8')
for line in f:
  text = line.rstrip()
  try:
    result = jumanpp.analysis(text)
    tokens = [mrph.midasi for mrph in result.mrph_list()]
    print('\t'.join(tokens))
  except ValueError:
    print('VALUE ERROR')


f.close

# # JUMAN++をsubprocessモードで使用
# jumanpp = Jumanpp()
# result = jumanpp.analysis(u"ケーキを食べる")
# for mrph in result.mrph_list():
#     print(u"見出し:{0}".format(mrph.midasi))
