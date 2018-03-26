import sys
import codecs

f1 = codecs.open(sys.argv[1], 'r', 'utf-8')
f2 = codecs.open(sys.argv[2], 'r', 'utf-8')

fw1 = codecs.open('texts.v1', 'w', 'utf-8')
fw2 = codecs.open('images.v1', 'w', 'utf-8')

for line in f1:
  l1 = line.rstrip()
  l2 = f2.readline().rstrip()

  if l1 == 'VALUE ERROR':
    print('VALUE ERROR')
    continue
  # print("{}:{}".format(l1, l2))

  fw1.write(l1 + "\n")
  fw2.write(l2 + "\n")
