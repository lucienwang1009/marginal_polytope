import tempfile
import os

smoker_drinkers_str1 = '''
predicate stress 1 1 1
predicate smokes 1 1 1
predicate drinks 1 1 1
predicate friends 2 1 1
predicate aux1 1 3.4 1
predicate aux2 2 8 1
predicate aux4 1 7.4 1
predicate aux5 1 4.5 1

-aux1(X), -stress(X), smokes(X)
aux1(X), stress(X)
aux1(X), -smokes(X)

-aux2(X,Y), -friends(X,Y), -smokes(X), smokes(Y)
aux2(X,Y), friends(X,Y)
aux2(X,Y), smokes(X)
aux2(X,Y), -smokes(Y)

-aux4(X), -stress(X), drinks(X)
aux4(X), stress(X)
aux4(X), -drinks(X)

-aux5(X,Y), -friends(X,Y), -drinks(X), drinks(Y)
aux5(X,Y), friends(X,Y)
aux5(X,Y), drinks(X)
aux5(X,Y), -drinks(Y)
'''

smoker_drinkers_str2 = '''
predicate smokes 1 1 1
predicate drinks 1 1 1
predicate friends 2 1 1
predicate aux1 1 3.4 1
predicate aux2 2 8 1
predicate aux3 1 2 1
predicate aux4 2 4.5 1

-aux1(X), smokes(X)
aux1(X), -smokes(X)

-aux2(X,Y), -friends(X,Y), -smokes(X), smokes(Y)
aux2(X,Y), friends(X,Y)
aux2(X,Y), smokes(X)
aux2(X,Y), -smokes(Y)

-aux3(X,Y), friends(X,Y)
aux3(X,Y), -friends(X,Y)

-aux4(X,Y), -drinks(X), -friends(X,Y), drinks(Y)
aux4(X,Y), drinks(X)
aux4(X,Y), friends(X,Y)
aux4(X,Y), -drinks(Y)
'''

friends_smokers_str = '''
predicate smokes 1 1 1
predicate friends 2 1 1
predicate aux1 1 3.4 1
predicate aux2 2 8 1
predicate aux3 2 2 1

-aux1(X), smokes(X)
aux1(X), -smokes(X)

-aux2(X,Y), -friends(X,Y), -smokes(X), smokes(Y)
aux2(X,Y), friends(X,Y)
aux2(X,Y), smokes(X)
aux2(X,Y), -smokes(Y)

-aux3(X,Y), friends(X,Y)
aux3(X,Y), -friends(X,Y)
'''


evidences = [
    'friends(0,2)',
    'friends(1,4)',
    '-friends(2,4)',
    '-friends(4,3)',
    'friends(4,5)',
    '-friends(4,2)',
    'friends(0,1)',
    'friends(1,3)',
    '-friends(3,4)',
    '-friends(0,3)',
    'friends(2,5)',
    '-friends(0,4)'
]

for i in range(len(evidences)):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tf:
        tf.write(smoker_drinkers_str2)
        for j in range(i + 1):
            tf.write(evidences[j] + '\n')
        tf.flush()
        os.system(
            'python approxwfomc.py 6 {} --debug --log logs/smokes_drinks2/evidence_new/origin/{}.log'.format(
                tf.name, i + 1
            )
        )
