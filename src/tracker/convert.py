import os
labels = ['car', 'van', 'truck', 'pedestrian', 'cyclist', 'tram', 'dontcare']

for file in os.listdir('label_02-others'):
    with open(os.path.join('label_02-others', file)) as f:
        content = f.readlines()
    content = [x.strip().split(' ') for x in content]
    with open(os.path.join('label_02', file), 'w') as f2:
        for x in content:
            if int(x[1]) != -1 and labels.index(str(x[2]).lower()) != 6:
                f2.write(x[0] + ' ' + '{}'.format(labels.index(str(x[2]).lower())+1) + ' '
                         + x[6] + ' ' + x[7] + ' ' + x[8] + ' ' + x[9] + ' ' + '1.0' + '\n')
