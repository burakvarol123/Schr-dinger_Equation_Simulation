_min = 5e-6
step = 5e-6
with open('dts', 'w') as fd:
    for i in range(100):
        fd.write('{:.2e}'.format(_min + i * step) + ' ')
