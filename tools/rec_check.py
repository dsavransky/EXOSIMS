
from importlib import metadata
reqs = metadata.distribution('EXOSIMS').requires
required_packages = [str(req).split('>=')[0] for req in reqs]

with open('requirements.txt', 'r') as f:
    lines = f.readlines()

for package in required_packages:
    flag = False
    for line in lines:
        if package in line:
            flag = True
    assert flag, f'{package} not found in requirements.txt'
    print(f'{package} found in requirements.txt')
