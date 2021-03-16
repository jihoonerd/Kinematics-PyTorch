from setuptools import setup, find_packages

setup(name='kinematics-pytorch',
      version='0.0.1',
      description='Kinematics with PyTorch',
      author='Jihoon Kim',
      author_email='jihoon_kim@outlook.com',
      url='https://github.com/jihoonerd/Kinematics-PyTorch',
      packages=find_packages(exclude=['docs', 'tests*']),
     )