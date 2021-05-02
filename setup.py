from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='kpt',
      version='0.0.7',
      description='Kinematics with PyTorch',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Jihoon Kim',
      author_email='jihoon_kim@outlook.com',
      url='https://github.com/jihoonerd/Kinematics-PyTorch',
      packages=['kpt', 'kpt.model'],
      install_requires=[
        'numpy',
        'torch',
        'pytorch3d'
      ]
)