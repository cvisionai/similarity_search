from setuptools import setup, find_packages

setup(
  name = 'grafit-pytorch',
  packages = find_packages(exclude=['examples']),
  version = '0.1.0',
  license='MIT',
  description = 'Grafit Pytorch Implementation',
  author = 'Ben Woodward',
  author_email = 'benjamin.woodward@cvisionai.com',
  url = 'https://github.com/cvisionai/grafit-pytorch',
  keywords = [
      'self-supervised learning',
      'artificial intelligence'
  ],
  install_requires=[
      'torch>=1.10',
      'torchvision>=0.11.1',
      'pandas',
      'Pillow'
  ],
  classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6',
  ],
)
