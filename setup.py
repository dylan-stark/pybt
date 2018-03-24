from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='pybt',
      version='0.3.0',
      description='Simple Populatin Based Training',
      long_description=readme(),
      url='http://github.com/dylan-stark/pybt',
      author='Dylan Stark',
      author_email='dylan.stark@gmail.com',
      license='MIT',
      install_requires=['numpy>=1.9.1',
                        'pandas'],
      extras_require={
          'tests': ['pytest',
                    'pytest-pep8'],
      },
      packages=['pybt'],
      zip_safe=False)

