from setuptools import setup


def readme():
    with open('README.rst') as handle:
        return handle.read()

setup(
    name='rosaceae',
    version='0.0.1',
    description='Python pacakge for credit risk scorecards',
    long_description=readme(),
    author='l0o0',
    author_email='linxzh1989@gmail.com',
    license='MIT',
    keywords=['scorecards', 'woe'],
    url='',
    install_requires=['numpy', 'pandas', 'seaborn']
)
