from setuptools import setup, find_packages


def readme():
    with open('README.md') as handle:
        return handle.read()

setup(
    name='rosaceae',
    version='0.0.2',
    description='Python pacakge for credit risk scorecards',
    long_description=readme(),
    author='l0o0',
    author_email='linxzh1989@gmail.com',
    keywords=['scorecards', 'woe', 'iv'],
    url='',
    install_requires=['numpy', 'pandas', 'seaborn'],
    packages=find_packages()
)
