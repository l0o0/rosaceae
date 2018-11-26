from setuptools import setup, find_packages


def readme():
    with open('README.md') as handle:
        return handle.read()

setup(
    name='rosaceae',
    version='0.0.7',
    description='Python pacakge for credit risk scorecards',
    long_description=readme(),
    long_description_content_type='text/markdown',
    author='l0o0',
    author_email='linxzh1989@gmail.com',
    keywords=['scorecards', 'woe', 'iv'],
    url='https://github.com/l0o0/rosaceae',
    install_requires=['numpy', 'pandas', 'seaborn', 'sklearn'],
    packages=find_packages()，
    install_requires=['seaborn', 'sklearn', 'numpy', 'pandas', 'scipy']
)
