from distutils.core import setup

setup(
    name='workflow',
    version='0.1.0',
    author='Justin Yuan',
    author_email='justin.zcyuan@gmail.com',
    packages=['workflow'],
    scripts=[],
    url='http://pypi.python.org/pypi/workflow/',
    license='LICENSE.txt',
    description='My machine learning experimentation workflow',
    long_description=open('README.txt').read(),
    install_requires=[
        "mlflow == 2.3.1",
        "yacs == 0.1.4",
    ],
)