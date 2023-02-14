import io

from setuptools import setup

with open('requirements.txt') as f:
    required = f.readlines()

setup(
    name='luxonis-ml',
    version='0.0.1',
    description='This package provides MLOps tools for training models for OAK devices',
    long_description=io.open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url='https://github.com/luxonis/luxonis-ml-library/',
    keywords="ml ops luxonis oak camera",
    author='Luxonis',
    author_email='support@luxonis.com',
    license='MIT',
    packages=['luxonis_ml'], # https://docs.python.org/3/distutils/setupscript.html point 2.1
    package_dir={"": "src"},  # https://stackoverflow.com/a/67238346/5494277
    install_requires=required,
    include_package_data=True,
    classifiers=[
        'License :: MIT License',
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    entry_points={
        "mlflow.request_header_provider": "unused=luxonis_ml.ops.mlflow_plugins:LuxonisRequestHeaderProvider",
        "console_scripts": ["luxonis_ml=luxonis_ml.luxonis_ml:main"]
    },
)
