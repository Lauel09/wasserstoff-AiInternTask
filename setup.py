from setuptools import setup, find_packages

setup(
    name='internship-task',
    version='0.1',
    author='Siddharth Upadhyay',
    description='Internship task for a company',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'pandas',
        'matplotlib',
        'tabulate',
        'streamlit',
        'easyocr',
        'numpy',
        'Pillow',
        'ultralytics'
    ],
)