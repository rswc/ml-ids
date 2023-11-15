from setuptools import setup, find_packages


setup(
    name='ml-ids',
    version='0.1',
    packages=find_packages('src'),
    install_requires=[
        'colorama==0.4.6',
        'tqdm==4.66.1',
        'river==0.20.1',
        'pytest==7.4.3'
    ],
)
