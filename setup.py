from setuptools import setup, find_packages

setup(
    name='scikit-learn-bench',
    version='0.0.1',
    description='A benchmarking suite for ML algorithms with profiling support',
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',
    author='Pierrick Pochelu',
    author_email='pierrick.pochelu@gmail.com',
    url='https://github.com/PierrickPochelu/scikit-learn-bench',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scikit-learn',
    ],
    entry_points={
        'console_scripts': [
             'scikit_learn_bench=scikit_learn_bench.cli:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    include_package_data=True,
)
