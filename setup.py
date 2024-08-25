from setuptools import setup, find_packages

setup(
    name="llmtrack",
    version="0.2.0",  
    packages=find_packages(),
    install_requires=[
        'diskcache>=5.6.3',
        'openai>=1.23.1'
    ],
    include_package_data=True,
    description="A concise description of your package.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/xinzhel/llmtrack",  
    author="Xinzhe Li",
    author_email="xinzheli212@gmail.com",
    license="MIT", 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  
)
