from setuptools import setup, find_packages

setup(
    name="llmtrack",
    version="0.1.0",  # Update with your version number
    packages=find_packages(),
    install_requires=[
        'diskcache>=5.6.3',
        'openai>=1.23.1',  
        'tenacity>=8.2.3'
    ],
    include_package_data=True,
    description="A concise description of your package.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/xinzhel/llmtrack",  # Replace with your repo URL
    author="Xinzhe Li",
    author_email="xinzheli212@gmail.com",
    license="MIT",  # Replace with your chosen license
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Adjust based on your license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Adjust based on the Python versions you support
)
