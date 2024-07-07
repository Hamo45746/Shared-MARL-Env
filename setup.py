import os
from setuptools import find_packages, setup
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# The path to the requirements file
requirement_path = HERE / 'requirements.txt'

# Read the requirements
install_requires = []
if requirement_path.is_file():
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

setup(
    name="shared-marl-env",
    version="0.1.0", 
    description="A multi-agent reinforcement learning search & track environment with communication and jamming enabled",
    # long_description=README,
    # long_description_content_type="text/markdown",
    # author="Hamish Macintosh, Alex Martin-Wallace",
    # author_email="your.email@example.com",
    # license="MIT",
    # classifiers=[
    #     "License :: OSI Approved :: MIT License",
    #     "Programming Language :: Python :: 3",
    #     "Programming Language :: Python :: 3.9",],
    packages=find_packages(exclude=["tests", "docs"]),
    include_package_data=True,
    install_requires=install_requires,
    python_requires=">=3.9",
)