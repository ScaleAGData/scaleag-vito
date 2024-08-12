from setuptools import find_packages, setup

setup(
    name="scaleagdata_vito",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "git+https://github.com/WorldCereal/presto-worldcereal.git",
    ],
    author="Giorgia Milli",
    author_email="giorgia.milli@vito.be",
    description="using Presto for few-shot learning downstream tasks",
    url="https://github.com/ScaleAGData/scaleag-vito.git",
)
