from setuptools import find_packages, setup

from src.scaleagdata._version import __version__

setup(
    name="scaleagdata",
    version=f"{__version__}",
    include_package_data=True,
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "scikit-learn",
        "scikit-image",
        "matplotlib",
    ],
    dependency_links=[
        "git+https://github.com/WorldCereal/presto-worldcereal.git#egg=presto_worldcereal"
    ],
    zip_safe=True,
    author="Giorgia Milli",
    author_email="giorgia.milli@vito.be",
    description="using Presto for few-shot learning downstream tasks",
    url="https://github.com/ScaleAGData/scaleag-vito.git",
)
