from setuptools import find_packages, setup

__version__ = None

with open("src/scaleagdata/_version.py") as fp:
    exec(fp.read())

setup(
    name="scaleagdata_vito",
    version=f"{__version__}",
    include_package_data=True,
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"": ["scaleagdata/*", "scaleagdata/*/*"]},
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
