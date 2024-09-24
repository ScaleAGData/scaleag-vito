from setuptools import find_packages, setup

from scaleagdata_vito._version import __version__

setup(
    name="scaleagdata_vito",
    version=f"{__version__}",
    description="using Presto for few-shot learning downstream tasks",
    url="https://github.com/ScaleAGData/scaleag-vito.git",
    author="Giorgia Milli",
    author_email="giorgia.milli@vito.be",
    include_package_data=True,
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "scikit-learn",
        "scikit-image",
        "matplotlib",
        "loguru",
        "rioxarray",
        "rasterio",
    ],
    dependency_links=[
        "git+https://github.com/WorldCereal/presto-worldcereal.git#egg=presto",
        "git+https://github.com/Open-EO/openeo-gfmap.git#egg=openeo_gfmap",
        "git+https://github.com/WorldCereal/worldcereal-classification.git#egg=worldcereal",
    ],
    zip_safe=True,
)
