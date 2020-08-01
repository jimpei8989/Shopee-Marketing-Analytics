import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

readme = (here / "README.md").read_text()
reqs = (here / "requirements.txt").read_text()

setup(
    name="scl8",
    version="0.0.1",
    author="AKA",
    url="https://github.com/jimpei8989/Shopee-Marketing-Analytics",
    description="Shopee Code League 2020, Competition #8 - Marketing Analytics",
    long_description_content_type="text/markdown",
    long_description=readme,
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=reqs,
    entry_points={
        'console_scripts': ['scl8=scl8:main']
    },
)
