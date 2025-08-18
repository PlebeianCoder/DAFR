from setuptools import setup

setup(
    name="pyfacear",
    version="0.1.0",
    packages=[
        "pyfacear",
        "pyfacear.data",
        "pyfacear.platform",
        "pyfacear.face_geometry",
    ],
    package_dir={"": "src"},
    description="Augment faces with 3D effects.",
)
