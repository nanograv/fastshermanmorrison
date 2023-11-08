import os
from setuptools import setup, find_packages
from setuptools import Extension
from distutils.command.build import build as build_orig
from setuptools.command.build_py import build_py

# Dynamically calculate the version based on fastshermanmorrison.VERSION.
version_path = os.path.join(
    os.path.dirname(__file__), "fastshermanmorrison", "version.py"
)
version_ns = {}
with open(version_path) as f:
    exec(f.read(), {}, version_ns)


class BuildPyCommand(build_py):
    # Ensure that build_py command runs before build_ext.
    def run(self):
        self.run_command("build_ext")
        return super().run()


ext_modules = [
    Extension(
        name="fastshermanmorrison.cython_fastshermanmorrison",
        sources=[
            "fastshermanmorrison/cython_fastshermanmorrison.pyx",
            "fastshermanmorrison/fastshermanmorrison.c",
        ],
        extra_compile_args=["-O2", "-fno-wrapv"],
    )
]


class build(build_orig):
    def finalize_options(self):
        super().finalize_options()
        # __builtins__.__NUMPY_SETUP__ = False
        import numpy

        for extension in self.distribution.ext_modules:
            extension.include_dirs.append(numpy.get_include())
        from Cython.Build import cythonize

        self.distribution.ext_modules = cythonize(
            self.distribution.ext_modules, language_level=3
        )


setup(
    name="fastshermanmorrison-pulsar",
    version=version_ns["__version__"],
    description="Fast Sherman Morrison calculations for Enterprise",
    license="MIT",
    author="Rutger van Haasteren",
    author_email="rutger@vhaasteren.com",
    packages=find_packages(),
    package_dir={"fastshermanmorrison": "fastshermanmorrison"},
    url="http://github.com/vhaasteren/fastshermanmorrison/",
    long_description=open("README.md").read(),
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    ext_modules=ext_modules,
    package_data={
        "base": ["README", "LICENSE", "AUTHORS.md"],
        "fastshermanmorrison": [
            "fastshermanmorrison/cython_fastshermanmorrison.pyx",
            "fastshermanmorrison/fastshermanmorrison.c",
            "fastshermanmorrison/fastshermanmorrison.py",
        ],
    },
    cmdclass={"build": build},
)
