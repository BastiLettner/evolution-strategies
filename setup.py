from setuptools import setup


setup(
    name='evolution strategies',
    py_modules=['es'],
    version="1.0",
    install_requires=[
        'numpy',
        'ray',
        'lz4',  # recommended by ray for performance
        'matplotlib'
    ],
    description="Simplified implementation of ES with ray with different interface",
    author="sebastian-lettner",
)
