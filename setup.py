from setuptools import setup, find_packages

setup(
    name="rlhelper",
    version="0.1.0",
    description="Utility per Reinforcement Learning tabulare e deep RL (Q-Learning, SARSA, Dyna-Q, DDQN)",
    long_description=open("README.md", encoding="utf-8").read() + "\n\nNota: rlhelper richiede Gymnasium >=0.28. La funzione dyna_q utilizza 'current_action_space', che potrebbe non essere presente in tutti gli ambienti standard.",
    long_description_content_type="text/markdown",
    author="Tuo Nome",
    author_email="email@example.com",
    url="",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "gymnasium>=0.28"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
    license="Apache-2.0",
)
