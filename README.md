### Overview
Reinforcement Learning agents for playing flappy-bird game.

### Installation
```
git clone https://github.com/curiousguy13/flappy-bird-agent.git
cd flappy-bird-agent
conda env create -f requirements/flappy-bird-linux.yml
source activate flappy-bird-project2
python requirements/pleInstall.py
sudo apt-get install libsm6 libxrender1 libfontconfig1 libgtk2.0 (for Ubuntu)
or
sudo yum install libXext libSM libXrender (for CentOS/Fedora)
```

### Usage
```
cd src
python a3c.py
```

All Configurations and hyper-parameters are in helper.py <br />
For Training: <br />
  Set TRAINING = True and TESTING = False in helper.py <br />
For Testing: <br />
  Set TRAINING = False and TESTING = True  <br />
  In case of Testing, the latest saved model will be loaded <br />
