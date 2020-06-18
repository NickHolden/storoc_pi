# storoc_pi

Installation steps **not** guaranteed work without adjustments on devices other than the Raspberry Pi 4 using PiCamera V2.1.

## Installation steps for Raspberry Pi

### Update Raspberry Pi

```bash
sudo apt-get update && sudo apt-get dist-upgrade
```

### Setting up virtual environment

Install virtualenv package

```bash
sudo pip3 install virtualenv
```

Create virtual environment

```bash
python3 -m venv storoc
```

Source virtual environment

```bash
source storoc/bin/activate
```

### Install TensorFlow Lite dependencies and OpenCV

Run the shell script included in the repository which will take care of this

```bash
bash install-requirements.sh
```

Note - if trying to run this step on a different machine than raspberry pi a different TensorFlow Lite version must be installed.



