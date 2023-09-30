(install)=

# Installation

## Prerequisites

This project requires python3 (>=3.8)

Graphics require the installation of [pygame](https://www.pygame.org/news), which itself has dependencies to be 
installed.

### Ubuntu

We recommend using [apt](https://ubuntu.com/server/docs/package-management)

```bash
sudo apt-get update -y
sudo apt-get install -y python-dev libsdl-image1.2-dev libsdl-mixer1.2-dev
    libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev python-numpy subversion libportmidi-dev
    ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev gcc
```

### MacOS

We recommend using [Homebrew](https://brew.sh) 

```bash
brew update
brew install apt-get install -y python-dev libsdl-image1.2-dev libsdl-mixer1.2-dev
    libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev python-numpy subversion libportmidi-dev
    ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev gcc
```

### Windows 10

We recommend using [Anaconda](https://conda.io/docs/user-guide/install/windows.html)

## Stable Release

To install the latest stable version:

```bash
pip install flyer-env
```

## Development version

To install the current version:

```bash
pip install --user git+https://github.com/AOS55/flyer-env
```