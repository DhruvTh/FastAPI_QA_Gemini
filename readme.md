```markdown
# OFA Setup Guide

Quick setup instructions for the Open-Ended Feature Aggregator (OFA) environment and Docker container.

## Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html)
- [Docker](https://docs.docker.com/get-docker/)

## Installation

### Create Conda Environment

```bash
conda create -n ofa python=3.11
conda activate ofa
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Build Docker Image

```bash
sudo docker build -t ofa:1 .
```

### Run Docker Container

```bash
sudo docker run --env-file .env -p 9000:9000 ofa:1
```

Access the application at `localhost:9000`.
```