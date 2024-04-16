# vivaldi_bh - Model Name: Vivaldi 
POC: Clinton Stipek - stipekcw@ornl.gov

## Getting started

This project's goal is to ingest morphology features (2D) and infer height (3D) for individual buildings:

1. The following breaks the vivald process into the respective steps
    - Identify 2D buildings from AOI
    - Generate morphology features using the Gauntlet feature morphology process
        - Please see Taylor Hauser (hausertr@ornl.gov) for availability of Gauntlet features
    - Run a recursive feature eliminator to streamline modelling process
    - Hyper-tune parameters via bayesian optimization
    - Infer building heights at a building-by-building level leveraging a XGBoost algorithm


## Docker
- There is a docker image for this project, to use the image please clone the repo and then go to vivaldi_bh/src for the docker files
- once cloned and in the right file trajectory, run the following lines in order in cmd line:
    1. docker-compose build vivaldi_bh
    2. docker-compose up -d vivaldi_by
    3. docker-compose exec vivaldi_bh python /files/vivaldi.py
- please note that for command 3, the 'vivaldi.py' is the vivaldi process outlined in Getting Started
- Please message Clinton Stipek (stipekcw@ornl.gov) for assistance

## Script run order
1. Run rfe.py (docker-compose exec vivaldi_by python /files/rfe.py - if using linux)
2. Run vivaldi_bh.py (docker-compose exec vivaldi_bh python /files/vivaldi_bh.py - if using linux)

## Data
- The data that vivaldi works with is built off the Gauntlet process
- Gauntlet v2 generates 65 morphological features that is in a tabular form at a building-by-building level 
- The Gauntlet features are stored in PostGresQL
- Please see Taylor Hauser (hausertr@ornl.gov) for access to the data
- Please see Clinton Stipek (stipekcw@ornl.gov) for gauntlet features necessary to run rfe and vivaldi_bh


