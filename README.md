# Music Emotion Recognition Research

This is a research about music emotion recognition conducted by Alex, Quan, and Rick.

## Folder structure of the project
* `/configs/`: contains global variables, setup, environment variables, training variables, and every other configurations to be plugged into the tool.
* `/data/`: contain music emotion datasets
* `/docs/`: contains resource, figures, and articles. Some of them are referred from the `backlog.md` file.
* `/mains/`: main files to be run
* `/mer/`: music emotion recognition code base library.
  * `/mer/mer/`: music emotion recognition code base library
    * `/mer/dataloader/`: contains files to preprocess data
    * `/mer/model/`: contains model builder files
  * `/mer/setup.py`: mer setup file
* `/models/`: contains trained models and checkpoint
* `/mss/`: music source separation code base library
  * `/mss/mss/`: music source separation code base library
    * `/mss/base/`: contains base file (model and trainer)
    * `/mss/dataloader/`: contains dataloader file
    * `/mss/model/`: contains model builder files
    * `/mss/trainer/`: contains trainer files
    * `/mss/utils/`: contain utility files
  * `/mss/setup.py`: mss setup file
* `/notebooks/`: contains notebook files for easy experiments.
* `/old/`: old experiment code base files, used for reference.

## How to test and experiment
`(to be implemented)`