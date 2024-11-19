# Project Concept
The goal is to develop a CycleGAN model capable of converting aerial photographs into maps. Ultimately, the project seeks to create an application that can perform this conversion efficiently and accurately.

# Model
We are implementing a CycleGAN that learn the mapping between the source domain (aerial photographs) and the target domain (maps).

# Dataset
We consider using the dataset available at Kaggle, which contains unpaired images of aerial photos and maps.
Geoportal API: We are exploring additional data sources using the Geoportal API. However, we encountered difficulties with data collection on a large scale through the program initially provided by Dominik.

# Training
We set up Lightning AI Studio in our local environment, and an exemplary dataset was loaded to evaluate its capabilities. A basic pipeline was implemented and tested on sample data to assess its performance. A basic GAN was trained to generate MNIST digits in VS Code, and the free GPU models offered by the platform proved sufficient for our needs, eliminating the need for an upgraded plan.

# Implementation
We investigated deploying and querying a basic model on Hugging Face Model Hub and Amazon SageMaker, comparing costs and capabilities. Hugging Face Inference Endpoints were found to be more affordable on average compared to AWS SageMaker. Based on this, we recommend using Hugging Face Inference Endpoints for our initial deployment, with the option to switch to SageMaker in the future if needed.

# Your ideas...