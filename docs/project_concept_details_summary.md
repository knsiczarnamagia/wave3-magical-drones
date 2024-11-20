# Project Concept
The goal is to develop a CycleGAN model capable of converting aerial photographs into maps. Ultimately, the project seeks to create an application that can perform this conversion efficiently and accurately.

# Model
We implement a CycleGAN that learns the bidirectional mapping between two domains: aerial images to maps and maps to aerial images, training in two cycles. To see if the model works, we trained it on this example Kaggle dataset: [CycleGAN datasets (maps)](<https://www.kaggle.com/datasets/suyashdamle/cyclegan>).

We also considered training on similar [Map Pix2Pix dataset](https://www.kaggle.com/datasets/valeriyl/yandex-maps-for-pix2pixhd)

# Dataset
Geoportal API: We are exploring additional data sources using the Geoportal API. However, we encountered difficulties with data collection on a large scale through the program initially provided by Mr. Dominik.

Plugin enables to only manually downloading images in small parts (max 7km^2). Plugin for QGIS is probably based on Geoportal WPS API. We are aware of Geoportal WMS API, but we don't know how to use it (asking for help).

We can scrape data by doing automatic screenshots of Google Maps localizations (based on list of cities etc.):
    - official Google Maps API is expensive and not designed for downloading map images.

We tested initial model trained on Kaggle dataset on several images from geoporal, but results are poor. We speculate that Kaggle data is highly skewed towards city images and the model is unable to properly convert rural terrains.

Potential of mixing datasets is low (Geoportal, Kaggle and Google Maps images have different styles)
    - Can we pretrain on Kaggle Datasets and then finetune on Google Maps scraped data? (distinct stages or smooth transition between data distributions).

We are open to receive any data from the company.

# Training
We set up Lightning AI Studio in our local environment, and an exemplary dataset was loaded to evaluate its capabilities. A basic pipeline was implemented and tested on sample data to assess its performance. A basic GAN was trained to generate MNIST digits in VS Code, and the free GPU models offered by the platform proved sufficient for our needs, eliminating the need for an upgraded plan.

We will be able to share logs from the next training run when we implement WandB and get new data compute. Each of us can create account with free credits and train model. Some of us also have local GPUs

# Implementation
We investigated deploying and querying a basic model on Hugging Face Model Hub and Amazon SageMaker, comparing costs and capabilities. Hugging Face Inference Endpoints were found to be more affordable on average compared to AWS SageMaker. Based on this, we recommend using Hugging Face Inference Endpoints for our initial deployment, with the option to switch to SageMaker in the future if needed.

Expected web application tech stack: Java, Spring Boot, PostgreSQL, Next.js, Docker, AWS EB, AWS S3.

Any advice from Mr. Dominik regarding model cloud deployment is welcome.

# Roles...

# Your ideas...