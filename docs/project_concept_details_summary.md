# Project Concept

The goal is to develop a CycleGAN model capable of converting aerial photographs into maps. Ultimately, the project seeks to create an application that can perform this conversion efficiently and accurately.

# Model

We implement a CycleGAN that learns the bidirectional mapping between two domains: aerial images to maps and maps to aerial images, training in two cycles. To see if the model works, we trained it on this example Kaggle dataset: [CycleGAN datasets (maps)](https://www.kaggle.com/datasets/suyashdamle/cyclegan).

We also considered training on similar [Map Pix2Pix dataset](https://www.kaggle.com/datasets/valeriyl/yandex-maps-for-pix2pixhd)

# Dataset

We are trying to create a dataset of aerial images and coresponding maps or sketches.

We are exploring additional data sources using the Geoportal API. However, we encountered difficulties with data collection on a large scale from QGIS.

Plugin "Pobieracz danych GUGiK" enables to only manually downloading images in small parts (max 7km^2). Though we found that it is probably based on Geoportal WPS API. We could potentially gather data via this API but we don't know how to use it. Any help (if possible) would be greatly appreciated.

We can scrape data by doing automatic screenshots of Google Maps locations (both map and satellite layer) based on list of cities etc. Official Google Maps API is not designed for downloading map images.

Potential of mixing datasets is low (Geoportal, Kaggle and Google Maps images have different styles) - Can we pretrian on kaggle datasets and then finetune on Google Maps scraped data? (distinct stages or smooth transition between data distributions).

We were thinking on mixing data - pre-train on kaggle datasets and then fine-tune on Google Maps scraped data but since the map images from kaggle and Google differ we do not know if it could work.

We are open to receive some dataset from Mr. Dominik’s data.

# Training

We set up Lightning AI Studio in our local environment, and an exemplary dataset was loaded to evaluate its capabilities. A basic pipeline was implemented and tested on sample data to assess its performance. A basic GAN was trained to generate MNIST digits in VS Code, and the free GPU models offered by the platform proved sufficient for our needs, eliminating the need for an upgraded plan.

We will be able to share logs from the next training run when we implement WandB and get new data compute. Each of us can create account with free credits and train model. Some of us also have local GPUs

# Implementation

We investigated deploying and querying a basic model on Hugging Face Model Hub and Amazon SageMaker, comparing costs and capabilities. Hugging Face Inference Endpoints were found to be more affordable on average compared to AWS SageMaker. Based on this, we recommend using Hugging Face Inference Endpoints for our initial deployment, with the option to switch to SageMaker in the future if needed.

Expected web application tech stack: Java, Spring Boot, PostgreSQL, Next.js, Docker, AWS EB, AWS S3.

Any advice from Mr. Dominik regarding model cloud deployment is welcome.

# Research

In WandB, we currently display the results of the CycleGAN model training allowing for a preliminary evaluation of generator and discriminator losses. The loss graphs provide insight into model stabilization.

Future plans:

- Long-term Metric Tracking: WandB will enable continuous monitoring of progress across future training and testing sessions on new datasets.
- Result Comparison: Different versions of the model, hyperparameters, and datasets can be compared to determine the optimal configuration.
- Automated Reporting: Results logged in WandB can be used to generate reports and visualize progress in real time.
