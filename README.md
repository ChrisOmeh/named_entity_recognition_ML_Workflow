# Intern Detalis

* Name: Omeh Chukwuemeka<br>
* Student_ID:c01e2e <br>
* Stage: D <br>
* Stage Lesson: Setting up Kubernetes using Microk8s.<br>
* Task Target: Deploy a Machine Learning Pipeline on Kubeflow using GCP.
* Github Codes: [link](https://github.com/ProsperOmeh/named_entity_recognition_ML_Workflow)

# Task Name: Named Entity Recognition with Kubeflow and Keras 
For this stage of the internship, apart from setting up Microk8s, we were required also to successfully use Kubeflow for model training and deployment. See project details below in kubeflow examples github link.

In this walkthrough, you will learn how to use Kubeflow to build reusable components to train your model on an kubernetes cluster and deploy it to AI platform. <b>Task Documentation link:</b> https://github.com/kubeflow/examples/tree/master/named_entity_recognition

####  What is Named Entity Recognition
Named Entity Recognition is a word classification problem, which extract data called entities from text.

# Task Goals/Aim

1. Demonstrate how to build reusable pipeline components
2. Demonstrate how to use Keras only models
3. Demonstrate how to train a Named Entity Recognition model on a Kubernetes cluster
4. Demonstrate how to deploy a Keras model to AI Platform
5. Demonstrate how to use a custom prediction routine
6. Demonstrate how to use Kubeflow metrics
7. Demonstrate how to use Kubeflow visualizations

# Task Problem Statement
Machine Learning processes is such an iterative one. By this i mean that Machine Learning work cycle is recycling in nature. As a result, most often Machine Learning Engineers need to automate this work flow process to make it easy and reuseable.

To achieve this, there comes the principle of building  workflow pipeline. An ordinary Machine Learning Engineer or Data Scientist may not be able to do this because he or she was trained to do so. Therefore, there is a need for Data or MLOps Engineer to build, compile and run a reuseable pipeline that everyone in the team can use at anytime.

In this project, the target is to build a pipeline for preprocessing Machine learning data, training and deployment.

# Files Descriptions

The project documentations on kubeflow examples on github shows we have some folder/files that contains things that will be done in each stage of the pipeline development. Listed below are these floders and there descriptions.

1. components: This folder contains the named_entity_recognition ML work flow pipeline components. The components are Preprocess, Train and Deploy components. Inside the each components, there are scripts for creating components docker image using Dockerfile, scripts to build the docker image and also python file for preprocessing and training the dataset.

2. documentation: The documentation folder contains several documentation steps taken to achieve this project. All Right and Reservation to kubeflow/examples.

3. notebook: The notebook folder contains Pipeline.ipynb file which is a notebook file that contains code artifacts for loading our dataset, building the components pipeline, compiling the pipeline, creating a pipeline experiment and finally running the pipeline on kubeflow dashboard.

4. routine: The routine folder is all about creating a custom prediction.

5. README.md: The README.md file contains the steps that a user of the files should observe to achieve likely if not same results.

![Files Image in my Githun repo](pictures/files.png)

# Steps
This is links to documented steps on kubeflow/examples on github

1. [Setup Kubeflow and clone repository](documentation/step-1-setup.md)
1. [Build the pipeline components](documentation/step-2-build-components.md)
1. [Upload the dataset](documentation/step-3-upload-dataset.md)
1. [Custom prediction routine](documentation/step-4-custom-prediction-routine.md)
1. [Run the pipeline](documentation/step-5-run-pipeline.md)
1. [Monitor the training](documentation/step-6-monitor-training.md)
1. [Predict](documentation/step-7-predictions.md)

# Step 1: Setting up of Kubeflow and cloning of task repository


### Deploying Kubeflow to Google Cloud Platform
In this step, i deployed kubeflow to Google Cloud Platform using the Command Line(CLI) approach. This is because this approach gives me the ability to run command directly from the terminal and it gives me the authority to understand and also see scripts/commands and know how they work and function. The documentations on how to setup a Kubeflow environment by using the [Command Line Approach](https://www.kubeflow.org/docs/gke/deploy/deploy-cli/).


 
 Below snapshot shows successful deployment of Kubeflow indicating its kubeflow dashboard<br>

###### Kubeflow UI

![Kubeflow dashboard](pictures/kubeflow_ui.png)

<br>

### Set environment variables

Create the following environment variables, follow the [documenation](https://cloud.google.com/resource-manager/docs/creating-managing-projects#identifying_projects) to get the project id :

Below is my PROJECT_ID and BUCKET name which i exported for easy echoing or referencing.

```bash
export BUCKET=chris-bucket
export PROJECT_ID=hamoye-kubeflow1
```

###  Create bucket
Creating a bucket that will contain everything required for our Kubeflow pipeline. My bucket region is US-CENTRAL1. The below bash command will create a google cloud storage bucket for our file storage.

```bash
gsutil mb -c regional -l us-central1 gs://${BUCKET}
```

### Cloning the examples repository

The Cloned repository contains everything needed for this example. The cloned repo is from kubeflow examples. The bash command below will clone the repo

```bash
git clone https://github.com/kubeflow/examples.git
```

Next is to open a Terminal and navigate to the folder `/examples/named_entity_recognition/` which is in the cloned repository.


[Link to kubeflow documentation of this step](https://github.com/kubeflow/examples/blob/master/named_entity_recognition/documentation/step-1-setup.md)

#  

# Step 2: Build the pipeline components

### Build components

A component is code that performs one step in the Kubeflow pipeline. It is a containerized implementation of an ML task. **Components can be reused in other pipelines.**

### Component structure
A component follows a specific structure and contains some configuartion files.The summary steps of kubeflow componenents structure is:

1. Write the program that contains your component’s logic. The program must use files and     command-line arguments to pass data to and from the component.
2. Containerize the program.
3. Write a component specification in YAML format that describes the component for the Kubeflow Pipelines system.
4. Use the Kubeflow Pipelines SDK to load your component, use it in a pipeline and run that pipeline.
The configuration files includes:
 
* `/src` - Component logic . 
* `component.yaml` - Component specification. 
* `Dockerfile` - Dockerfile to build the container. 
* `readme.md` - Readme to explain the component and its inputs and outputs. 
* `build_image.sh` - Scripts to build the component and push it to a Docker repository. 

## Components
This Kubeflow project contains 3 components:

### Preprocess component
The preprocess component is downloading the training data and performs several preprocessing steps. This preprocessing step is required in order to have data which can be used by our model. 


### Train component
The train component is using the preprocessed training data. Contains the model itself and manages the training process. 

### Deploy component
The deploy component is using the model and starts a deployment to AI Platform. 

## Build and push component images
In order to use the components later on in our pipelines,you have to build and then push the image to a Docker registry. In this example, you are using the 
[Google Container Registry](https://cloud.google.com/container-registry/), it is possible to use any other docker registry. 

Each component has its dedicated build script `build_image.sh`, the build scripts are located in each component folder:

* `/components/preprocess/build_image.sh`
* `/components/train/build_image.sh`
* `/components/deploy/build_image.sh`

To build and push the Docker images open a Terminal, navigate to `/components/` and run the following command:

```bash
$ ./build_components.sh
```


## Check that the images are successfully pushed to the Google Cloud Repository

Navigate to the Google Cloud Container Registry and validate that you see the components. As shown in below snapshot, my components container image are uploaded successfully in my GCR

![image.png](attachment:image.png)

## Upload the component specification
The specification contains anything needed to use the component. 
It also contains the path to our docker images, open `component.yaml` for each component and set **`<PROJECT-ID>`** to your Google Cloud Platform project id.

Upload all three component specifications to your Google Cloud Storage and make it public accessible by setting the permission to `allUsers`.



Navigate to the components folder `/components/` open `copy_specification.sh` set your bucket name `BUCKET="chrisc-bucket"` and run the following command:

```bash
$ ./copy_specification.sh
```

[Link to kubeflow documentation of this step](https://github.com/kubeflow/examples/blob/master/named_entity_recognition/documentation/step-2-build-components.md)

#  

# STEP 3: Dataset Dataset description

This example project is using the popular CoNLL 2002 dataset. The csv consists of multiple rows each containing a word with the corresponding tag. Multiple rows are building a single sentence. 

The dataset itself contains different tags
* geo = Geographical Entity 
* org = Organization 
* per = Person 
* gpe = Geopolitical Entity 
* tim = Time indicator 
* art = Artifact 
* eve = Event 
* nat = Natural Phenomenon

Each tag is defined in an IOB format, IOB (short for inside, outside, beginning) is a common tagging format for tagging tokens.

> B - indicates the beginning of a token

> I - indicates the inside of a token

> O - indicates that the token is outside of any entity not annotated

### Example

```bash
"London on Monday evening"
"London(B-geo) on(O) Monday(B-tim) evening(I-tim)"
```

## Data Preparation
You can download the dataset from the [Kaggle dataset](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus). In order to make it convenient we have uploaded the dataset on GCS.

```
gs://kubeflow-examples-data/named_entity_recognition_dataset/ner.csv
```

> The training pipeline will use this data, there are no further data preperation steps required.

[Kubeflow examples reference link](https://github.com/kubeflow/examples/blob/master/named_entity_recognition/documentation/step-3-upload-dataset.md)

#  

# STEP 4: Custom prediction routine

Custom prediction routines allow us to specify additional code that runs with every prediction request.
Without custom prediction routine the machine learning framework handles the prediction operation.

## Why custom prediction routine
Our model requires numeric inputs, which we convert from text before training (this is the preprocessing step). To perform the same conversion at prediction time, inject the preprocessing code by defining a custom prediction routine.

> Without a custom prediction routine, you would need to create a wrapper, e.g. with App Engine or Cloud Functions, which would add complexity and latency.

## How do custom prediction routines work?

Our custom prediction routine requires six parts

* `keras_saved_model.h5` - The model stored as part of our training component (artifact).
* `processor_state.pkl` - The preprocessing state stored as part of our training component (artifact).
* `model_prediction.py` - The custom prediction routine logic.
* `text_preprocessor.py` - The pre-processing logic.  
* `custom_prediction_routine.tar.gz` - A Python package `tar.gz` which contains our implementation.
* `setup.py` - Used to create the Python package. 

To build our custom prediction routine run the build script located `/routine/build_routine.sh`. This creates a `tar.gz` which is required when you deploy your model. 

Navigate to the routine folder `/routine/` and run the following build script:

```bash
$ ./build_routine.sh
```

## Upload custom prediction routine to Google Cloud Storage

```bash
gsutil cp custom_prediction_routine-0.2.tar.gz gs://${BUCKET}/routine/custom_prediction_routine-0.2.tar.gz
```

[Kubeflow examples Reference Link](https://github.com/kubeflow/examples/blob/master/named_entity_recognition/documentation/step-4-custom-prediction-routine.md)

#  

# STEP 5: Run the pipeline


###  Open the Kubeflow Notebook
The pipeline was created using a Jupyter notebook. For that,the Notebook was created in Kubeflow Notebook server. The notebook server and the Pipeline.ipynb snapshot is attached below<br>
![Notebook.png](attachment:image.png))

Open the Jupyter notebook interface and create a new Terminal by clicking on menu, New -> Terminal. In the Terminal, clone this git repo by executing:

```bash
git clone https://github.com/kubeflow/examples.git
```

Now you have all the code required to run the pipeline. Navigate to the `examples/named-entity-recognition/notebooks` folder and open `Pipeline.ipynb`

### Configuring the pipeline

The pipeline need several parameter in order to execute the components. After you set up all the parameter, run the notebook and click on the `Open experiment` link.

### Configuring preprocess component

* `input_1_uri` - The input data csv
* `output_y_uri_template` - Output storage location for our preprocessed labels.
* `output_x_uri_template` - Output storage location for our preprocessed features.
* `output_preprocessing_state_uri_template` - Output storage location for our preprocessing state.

### Configuring train component

* `input_x_uri` - Output of the previous pipeline step, contains preprocessed features.  
* `input_y_uri` - Output of the previous pipeline step, contains preprocessed labels.
* `input_job_dir_uri` - Output storage location for the training job files.
* `input_tags` - Output of the previous pipeline step, contains the number of tags.
* `input_words` - Output of the previous pipeline step, contains the number of words. 
* `output_model_uri_template` - Output storage location for our trained model. 


### Configuring deploy component
* `model_path` - The model path is the output of the previous pipeline step the training. 
* `model_name` - The model name is later displayed in AI Platform.
* `model_region` - The region where the model sould be deployed.
* `model_version` - The version of the trained model. 
* `model_runtime_version` - The runtime version, in your case you used TensorFlow 1.13 .
* `model_prediction_class` - The prediction class of our custom prediction routine. 
* `model_python_version` - The used python version
* `model_package_uris` - The package which contains our custom prediction routine. 


## Whats happening in the notebook?


### Load the component

Components can be used in Pipelines by loading them from an URL. Everyone with access to the Docker repository can use these components.
The component can be loaded via components.load_component_from_url()

```python
preprocess_operation = kfp.components.load_component_from_url(
    'https://storage.googleapis.com/{}/components/preprocess/component.yaml'.format(BUCKET))
help(preprocess_operation)

train_operation = kfp.components.load_component_from_url(
    'https://storage.googleapis.com/{}/components/train/component.yaml'.format(BUCKET))
help(train_operation)

ai_platform_deploy_operation = comp.load_component_from_url(
    "https://storage.googleapis.com/{}/components/deploy/component.yaml".format(BUCKET))
help(ai_platform_deploy_operation)
```

Example based on the training component:

1. `kfp.components.load_component_from_url` loads the pipeline component.
2. You then have a operation that runs the container image and accepts arguments for the component inputs.

![use component](files/load-component.png)

### Create the pipeline
The pipeline is created by defining a decorator.  The dsl decorator is provided via the pipeline SDK. `dsl.pipeline` defines a decorator for Python functions which returns a pipeline.

```python
@dsl.pipeline(
  name='Named Entity Recognition Pipeline',
  description='Performs preprocessing, training and deployment.'
)
def pipeline():
    ...
```

### Compile the pipeline
To compile the pipeline you use the `compiler.Compile()` function which is part of the pipeline SDK. 
The compiler generates a yaml definition which is used by Kubernetes to create the execution resources.

```python
pipeline_func = pipeline
pipeline_filename = pipeline_func.__name__ + '.pipeline.zip'

import kfp.compiler as compiler
compiler.Compiler().compile(pipeline_func, pipeline_filename)
```

### Create an experiment
Pipelines are always part of an experiment.
They can be created with the Kubeflow pipeline client `kfp.client()`. 
Experiments cannot be removed at the moment.

```python
client = kfp.Client()

try:
    experiment = client.get_experiment(experiment_name=EXPERIMENT_NAME)
except:
    experiment = client.create_experiment(EXPERIMENT_NAME)
```

### Run the pipeline
Use the experiment id and the compiled pipeline to run a pipeline. `client.run_pipeline()` runs the pipelines and provides a direct link to the Kubeflow experiment.

```python
arguments = {}

run_name = pipeline_func.__name__ + ' run'
run_result = client.run_pipeline(experiment.id, 
                                 run_name, 
                                 pipeline_filename, 
                                 arguments)
```
[Reference Link to kubeflow documentation of this step](https://github.com/kubeflow/examples/blob/master/named_entity_recognition/documentation/step-5-run-pipeline.md)

#  

# STEP 6: Monitor your pipeline


## Pipeline steps
Open Kubeflow and go to `pipeline dashboard` click `experiments` and open the `run`. You can see the pipeline graph which shows each step in our pipeline. As you can see all of your steps completed successfully but only the Deploy component failed. At time of deadline submission, i was still on unable to resolve the issue. This error prevented further visualization of the training metrics

![monitor pipelines](pictures/pipeline_graphs.png)


## Open TensorBoard
During the training of your model, you are interested how your model loss and accuracy changes for each iteration. TensorBoard provides a visual presenation of iterations. 

The logs of the training are uploaded to a Google Cloud Storage Bucket. TensorBoard automatically references this log location and displays the corresponding data. 

The training component contains a TensorBoard visualization (TensorBoard viewer), which makes is comfortable to open the TensorBoard session for training jobs.

To open TensorBoard click on the `training` component in your experiment run. Located on the ride side is the artifact windows which shows a very handy button called (Open TensorBoard).

In order to use his visualizations, your pipeline component must write a JSON file. Kubeflow provides a good documenation on [how visualizations are working](https://www.kubeflow.org/docs/pipelines/sdk/output-viewer/) and what types are available.

```
# write out TensorBoard viewer
metadata = {
    'outputs' : [{
      'type': 'tensorboard',
      'source': args.input_job_dir,
    }]
}

with open('/mlpipeline-ui-metadata.json', 'w') as f:
  json.dump(metadata, f)
```



## Training metrics

Your training component creates a metric (accuracy-score) which are displayed in the experiment UI. With those metrics, you can compare your different runs and model performance.
![Training Metrics](pictures/trainm.png)




[Kubeflow examples reference link](https://github.com/kubeflow/examples/blob/master/named_entity_recognition/documentation/step-6-monitor-training.md)

#  

# STEP 7 : Prediction

Due to deploy component unresolved bug at time of preparing this report for submission, this step was documented as required. The steps outlined below is as documented in kubeflow examples.

Open AI Platform and navigate to your [model](https://console.cloud.google.com/ai-platform/models), there is one model listed: 

![ai platform models](files/models.png)

Open the model and choose your version then click on the Tab `TEST & USE` and enter the following input data:

```
{"instances":  ["London on Monday evening"]}
```
![ai platform predict](files/predict.png)

After a couple of seconds, you get the prediction response. Where `London` got pedicted as geolocation (B-geo), and `Monday evening` as time where Monday is the beginning (B-tim) and evening is inisde (I-tim). 

```json
{
  "predictions": [
    [
      "B-geo",
      "O",
      "B-tim",
      "I-tim",
       ]
  ]
}
```

Congratulations you trained and deployed a Named Entity Recognition model where you can extract entities. There are many use cases where such models can be used.

Examples:

* Optimize search results by extract specific entities out of search queries.
* Classify large document archives by making entities filterable.
* Enhance access for digital research of large document archives.
* Route customer support message by extracting the department or product.

[Step reference link](https://github.com/kubeflow/examples/blob/master/named_entity_recognition/documentation/step-7-predictions.md)


[Pipepline building process steps](README.md)

# Named Entity Recognition pipeline codes Snippets


```python
EXPERIMENT_NAME = 'named-entity-recognition'
BUCKET = "chrisc-bucket"
```


```python
import tensorflow as tf
from tensorflow import keras
print(keras.__version__)
print(tf.__version__)
```

    2.4.0
    2.3.0


## Imports


```python
import kfp
from kfp import compiler
import kfp.components as comp
import kfp.dsl as dsl
from kfp import gcp
```

## Load components


```python
preprocess_operation = kfp.components.load_component_from_url(
    'https://storage.googleapis.com/{}/components/preprocess/component.yaml'.format(BUCKET))
help(preprocess_operation)

train_operation = kfp.components.load_component_from_url(
    'https://storage.googleapis.com/{}/components/train/component.yaml'.format(BUCKET))
help(train_operation)

ai_platform_deploy_operation = comp.load_component_from_url(
    "https://storage.googleapis.com/{}/components/deploy/component.yaml".format(BUCKET))
help(ai_platform_deploy_operation)
```

    Help on function preprocess:
    
    preprocess(input_1_uri:'GCSPath', output_x_uri_template:'GCSPath', output_y_uri_template:'GCSPath', output_preprocessing_state_uri_template:'GCSPath')
        preprocess
        Performs the IOB preprocessing.
    
    Help on function train:
    
    train(input_x_uri:'GCSPath', input_y_uri:'GCSPath', input_job_dir_uri:'GCSPath', input_tags:int, input_words:int, input_dropout:float, output_model_uri_template:'GCSPath')
        train
        Trains the NER Bi-LSTM.
    
    Help on function deploy:
    
    deploy(model_path:'GCSPath', model_name:str, model_region:str, model_version:str, model_runtime_version:str, model_prediction_class:str, model_python_version:str, model_package_uris:str)
        deploy
        Deploy the model with custom prediction route
    


## Build the Pipeline 


```python
@dsl.pipeline(
  name='Named Entity Recognition Pipeline',
  description='Performs preprocessing, training and deployment.'
)
def pipeline():
    
    preprocess_task = preprocess_operation(
        input_1_uri='gs://kubeflow-examples-data/named_entity_recognition_dataset/ner.csv',
        output_y_uri_template="gs://{}/{{workflow.uid}}/preprocess/y/data".format(BUCKET),
        output_x_uri_template="gs://{}/{{workflow.uid}}/preprocess/x/data".format(BUCKET),
        output_preprocessing_state_uri_template="gs://{}/{{workflow.uid}}/model".format(BUCKET)
    ).apply(kfp.gcp.use_gcp_secret('user-gcp-sa')) 
    
    
    train_task = train_operation(
        input_x_uri=preprocess_task.outputs['output_x_uri'],
        input_y_uri=preprocess_task.outputs['output_y_uri'],
        input_job_dir_uri="gs://{}/{{workflow.uid}}/job".format(BUCKET),
        input_tags=preprocess_task.outputs['output_tags'],
        input_words=preprocess_task.outputs['output_words'],
        input_dropout=0.1,
        output_model_uri_template="gs://{}/{{workflow.uid}}/model".format(BUCKET)
    ).apply(kfp.gcp.use_gcp_secret('user-gcp-sa')) 
    
    
    deploy_task = ai_platform_deploy_operation(
        model_path= train_task.output,
        model_name="named_entity_recognition_kubeflow",
        model_region="us-central1",
        model_version="version1",
        model_runtime_version="1.13",
        model_prediction_class="model_prediction.CustomModelPrediction",
        model_python_version="3.5",
        model_package_uris="gs://{}/routine/custom_prediction_routine-0.2.tar.gz".format(BUCKET)
    ).apply(kfp.gcp.use_gcp_secret('user-gcp-sa'))
```

## Compile the Pipeline


```python
pipeline_func = pipeline
pipeline_filename = pipeline_func.__name__ + '.pipeline.zip'

import kfp.compiler as compiler
compiler.Compiler().compile(pipeline_func, pipeline_filename)
```

## Create a Kubeflow Experiment


```python
client = kfp.Client()

try:
    experiment = client.get_experiment(experiment_name=EXPERIMENT_NAME)
except:
    experiment = client.create_experiment(EXPERIMENT_NAME)
    
print(experiment)
```

    {'created_at': datetime.datetime(2020, 9, 18, 12, 10, 34, tzinfo=tzlocal()),
     'description': None,
     'id': 'eb8858e8-2f1a-4c1e-b275-999fede00a0a',
     'name': 'named-entity-recognition',
     'resource_references': None,
     'storage_state': None}


## Run the Pipeline


```python
arguments = {}

run_name = pipeline_func.__name__ + ' run'
run_result = client.run_pipeline(experiment.id, 
                                 run_name, 
                                 pipeline_filename, 
                                 arguments)

print(experiment.id)
print(run_name)
print(pipeline_filename)
print(arguments)
```


Run link <a href="/pipeline/#/runs/details/40972998-11a7-4536-a7be-d1138b468bb3" target="_blank" >here</a>


    eb8858e8-2f1a-4c1e-b275-999fede00a0a
    pipeline run
    pipeline.pipeline.zip
    {}


#  

# CHALLENGE
* The first challenge was to getting Google cloud account due to credit card issues which i eventually succeeded.
* Quota issues due to the fact that my GCP account is Free Tier.
* Outdated kubeflow documentations which made it so difficult for me to deploy kubeflow.
* The example code had a lots of bugs especially the pipeline components.yaml files.
* Nothing forgeting that i couldn't resolve the deploy component bug at time of report preparation.
* May sound funny but light issues was such a hell of experience for me


# RECOMMENDATIONS
* In next stage, i recommend that if there is going to be any project attached, let it be given out early at start of the stage. Unlike this stage where project was given 10 days to end of the stage.

* The bug in Deploy component seems to be a general one as kubeflow examples github have a lot of users complaining about the deploy component Internal runtime error. So i suggest we are given due consideration before grading.

* I recommend that collabration on slack channel should also be considered in grading. This will encourage Team Spirit, Oneness and above all Synergy.
