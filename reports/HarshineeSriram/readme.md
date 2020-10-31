**Link to my proposal on Phabricator: https://phabricator.wikimedia.org/T266556**

**Table of Contents**

1. Candidate Details
2. Synopsis of the Project
3. Expected Internship Tasks
4. My method of approach (implementation details)
5. Proposed timeline (Complete)
6. Proposed timeline (Just the important parts)
7. Deliverables 
8. Work completed till date (Project contributions)
9. About Me
10. Past work experiences
11. Projects
12. Awards
13. Books

___

**Candidate Details**

**Name:** Harshinee Sriram <br>
**Email:** sriramharshinee@gmail.com | hsriram@cs.ubc.ca <br>
**Nationality:** Indian <br>
**Contact number:** +91 8369516382 <br>
**Website/Resume:** http://harshineesriram.github.io/ <br>
**LinkedIn:** https://www.linkedin.com/in/harshineesriram/ <br>
**GitHub:** https://github.com/HarshineeSriram <br>
**Location:** India (UTC +5:30) and, if my study permit is approved, Canada (UTC -7:00) <br>
**Typical working hours:** 12 PM to 8 PM at (UTC +5:30), and, if my study permit for Canada is approved, (UTC -7:00) <br>
___

**Synopsis of the project**

Wikimedia Commons is an online repository of free-use images, sounds, other media, and JSON files. Anyone can upload media to the Commons portal. The uploads are moderated by members and volunteers of the foundation manually. This project aims to build a classifier that can flag NSFW images/media for review. Upon successful completion of this internship, the intern would have designed, implemented and tested a machine learning model that would be able to classify the image and video media as SFW or NSFW with high accuracy. They would also be given the chance to deploy the model to Wikimedia test and production servers. Further, they would build a data processing pipeline and an API for the model.
Since this is a scratch project, applicants are required to do some research initially. A basic comparison of the existing NSFW classifiers along with their computational requirements is required. All applicants are expected to read various research papers and draw comparisons between them. They are expected to come up with a report detailing their research, the various options that can be used to implement the model and what they are proposing to do if they are selected. This report should also detail implementational methods and procedures.
Mentors: @abbasidaniyal and @chtnnh
___

**Expected Internship Tasks**

1)	Developing an image classifier for NSFW images [https://phabricator.wikimedia.org/T264049]: 
Design and implement a machine learning model which classifies images uploaded to Wikimedia Commons as Safe for Work (SFW) or Not Safe for Work (NSFW). Frameworks like Tensorflow / PyTorch can be used. The model should not be very computationally intensive and it should also be well tested.
2)	Creating an API for NSFW classification [https://phabricator.wikimedia.org/T264052]:  
An NSFW classifier model that takes an image as input and classifies it as SFW or NSFW. The model can be served using something like Tensorflow Serving (if built with TensorFlow). This task requires building a RESTful API for the machine learning model. The API can be built using frameworks like Flask / Django / FastAPI etc. The API will contain a simple endpoint that receives an image, passes it through the model and returns the NSFW classification result as well as the confidence.
3)	Video processing module for NSFW classifier  [https://phabricator.wikimedia.org/T264050]: 
An NSFW classifier takes an image as input and classifies it as SFW or NSFW.

This task aims at adding support for videos for the same model.
1.	Video Segmentation: Divide the video into difference scene segments using PySceneDetect or something similar.
2.	Frame Extraction: From the segmented video clips, extract individual frames using Decord or something similar.
3.	Feed the individual frames to the NSFW model and generate an overall score for the video.
___

**My method of approach (Implementation details)**

I believe that the 2 factors that make or break any classifier (irrespective of the algorithms it uses for its classification tasks), is the data that is available for it to train and validate on and the extent of scalability it can offer.  And in an NSFW classifier, the importance of these factors are heightened because:

- It needs support for images (from low resolution to very high resolution) and videos (SD to HD)
- This is essentially an anomaly detection problem where the two classes are A and -A (not A) where -A is U-A (everything in the universal set except A's elements). In other words, it's easy to determine what kind of  NSFW content should we be looking for but finding SFW (which could virtually be everything else) is trickier. 

For now, I would suggest transfer learning with personally constructed layers at the bottom. Based on my recent experiments, MobileNet (version 1) seems to come out on top because it is lightweight and provides a considerably high validation accuracy (96%). However, I will also try to experiment with one-class SVMs, Isolation forests, and Local Outlier Factor if the time permits, but that might mean increasing the dataset capacity further. I believe a vanilla binary classifier would be ideal. Multi-class NSFW (NSFW-artwork, NSFW-torture etc) would just add too much complexity, and there's hardly 1-2 datasets that contain good NSFW sub-categories.  If it is required that an image should have multiple annotations, not just NSFW or SFW, the binary classifier could act as an initial filter and then the native image is passed through Wikimedia's current annotation tool. 

As far as video classification goes, we could either use the image classifier that I will build or we could develop a model that trains on the Porn Dataset for videos (https://sites.google.com/site/pornographydatabase/) which includes clips that are blatantly NSFW (porn) as well as clips that are actually SFW but may be misclassified as NSFW (example: breastfeeding). The second method, of course, is bound by storage constraints for now.
___

**Proposed Timeline (complete)**

Since there are 3 major tasks to be completed in 12 weeks and the major task (around which the other 2 tasks revolve) is to develop a competitive model for image and video classification that can also withstand potential adversarial attacks. One additional task that goes hand-in-hand is developing an easy to understand documentation for further technology developments. 

| Date        | Task(s) Description |
| ------------- |:-------------:|
| December 1 2020	| • Blog prompt: "Introduce yourself" • Work on the dataset building along with the extent of NSFW content needed and the type of classifier to work with. • Address the video content filtration module by determining if it needs to work with the same dataset or a different one with the same framework. |
| December 7 2020	| • Finalize on the dataset and the method of approach for image and video classification and start working on the classifier details (such as extent of classification, different categories, dataset curation) •	Begin testing models and methods of approaches that would best fit the pre-determined image classifier characteristics and can be used for video classification as well. |
| December 8 2020	| • Introduction chat |
| December 11 2020 | • **INITIAL FEEDBACK DUE** |
| December 15 2020 | •	Blog prompt: "Everybody struggles"|
| December 22 2020 | •	Share something you’re stuck on |
|December 29 2020	 | •	Blog prompt: "Think about your audience" •	Finish building the preliminary model architectures that help with NSFW content detection (low computational power + high accuracy with low false-negative rate) • Start modifying the architecture to make it better equipped for robust classification |
| January 5 2021 | •	Share why you're passionate about your project or community|
| January 12 2021 | •	Mid-point project progress blog post  •	**MID-POINT FEEDBACK DUE** |
| January 19 2021 | • Share one "easy" project goal that took longer than expected |
| January 22 2021 | • Finish building the classifier.  • Start working on building the API |
| January 26 2021 | • Blog prompt: "Career opportunities" |
| February 2 2021 | • (Alums invited!) Advancing your career |
| February 9 2021 | • No blog post - interns work on their resume • Finish building an API that works with the Wikimedia servers • Develop the necessary documentation for future improvements/for understanding the method of approach | 
| February 16 2021 | • Networking skills |
| February 23 2021 | • Finish everything, model, API, and relevant documentation • Final project progress blog post |
| March 2 2021 | • Internship wrap up chat •	**FINAL FEEDBACK DUE** |
___

**Proposed Timeline (just the important parts)**

| Date        | Task(s) Description |
| ------------- |:-------------:|
| December 1 2020	| • Work on the dataset building along with the extent of NSFW content needed and the type of classifier to work with. • Address the video content filtration module by determining if it needs to work with the same dataset or a different one with the same framework. |
| December 7 2020 | • Finalize on the dataset and the method of approach for image and video classification and start working on the classifier details (such as the extent of classification, different categories, dataset curation) • Begin testing models and methods of approaches that would best fit the pre-determined image classifier characteristics and can be used for video classification as well. |
| December 29 2020 | • Finish building the preliminary model architectures that help with NSFW content detection (low computational power + high accuracy with low false-negative rate) • Start modifying the architecture to make it better equipped for robust classification |
| January 22 2021 | • Finish building the classifier.  • Start working on building the API |
| February 9 2021 | • Finish building an API that works with the Wikimedia servers • Develop the necessary documentation for future improvements/for understanding the method of approach |
| February 23 2021 | • Finish everything, model, API, and relevant documentation |


**Project deliverables**

1.	Initial Feedback: Comparison of all important classifier models to determine best model in terms of processing requirements
2.	Mid-point Feedback: Finish building the early stage model and begin with modifications to better suit our inputs and the kind of information we would like to show
3.	Final Feedback: Build final model, create an API for the same, deploy it, add relevant documentation for the same
___

**Work completed till date (Project contributions)**

**1)	Microtasks 1 and 2:**
I compare existing NSFW classifiers and also lookup datasets that I can use. In order to identify what is NSFW, the architecture needs to know how NSFW content looks like and develop a hypothesis that approximates the actual function which determines the result. It is not feasible to develop an entire dataset from scratch with labelling, and hence, I propose an amalgamation of existing extremely relevant datasets and web scraping.
Phabricator microtask descriptions:
Link: https://github.com/HarshineeSriram/Wikimedia-NSFW-Classifier-Reports/blob/main/reports/HarshineeSriram/NSFW_Classifier_for_Wikimedia_Harshinee_Sriram.ipynb  

**2)	Curating a preliminary dataset in order to determine what kind of an image classifier will work and to identify shortcomings:**

I decided to work with just one dataset for now, for trial purposes. I will add on to this dataset after testing architectures. I opted for the "NSFW Data Source URLs" developed by Data Scientist Evgeny Bazarov. This is a large and high-quality image dataset of sexually explicit images containing over 1.58 million data volumes in 159 categories.  I decided to construct a binary classifier (two outcomes: NSFW or not-NSFW).

Link: https://github.com/HarshineeSriram/Wikimedia-NSFW-Classifier-Reports/blob/main/reports/HarshineeSriram/1)%20How%20I%20curated%20a%20tentative%20dataset%20for%20finding%20the%20right%20model.ipynb 

**3)	Applying Transfer Learning on the Preliminary Curated Dataset:**
I chose transfer learning because of the number of training/validation samples I have. I have around 2400 training images (1200 for NSFW and 1200 for SFW), 800 validation images (400 for NSFW and 400 for SFW), and 800 test images, following the 60%-20%-20-% split.
I thought of approaching transfer learning because it is possible to get a high accuracy even with less number of examples to train/validate on. The model I used in this example is Inception V3.

Link: https://github.com/HarshineeSriram/Wikimedia-NSFW-Classifier-Reports/blob/main/reports/HarshineeSriram/2)%20Trying%20the%20Transfer%20Learning%20Approach%20-%20Inception%20V3.ipynb 

**4)	Improving my dataset and further improving my hyper-parameters:**

I add to the dataset but web scraping and NSFW art-specific dataset downloads. I probably had around 5000 images in each class at the end. I ran them through an image duplication detection script. After this, I ran the following script to make sure that all the final images are actually readable (this is because I received an error saying: cannot identify image file <_io.BytesIO object). This made me realize that if I want to increase the data in my datasets next time, I should not visit Shutterstock because almost all of their thumbnails are bytes files which are unreadable by PIL and hence cannot be trained on. 

This is my final dataset: 

NSFW Train: 3231 images
NSFW Validation: 1076 images
NSFW Test: 1066 images

SFW Train: 3140 images
SFW Validation:1048 images
SFW Test: 1060 images

I added the callbacks of "early stopping", "reduce LR", and model checkpoint (although I disabled the save_best_only for now). I also reduce the initial learning rate from 0.0001 to 0.00001. I only trained it for 20 epochs because this took me a while, but I am guessing that training for 30-40 epochs would be ideal. I was able to achieve an accuracy of 94.44% on the Validation set despite the huge increase in data. Now that I have a moderately good dataset that encompasses different types of NSFW as well as SFW media, I can use this for comparable performance. Next, I will be implementing the MobileNet architecture, which was my originally proposed model due to a less amount of hyper-parameters to regulate as well as due to the light-weight nature of the model (it's only a couple of MB huge). As of now, I am still finding solutions on the following drawbacks:
- Not enough training data for POC: the majority of the NSFW images that were scrapped are of Caucasian people
- Building a model to detect NSFW content in videos

Link: https://github.com/HarshineeSriram/Wikimedia-NSFW-Classifier-Reports/blob/main/reports/HarshineeSriram/3)%20Hyperparameter%20and%20Dataset%20updates%20for%20notebook%202.ipynb 

**5) Trying to improve the InceptionV3 model:**
This notebook builds on the previously modified InceptionV3 model, experimenting with new layers. I tried changing the architecture more this time. This includes increasing the number of trainable neurons and increasing the dropout layers. In this architecture, I add 3 fully trainable neuron layers and 2 dropout layers to the output layer. Even though this prevents overfitting to an extent, the final performance is similar to the previous architecture.

Link: https://github.com/HarshineeSriram/Wikimedia-NSFW-Classifier-Reports/blob/main/reports/HarshineeSriram/4)%20Architectural%20changes%20for%20Transfer%20Learning%20-%20InceptionV3.ipynb 

**6)	Implementing MobileNet architecture:**

In this notebook, I test out the model that I believe would be a good choice for the final preliminary model because of its characteristics of being the computationally relatively inexpensive and low amount of hyper-parameter tuning.  Even though this model is smaller than the InceptionV3, it already performs better with lower validation loss and higher validation accuracy (with a max val_accuracy of 96.62% after only 7 epochs) despite similar kinds of layers applied to both (this and the InceptionV3). Additionally, because this is more robust, the time to run each epoch was also lesser than the InceptionV3.

Link: https://github.com/HarshineeSriram/Wikimedia-NSFW-Classifier-Reports/blob/main/reports/HarshineeSriram/5)%20Implementing%20MobileNet%20model%20for%20transfer%20learning.ipynb 

**7)	Determining the percentage of NSFW and SFW content in a video:**
For this, I take inspiration from the libraries mentioned in the Phabricator task description: https://phabricator.wikimedia.org/T264050. I use the PySceneDetect library.
I use the following to extract frame with content-aware detection feature of PySceneDetect (which reduces the number of extracted frames by a huge margin, compared to a generic video frame extractor, and hence helps with faster classification).
scenedetect --input video1.mp4 detect-content list-scenes save-images 
After this, I loop through the extracted frames to determine the NSFW and SFW percentages.

Link: https://github.com/HarshineeSriram/Wikimedia-NSFW-Classifier-Reports/blob/main/reports/HarshineeSriram/6)%20NSFW%20content%20detection%20in%20videos.ipynb 

**8) Implementing the MobileNet_v2 architecture.**

In this notebook, based on the recommendation of Daniyal Abbasi, I have tested the MobileNet V2 architecture. I have used the same final layer architecture as the MobileNet (V1) to see if this model performs better or worse given the same data. From what I could infer, the performances of the MobileNet V1 and V2 are comparable in terms of Validation accuracies (96.62% in V1 vs 96.38% in V2), however, V2 sustains more validation losses (10.04% loss in V1 vs 12.51% loss in V2). So, I would rank the tested models as: MobileNet > MobileNet_V2 >> Inception_V3.

Link: https://github.com/HarshineeSriram/Wikimedia-NSFW-Classifier-Reports/blob/main/reports/HarshineeSriram/7)%20Implementing%20MobileNet_V2%20model%20for%20transfer%20learning.ipynb

**9) Testing random images from the internet.**

I thought of writing a script via Beautiful Soup to extract images but Google images seems to have changed its storage syntax twice in a span of a year and a half, so, for now, I'll just use a simple Chrome extension that downloads random images to see further improvements. The Chrome extension I downloaded: https://chrome.google.com/webstore/detail/image-downloader/cnpniohnfphhjihaiiggeabnkjhpaldj

Inferences
- Need more NSFW and SFW content which are monochrome and/or low lighting
- Need more NSFW content where the images are partially blurred (as in the second test image)
- Seems to detect NSFW content even if the people are not Caucasian (despite the dataset being predominantly Caucasian) - need to run more tests to determine
- Inception V3 and MobileNet classify the second last test image (Miley Cyrus) as SFW whereas MobileNet_V2 classifies it as NSFW. The extent of SFW content (which has to be discussed) will determine which output is more favourable

Link: https://github.com/HarshineeSriram/Wikimedia-NSFW-Classifier-Reports/blob/main/reports/HarshineeSriram/8)%20Testing%20random%20images%20from%20the%20internet.ipynb

**10) Deployment of Model for testing.**

I deployed my model via Flask. It is easy to use and assess the MobileNet model with any picture from the internet.
Link to the model.h5 file: https://drive.google.com/file/d/1OnbYjGu_Wj1TIw-4jpi0-i0Nk-ZClHtB/view?usp=sharing

Link: https://github.com/HarshineeSriram/Wikimedia-NSFW-Classifier-Reports/tree/main/reports/HarshineeSriram/keras-flask-image-classifier
___

**About Me**

Hi! I am Harshinee Sriram and I recently graduated from my undergrad at SRM Institute of Science and Technology, Kattankulathur and will begin my MSc Computer Science program at the University of British Columbia, Vancouver on a full scholarship sometime in 2021 (depending on the COVID 19 situation). I started learning and working in the area of Machine Learning and Deep Learning from 2018, and since then I have had the privilege to work with some outstanding researchers who helped cultivate my skills as a potential researcher in the Artificial Intelligence space.
___

**Past work experiences**

| Position        | Company/University | Dates | Tasks |
| :-------------: |:-------------:|:-------------:|:-------------:|
| **Visiting Research Scholar** | The University of Auckland, New Zealand | December 2019 - March 2020 | Conducted an extensive study on researcher development frameworks and initiatives in a number of research-intensive universities all around the world. Drew parallels on features and development solutions for research between Early Career Researchers (ECRs) and Post-Doctoral Researchers/Postgraduate Researchers (PGRs). Developed a comprehensive scoring system that takes into account broad categories of researcher development activities as well as the extent of contribution to their sub-categories. Performed Data Science techniques to draw inferences and laid out a conclusion on Researcher Development practices among countries, geographical regions, and research alliances. |
| **Remote Research Intern** | Indian Institute of Science (IISc Bengaluru), India | October 2019 - December 2019 | Was selected to be a part of a small team that works on the NeurIPS reproducibility challenge for the 2019 NeurIPS conference. Worked on real-time reinforcement learning - ablation track. |
|  **Artificial Intelligence Intern - Department of Automation** | Tech Mahindra Pvt. Ltd. Bengaluru, India | June 2019 - July 2019 |  Developed an API that converts speech-to-text reports for medical documents. Conducted unit testing of the generated reports to determine areas of improvement. Worked on Project Terumo by Terumo BCT (a US-based medical devices company) at the Offshore Development Centre (ODC). Developed an image recognition and classification modules for determining if the configurations of the medical equipment are ideal. |
| **Artificial Intelligence Intern** | National University of Singapore (NUS), Singapore |  December 2018 - January 2019 |  Worked on the Driver Distraction Detection problem. Used video feeds from cameras that focus on the driver's actions to develop a classifier model. Took into account different forms of distraction from media-related to passenger-related. Tested self-built CNNs, Transfer Learning techniques, and LSTMs for classification and evaluation. Used image filtering and feature extraction techniques using R and Python. Some libraries used in this stage were: dplyr and EBImage (for R) and Scikit-Learn (for Python). Worked on data management systems using Apache Hadoop. |
___

**Projects**

| Project Name        | Description |
| :-------------: |:-------------:|
| **Demenia** |  Developed a serious game that utilizes fuzzy logic and artificially intelligent navigation for implementing memory retention and improvement techniques for geriatric patients suffering from Dementia. |
| **Distracted Driver Detection** |  The major libraries utilized were dplyr and EBImage (for R) and Scikit Learn (for Python). The project uses various techniques and models to find out the most efficient model in distraction/fatigue detection. |
|  **Project TunnelEnd** |  Created machine learning models to identify places of high alert and growth rates from surveys conducted from the year 2005 to the year 2011, which contains information regarding state-wise statistics of civilian and veteran suicides based on gender, along with their probabilities. | 
| **Eunoia**|  This project uses Artificial Intelligence to act upon a 2014 Mental Health Survey in order to produce results regarding detection and requirement of treatment. I compare efficiency rates among Logistic Regression, K Neighbors Classifier, and Tree Classifier to obtain the best evaluation metric. |
___

**Awards**

| Award | Issuing body |
| :-------------: |:-------------:|
| International Tuition Award | University of British Columbia, Vancouver, Canada |
| Summer Scholarship for MS in Artificial Intelligence | Northeastern University, Boston, United States |
|  Engineering Silver Star Graduate Scholarship for MS in Computer Science | University of Texas, Arlington, United States | |
| Youngest Researcher Award | Women in Analytics 2019 Conference at Columbus, Ohio, United States |
| Student Travel Award |  Women in Analytics 2019 Conference at Columbus, Ohio, United States |
| Best Project Award for Eunoia | Institute of Engineering and Technology (IET) |
___

**Books**
1. Author of "The Horrors of Happiness" - opening week category bestseller on Amazon.
2. Co-Author of "Pandora's Box - Horn OK Please" by Karthik Iyengar
