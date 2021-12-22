# Collaborative Filtering on netflix prize dataset
Paper referred https://arxiv.org/pdf/1301.7363.pdf



# If you are getting errors or not getting the output in PART 1 then try PART 2

## IMP: Please give the path till the dataset folder. The code takes the data from the train and test folder inside the path

> pip install -r requirements.txt
# -------------PART 1 a-------------
## Run the collaborative filtering code with train and test files as parameters (using all users for final ratings)
> python collaborative_filtering.py TrainingRatings.txt TestingRatings.txt

# same as above but considering top n similar users (here in the arguments it is 100) for final ratings with training and testing files
> python collaborative_filtering_top_n.py 100 TrainingRatings.txt TestingRatings.txt



# -------------PART 2-------------
# Steps to run the code... commands are tested in linux.. you can apply alternative commands for windows/MacOS
## Step 1 creating a virtual environment to run the code so that it does not conflicts with other instaled packages on the machine
> python3 -m venv my_env
## Step 2 if the above gives error then make sure your python version is 3.6 or above and install the venv package. If no error move to Step 3
	### for linux and MacOS
	> python3 -m pip install --user virtualenv
	### for windows
	> py -m pip install --user virtualenv

## Step 3 activate the environment
> source my_env/bin/activate
> pip install --upgrade pip

## Step 2 use requirements.txt file to install required packages
> pip install -r requirements.txt

After this you are good to use the python files and can run using the above commands specified

### once done with grading of the code you can deactivate the environment and delete it
> deactivate
> rm -r my_env

