Developer Dependencies
conda (>= 24.11.0)
conda-lock (>= 2.5.7)
Instructions for Adding New Dependencies
Open your terminal locally, direct to the root directory. Make sure you have conda and conda-lock installed on your local computer.

Create a conda environment called "ai_env" using the "conda-lock.yml" by running in your terminal:

conda-lock install --name ai_env conda-lock.yml
Activate the conda environment

conda activate ai_env
Use conda to install new packages (e.g., conda install {NEW-PACKAGE-NAME}). If you are installing a new package that is only available on PyPI (e.g., pip install {NEW-PACKAGE-NAME}), conda does not track pip-installed packages, you need to append a new "RUN" command to pip install that package (with version number; e.g., RUN pip install openai==1.57.0) at the end of the Dockerfile (living at the root of this directory).

At root directory, update environment.yml using:

conda env export --from-history > environment.yml 
Automatically append dependency version numbers to each of the packages you installed. The conda virtual environment I created is called "ai_env", if you are using another name, please change --env_name:

python scripts/00-update_enviroment_yml.py --root_dir="." --env_name="ai_env"
Use Conda-lock to solve and lock the updated environment. I'm using Linux-64 because that's the operating system of my docker image

conda-lock lock --file environment.yml
conda-lock -k explicit --file environment.yml -p linux-64