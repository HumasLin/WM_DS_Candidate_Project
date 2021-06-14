Hi!

This is Haomin Lin, here I include my solution to the DS Candidate Project from WarnerMedia. This is a very interesting project, and I enjoyed it very much! Thank you for preparing it!

There are two Jupyter notebooks covering different part of the project. The first one `WM_project_EDA.ipynb`includes the Exploratory Data Analysis on the data, where I investigate how the data are distributed and also include some sanity check.

The second notebook `WM_project_modeling.ipynb` is mostly about predictive modeling, where I used different data pre-processing and implemeneted tree-based ensemble models in Machine Learning to make the prediction. After that, I used the best model to make prediction so that the dataset can be expanded. And I did some analysis based on the results. The best model trained is also included as `best_model.pkl`.

Thank you so much! Please let me know if you have any questions :)

All the best,

Haomin

-------

**Update on web application**

Based on the above two notebooks, I've created a web application based on `Streamlit`. It can be used to explore the analysis and modeling in this project.

Before you run the application, please make sure you have all the preset packages by running
```
pip install -r requirements.txt
```
in your shell (*pip3 for python3*).

Then, to run this application, simply go to the directory and run:
```
streamlit run app.py
```
If the website doesn't launch automatically, you may need to go to `http://localhost:8501` in your browser. With it lauched, you can easily interact with the application to see the results.
