Hello!

This is my PyCharm Project + Documents used for the Machine Learning Final Project.

In the Documents Folder, you will find 4 files:

Final_Output: This is the output text that was generated the last time I ran the project

Britton_Stephen_Report:  I have provided this as a pages file as well as a docx and a pdf file.  This is the file that contains the answers to the 6 questions for the project.

This code is designed to run with the Python 3 interpreter.  I have removed the emails_by_address files because they add 100+mb to my file submission.

Thank you for your evaluation!

=========================================================================================
V2 - Changes Requested from Review:

There was an issue with running the poi_id.py file.

-SB- I have edited the file to include the import of sys as well as the append to the path variable.  Instead of only testing in pycharm, I have also tested with IDLE.  Screenshot will be available in documentation.  Hopefully this will resolve the run issues for you.

The report should include the following key characteristics:

The total number of data points
The number of POIs.

-SB- I have included both the data points and the number of POIs, you can find these in the report on pages 1-2

The report should state the name of the outliers that were removed.

-SB- I have included a section about the removed outliers on page 1

REQUIRED

The report should mention at least one new feature that was created. Also, explain the effect of this new feature on the final algorithm's performance. This can be done by training a simple classifier with and without the new features. The feature importance scores of all the features (new and existing) could also be provided to show the strength of the new features.

-SB- I have adjusted the feature that I created (this was a suggestion in the code review) and have included a section in the report about this new feature on page 3.  I have mentioned an impact of roughly .01 - .03 on the precision and recall of the classifier.

REQUIRED

It's not clear if the features were manually selected or if an automated process was used. If the features were manually tested, provide the performance achieved with several combinations. This way, we can see which gives the best results.

-SB- I have included in code a section that zips the score of the selectkbest algorithm along with the feature names.  I have also included the results in the report on pages 2 - 3.

REQUIRED

The report discusses the parameters tuned, but please state the specific names of these parameters for clarity purposes. Also, it would be good to include the various settings tested for each parameter.

-SB- I included the names of the features as well as some of the settings tested in the report on page 5.

REQUIRED

Explain what precision and recall measure in the context of the project task. For example, talk about these metrics in terms of the model's ability to predict POIs.

-SB- I have added a brief section explaining what precision and recall mean in terms of this project's task.  This can be found on page 7 at the end of the report.

REQUIRED

The report should specify the type of validation method performed.

-SB- I have adjusted the train_test_split in my code to instead use Stratified Shuffle Split instead based on the suggestion in the Code Section of my review.  I have also added an explanation about this cross validation in my report on page 6.

