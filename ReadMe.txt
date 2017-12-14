Hello!

This is my PyCharm Project + Documents used for the Machine Learning Final Project.

In the Documents Folder, you will find 4 files:

Final_Output: This is the output text that was generated the last time I ran the project

Britton_Stephen_Report:  I have provided this as a pages file as well as a docx and a pdf file.  This is the file that contains the answers to the 6 questions for the project.

This code is designed to run with the Python 3 interpreter.  I have removed the emails_by_address files because they add 100+mb to my file submission.

Thank you for your evaluation!

=========================================================================================
V3 - Changes Requested from Review:

REQUIRED

The poi_id.py threw the following error
***
--SB--
I am unable to re-create this error locally, I have tested in PyCharm and IDLE.  I worked with my UDACITY Mentor and he was unable to re-create this error, he successfully ran the code in BASH.

Can you please provide more information about this error so that I may be able to re-create it locally for a fix.  Can you provide the version of python and the versions of the packages used when testing.  Can you also include the data used during the evaluation in the return files?  When I break my code with a breakpoint, every value in my dataset is a float, I do not understand where non-float values are coming from when the tester runs the code.

REQUIRED

It's not clear if the features were manually selected or if an automated process was used. If the features were manually tested, provide the performance achieved with several combinations. This way, we can see which gives the best results.

--SB--
This is explained on page 2 and 3 of my report, I discuss the usage of automatic feature selection (SelectKBest) and list the output scoring for the top ranked features.  The code in poi_id.py for this process starts at line 159.

REQUIRED

The definitions for recall isn't entirely correct. For example, the report mentions that "The recall is the opposite, showing that the algorithm can indicate that the prediction matches the label in determining that a person is not a person of interest". Here's a link that may help with defining recall in the context of the project task: https://www.quora.com/What-is-the-best-way-to-understand-the-terms-precision-and-recall

--SB--

Thank you for the information on this one, I better understand precision and recall and have updated the end of my report to reflect this.

SUGGESTION

The report thoroughly discusses the various parameters that were tuned. But there's just a minor adjustment required in this section. The report refers to these parameters as features, which isn't correct and can be misleading. I recommend revising the report for clarity purposes.

--SB--

I have updated this section of the report to accurately term 'parameters' instead of 'features'


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

